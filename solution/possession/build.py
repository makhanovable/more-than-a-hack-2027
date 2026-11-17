import hashlib
import os
import subprocess
import sys

import numpy as np

from . import adapt, dataio, decode, download, idsse, labelling, metrica, models, nn, pipeline, \
    tune, validate

SOLUTION = idsse.SOLUTION_DIR
OUTPUT = os.path.join(SOLUTION, "output", "submission.csv")
VALIDATOR = os.path.join(dataio.ROOT, "evaluation", "validate_submission.py")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_external():
    wrong = 0
    with open(os.path.join(SOLUTION, "external_data.sha256"), "r") as handle:
        expected = [line.split(None, 1) for line in handle.read().splitlines() if line.strip()]
    for digest, name in expected:
        path = os.path.join(SOLUTION, "external", name.strip())
        if not os.path.isfile(path):
            print("  MISSING    {}".format(name.strip()))
            wrong += 1
        elif sha256(path) != digest:
            print("  DIFFERENT  {}".format(name.strip()))
            wrong += 1
    print("external data: {} of {} files match external_data.sha256".format(
        len(expected) - wrong, len(expected)))
    return wrong


def step_download():
    print("DFL / IDSSE dataset (CC-BY 4.0, doi:10.1038/s41597-025-04505-y) ...", flush=True)
    print("  {} matches".format(len(download.fetch_all())))
    print("Metrica Sports sample data (github.com/metrica-sports/sample-data) ...", flush=True)
    print("  {} matches".format(len(metrica.fetch_all())))
    return 1 if verify_external() else 0


def step_boosting(jobs):
    tactical = dataio.load_atd("tactical")
    ids = models.available()
    everything = {"all": ids}
    print("1/5  profile of the competition's tracking: smoothness, players tracked (positions "
          "only) ...", flush=True)
    profile = adapt.measure(tactical)
    print("2/5  players-only models on the {} DFL matches ...".format(len(ids)), flush=True)
    players_only = models.train_stage(everything, profile, models.CONFIG, jobs,
                                      models.PLAYERS_ONLY)
    print("3/5  profile, continued: when the tracker sees the ball (against the players-only "
          "model's opinion of where the ball is in play) ...", flush=True)
    live = pipeline.live_estimate(players_only["all"])
    profile = adapt.save(dict(profile, **adapt.measure_visibility(tactical, live)))
    graded = dataio.graded_mask()
    seen = np.isfinite(tactical["bx"])
    print("  in play, by the model, on {:.1f}% of the graded frames; ball seen on {:.1f}% of "
          "those and on {:.1f}% of the rest".format(
              100.0 * live[graded].mean(), 100.0 * seen[graded & live].mean(),
              100.0 * seen[graded & ~live].mean()), flush=True)
    print("4/5  the model that reads the ball, on the same matches ...", flush=True)
    with_ball = models.train_stage(everything, profile, models.CONFIG, jobs, models.WITH_BALL,
                                   players_only)
    models.save(models.merged(players_only["all"], with_ball["all"]), ids)
    members, _ = models.load()
    print("5/5  the Metrica matches, which no model has seen, read by those models ...",
          flush=True)
    validate.unseen(members, profile)


def sequences(match_ids, copies, profile, jobs):
    work = [(match_id, copy, profile) for match_id in match_ids for copy in range(copies)]
    work = [job for job in work if not os.path.isfile(nn.sequence_path(job[0], job[1]))]
    if not work:
        return
    print("  building {} training sequences ...".format(len(work)), flush=True)
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(min(jobs, len(work))) as pool:
            pool.map(nn.build_sequence, work, chunksize=1)
    else:
        for job in work:
            nn.build_sequence(job)


def step_network(jobs):
    from . import nn_train
    profile = adapt.load()
    ids = models.available()
    print("1/4  training sequences from the {} DFL matches ...".format(len(ids)), flush=True)
    sequences(ids, nn.CONFIG["copies"], profile, jobs)
    print("2/4  the networks ...", flush=True)
    nets, swap = nn_train.train_many({"all": ids}, nn.CONFIG, jobs)["all"]
    nn.save(nets, swap, {"config": nn.CONFIG, "matches": ids,
                         "source": "DFL / IDSSE open dataset, doi:10.1038/s41597-025-04505-y"})
    trained = nn.load()
    print("3/4  the Metrica matches, which no model has seen, read by the networks ...", flush=True)
    for match_id in metrica.MATCHES:
        if os.path.isfile(validate.path_of(match_id)):
            nn.write_stream(match_id, trained)
    print("4/4  decoder settings, chosen on the Metrica matches ...", flush=True)
    tune.tune(jobs=jobs)


def write(path, codes):
    path = dataio.write_submission(path, codes)
    print("\nwrote {}".format(os.path.relpath(path, dataio.ROOT)))
    check = subprocess.run([sys.executable, "-B", VALIDATOR, path], capture_output=True, text=True)
    print("validate_submission.py: {}".format(
        check.stdout.strip().splitlines()[0] if check.stdout.strip() else check.stderr.strip()))
    if check.returncode != 0:
        print(check.stdout)
        sys.exit(1)
    return path


def summary(codes, details, path):
    graded = dataio.graded_mask()
    scene, params = details["scene"], details["params"]
    seen = np.isfinite(scene["ball"]["bx"])
    proven = labelling.ball_proofs(scene)[0] & graded
    holds = [episode for episode in details["episodes"]
             if episode["kept"] and graded[episode["start"]]]
    return [
        "file    {}".format(os.path.relpath(path, dataio.ROOT)),
        "sha256  {}".format(sha256(path)),
        "",
        "decoder settings (trained/decoder.json): " + ", ".join(
            "{} {:g}".format(key, params[key]) for key in sorted(params)),
        "",
    ] + pipeline.describe(codes) + [
        "ball seen by the tracker on {:.1f}% of the graded frames".format(
            100.0 * seen[graded].mean()),
        "",
        "what the rules decided",
        "  ball off the pitch or lying unattended: {} frames in {} stretches".format(
            int(proven.sum()), len(decode.runs_of(proven))),
        "  goalkeeper holds: {} ({} frames)".format(len(holds), int(details["hold"].sum())),
        "  penalty kicks: {}".format(len(details["penalties"])),
        "  set-piece tags: {}; {} had no stoppage next to them and were given one".format(
            len(details["set_pieces"]), len(details["repaired"])),
    ]


def step_predict(out=OUTPUT):
    codes, details = pipeline.run()
    path = write(out, codes)
    print()
    print("\n".join(summary(codes, details, path)))
    return path
