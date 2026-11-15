import json
import os
import sys

import numpy as np

from . import idsse, labelling, metrica, nn, scenes, validate

EVALUATION_DIR = os.path.join(os.path.dirname(idsse.SOLUTION_DIR), "evaluation")
if EVALUATION_DIR not in sys.path:
    sys.path.insert(0, EVALUATION_DIR)

import possession_metrics as pm

DECODER = os.path.join(idsse.SOLUTION_DIR, "trained", "decoder.json")

GRID = {
    "alive_clip": (3.0, 6.0),
    "alive_penalty": (20.0, 30.0, 45.0, 60.0, 90.0, 135.0, 200.0),
    "dead_lead": (0, 8, 15, 22, 30, 38, 45, 55),
    "team_clip": (2.0, 4.0),
    "team_penalty": (1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 18.0, 27.0),
    "end_factor": (0.0, 1.0, 2.0, 4.0, 6.0, 10.0, 15.0),
    "seq_weight": (0.0, 0.25, 0.5, 0.75, 1.0),
}
TIE = 0.002
RULES_ON_METRICA = ("penalty",)


def score(codes, truth, half):
    result = pm.evaluate(scenes.split_halves(truth, half), scenes.split_halves(codes, half))
    return result["spell_f1_mean"], result["macro_f1"]


def external_scene(match_id):
    scene, truth = scenes.external(validate.load(match_id))
    return scene, truth, scene["team_logit"].copy(), np.load(nn.stream_path(match_id))


def _combination(job):
    alive_clip, alive_penalty, dead_lead, match_ids, grid = job
    loaded = [external_scene(match_id) for match_id in match_ids]
    base = {"alive_clip": alive_clip, "alive_penalty": alive_penalty, "dead_lead": dead_lead}
    states = [labelling.alive_stage(scene, base, use=RULES_ON_METRICA)
              for scene, _, _, _ in loaded]
    rows = []
    for weight in grid["seq_weight"]:
        for scene, _, boosted, network in loaded:
            scene["team_logit"] = nn.blend(boosted, network, weight)
        for team_clip in grid["team_clip"]:
            for team_penalty in grid["team_penalty"]:
                for end_factor in grid["end_factor"]:
                    params = dict(base, team_clip=team_clip, team_penalty=team_penalty,
                                  end_factor=end_factor, seq_weight=weight)
                    scores = [score(labelling.team_stage(scene, state, params)[0], truth,
                                    scene["half"])
                              for (scene, truth, _, _), state in zip(loaded, states)]
                    rows.append((params, scores))
    return rows


def table(match_ids, grid=GRID, jobs=1):
    work = [(alive_clip, alive_penalty, dead_lead, list(match_ids), grid)
            for alive_clip in grid["alive_clip"] for alive_penalty in grid["alive_penalty"]
            for dead_lead in grid["dead_lead"]]
    if jobs > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(min(jobs, len(work))) as pool:
            parts = pool.map(_combination, work, chunksize=1)
    else:
        parts = [_combination(job) for job in work]
    rows = []
    for part in parts:
        for params, scores in part:
            scores = np.array(scores)
            rows.append((params, float(scores[:, 0].mean()), float(scores[:, 1].mean()),
                         scores.tolist()))
    return rows


def plateau_of(rows):
    best = max(row[1] for row in rows)
    return [row for row in rows if row[1] >= best - TIE]


def choose(rows, grid=GRID):
    plateau = plateau_of(rows)
    middle = {}
    for key, values in grid.items():
        steps = sorted(values.index(row[0][key]) for row in plateau)
        middle[key] = steps[(len(steps) - 1) // 2]

    def distance(row):
        return sum(abs(grid[key].index(row[0][key]) - middle[key]) for key in grid)

    nearest = min(distance(row) for row in plateau)
    return max((row for row in plateau if distance(row) == nearest),
               key=lambda row: (row[1], row[2]))


def left_out(rows, chooser=choose):
    out = []
    for held in range(len(rows[0][3])):
        rest = [i for i in range(len(rows[0][3])) if i != held]
        others = [(params, float(np.mean([scores[i][0] for i in rest])),
                   float(np.mean([scores[i][1] for i in rest])), scores)
                  for params, _, _, scores in rows]
        params, _, _, scores = chooser(others)
        out.append((params, scores[held][0], scores[held][1]))
    return out


def tune(jobs=1, verbose=True):
    match_ids = [m for m in metrica.MATCHES if os.path.isfile(nn.stream_path(m))]
    rows = table(match_ids, jobs=jobs)
    params, spell, macro, per_match = choose(rows)
    plateau = plateau_of(rows)
    held_out = left_out(rows)
    without = [row for row in rows if row[0]["seq_weight"] == 0.0]
    record = {
        "params": params, "tuned_on": match_ids,
        "metric": "mean over the tuning matches of spell_f1_mean; the middle of the plateau of "
                  "combinations within {} of the best".format(TIE),
        "spell_f1": spell, "macro_f1": macro,
        "per_match": {m: s for m, s in zip(match_ids, per_match)},
        "best_spell_f1": max(row[1] for row in rows),
        "plateau_size": len(plateau), "combinations": len(rows),
        "left_out": {m: {"params": p, "spell_f1": s, "macro_f1": f}
                     for m, (p, s, f) in zip(match_ids, held_out)},
        "left_out_mean": [float(np.mean([s for _, s, _ in held_out])),
                          float(np.mean([f for _, _, f in held_out]))],
        "best_by_seq_weight": {str(w): max(row[1] for row in rows if row[0]["seq_weight"] == w)
                               for w in GRID["seq_weight"]},
        "plateau_by_seq_weight": {str(w): sum(1 for row in plateau if row[0]["seq_weight"] == w)
                                  for w in GRID["seq_weight"]},
        "best_by_team_penalty": {str(v): max(row[1] for row in rows if row[0]["team_penalty"] == v)
                                 for v in GRID["team_penalty"]},
        "boosted_alone_best": max(row[1] for row in without),
        "grid": {key: list(values) for key, values in GRID.items()},
    }
    os.makedirs(os.path.dirname(DECODER), exist_ok=True)
    with open(DECODER, "w") as handle:
        json.dump(record, handle, indent=2, sort_keys=True)
        handle.write("\n")
    if verbose:
        print("  decoder settings: {}".format(params))
        print("  the plateau: {} of {} combinations; best spellF1 {:.4f}".format(
            len(plateau), len(rows), record["best_spell_f1"]))
        print("  best spellF1 by the network's weight: " + "  ".join(
            "{}: {:.4f}".format(w, s) for w, s in record["best_by_seq_weight"].items()))
        print("  Metrica, with these settings: spellF1 {:.4f}  macroF1 {:.4f}".format(spell, macro))
        print("  Metrica, each match with settings chosen on the other two: "
              "spellF1 {:.4f}  macroF1 {:.4f}".format(*record["left_out_mean"]))
    return record, rows


def load():
    with open(DECODER, "r") as handle:
        return json.load(handle)["params"]
