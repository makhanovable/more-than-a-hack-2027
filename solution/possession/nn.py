import json
import os

import numpy as np

from . import adapt, features, idsse, models, validate

STEP = 5
TRAINED = os.path.join(idsse.SOLUTION_DIR, "trained", "seq_team.npz")

CONFIG = {
    "channels": 64,
    "dilations": [1, 2, 4, 8, 16, 32, 1, 2, 4, 8],
    "dropout": 0.1,
    "window": 1200,
    "batch": 12,
    "iterations": 2000,
    "learning_rate": 2e-3,
    "weight_decay": 1e-4,
    "edge": 80,
    "clip": 6.0,
    "copies": 3,
    "nets": 3,
    "seed": 31,
    "threads": 2,
}


def sequence_path(match_id, copy):
    return os.path.join(idsse.CACHE_DIR, "seq_{}_{}.npz".format(match_id, copy))


def build_sequence(job):
    match_id, copy, profile = job
    target = sequence_path(match_id, copy)
    if os.path.isfile(target):
        return target
    members, _ = models.load()
    config = models.CONFIG
    seed = config["degrade_seed"] + 1000 * copy + models.number_of(match_id)
    match = models.load_match(match_id)
    recording = adapt.degrade_players(match, profile, seed)
    table = features.players_table(recording)
    players = table.matrix()
    recording = adapt.add_ball(recording, match, profile,
                               models.opinion(members, recording, players), seed)
    ball_table = features.ball_table(recording)
    ball = ball_table.matrix()
    swap = np.concatenate([table.swap_order(), players.shape[1] + ball_table.swap_order()])
    alive = match["alive"].astype(bool) & (match["half"] > 0)
    partial = target[:-4] + ".{}.part.npz".format(os.getpid())
    np.savez(partial, x=np.hstack([players, ball]).astype(np.float32),
             home=(match["owner"] == 0), half=match["half"].astype(np.int8),
             weight=(models.possession_weight(match, config) * alive).astype(np.float32),
             swap=swap.astype(np.int64), n_players=np.array(players.shape[1]))
    os.replace(partial, target)
    return target


def standardise(x, net):
    finite = np.isfinite(x)
    z = np.clip((np.where(finite, x, 0.0) - net["mean"]) / net["std"], -net["clip"], net["clip"])
    z = z * finite
    n_players = int(net["n_players"])
    seen = finite[:, n_players:].any(axis=1)
    whole = finite[:, :n_players].all(axis=1)
    return np.concatenate([z, seen[:, None], whole[:, None]], axis=1).astype(np.float32)


def forward(net, z):
    with np.errstate(all="ignore"):
        h = z @ net["inp_w"].T + net["inp_b"]
        for w, b, d in zip(net["blk_w"], net["blk_b"], net["dilations"]):
            a = np.maximum(h, 0.0)
            before = np.zeros_like(a)
            after = np.zeros_like(a)
            if len(a) > d:
                before[d:] = a[:-d]
                after[:-d] = a[d:]
            h = h + before @ w[:, :, 0].T + a @ w[:, :, 1].T + after @ w[:, :, 2].T + b
        out = (np.maximum(h, 0.0) @ net["out_w"] + net["out_b"]).astype(np.float64)
    if not np.isfinite(out).all():
        raise FloatingPointError("the network produced a value that is not finite")
    return out


def predict(trained, players, ball, half):
    x = np.hstack([players, ball])
    out = np.full(len(x), np.nan)
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        first, last = rows[0], rows[-1] + 1
        for phase in range(STEP):
            index = np.arange(first + phase, last, STEP)
            part = x[index]
            mirrored = part[:, trained["swap"]]
            total = np.zeros(len(index))
            for net in trained["nets"]:
                total += 0.5 * (forward(net, standardise(part, net))
                                - forward(net, standardise(mirrored, net)))
            out[index] = total / len(trained["nets"])
    return out


def save(nets, swap, meta, path=TRAINED):
    arrays = {"meta": np.array(json.dumps(meta, sort_keys=True)), "swap": np.asarray(swap),
              "count": np.array(len(nets))}
    for k, net in enumerate(nets):
        for key in ("mean", "std", "inp_w", "inp_b", "out_w", "out_b", "n_players", "clip"):
            arrays["n{}__{}".format(k, key)] = np.asarray(net[key])
        arrays["n{}__dilations".format(k)] = np.asarray(net["dilations"])
        for j, (w, b) in enumerate(zip(net["blk_w"], net["blk_b"])):
            arrays["n{}__blk{}_w".format(k, j)] = w
            arrays["n{}__blk{}_b".format(k, j)] = b
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load(path=TRAINED):
    with np.load(path, allow_pickle=False) as stored:
        nets = []
        for k in range(int(stored["count"])):
            net = {key: stored["n{}__{}".format(k, key)]
                   for key in ("mean", "std", "inp_w", "inp_b", "out_w", "out_b")}
            net["n_players"] = int(stored["n{}__n_players".format(k)])
            net["clip"] = float(stored["n{}__clip".format(k)])
            net["dilations"] = [int(d) for d in stored["n{}__dilations".format(k)]]
            net["blk_w"] = [stored["n{}__blk{}_w".format(k, j)] for j in range(len(net["dilations"]))]
            net["blk_b"] = [stored["n{}__blk{}_b".format(k, j)] for j in range(len(net["dilations"]))]
            nets.append(net)
        return {"nets": nets, "swap": stored["swap"], "meta": json.loads(str(stored["meta"]))}


def stream_path(match_id):
    return os.path.join(idsse.CACHE_DIR, "nn_seq_{}.npy".format(match_id))


def write_stream(match_id, trained):
    data = validate.load(match_id)
    recording = {"px": data["px"].astype(np.float64), "py": data["py"].astype(np.float64),
                 "bx": data["bx"].astype(np.float64), "by": data["by"].astype(np.float64),
                 "team": data["team"], "outfield": ~data["is_gk"], "is_gk": data["is_gk"],
                 "half": data["half"], "pitch": data["pitch"]}
    players, ball, _, _ = models.tables(recording)
    target = stream_path(match_id)
    np.save(target, predict(trained, players, ball, data["half"]))
    return target


def blend(boosted, network, weight):
    return (1.0 - weight) * boosted + weight * np.where(np.isfinite(network), network, boosted)
