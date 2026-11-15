import json
import os

import numpy as np

from . import adapt, features, idsse, metrica
from .gbdt import Binner, Booster

CONFIG = {
    "members": 5,
    "step": 5,
    "n_trees": 300,
    "depth": 4,
    "learning_rate": 0.1,
    "min_child_weight": 50.0,
    "column_fraction": 0.4,
    "row_fraction": 0.7,
    "max_bins": 32,
    "seeds": {"players_team": 11, "players_alive": 12, "ball_team": 13},
    "degrade_seed": 7,
    "short_weight": {"reference": 125.0, "low": 0.5, "high": 2.0},
}
TRAINED = os.path.join(idsse.SOLUTION_DIR, "trained", "models.npz")
TASKS = (
    ("players_team", "players", "team"),
    ("players_alive", "players", "alive"),
    ("ball_team", "ball", "team"),
)
ALL_MODELS = tuple(name for name, _, _ in TASKS)
PLAYERS_ONLY = tuple(name for name, source, _ in TASKS if source == "players")
WITH_BALL = tuple(name for name, source, _ in TASKS if source == "ball")


def available():
    return idsse.available()


def load_match(match_id):
    return metrica.load_match(match_id) if match_id in metrica.MATCHES \
        else idsse.load_match(match_id)


def number_of(match_id):
    return (list(idsse.MATCHES) + list(metrica.MATCHES)).index(match_id)


def tables(recording):
    players = features.players_table(recording)
    ball = features.ball_table(recording)
    return players.matrix(), ball.matrix(), players.swap_order(), ball.swap_order()


def with_ball(players, ball):
    return np.hstack([players, ball])


def opinion(members, recording, players=None):
    if players is None:
        players = features.players_table(recording).matrix()
    half = recording["half"]
    return adapt.steady_opinion(log_odds(members, "players_alive", players, half > 0), half)


def degraded(match_id, profile, seed, members):
    match = load_match(match_id)
    recording = adapt.degrade_players(match, profile, seed)
    return match, adapt.add_ball(recording, match, profile, opinion(members, recording), seed)


def possession_weight(match, config):
    setting = config.get("short_weight")
    n = len(match["alive"])
    weight = np.ones(n)
    if not setting:
        return weight
    alive = match["alive"].astype(bool)
    label = np.where(alive, match["owner"].astype(np.int64), 2) + 10 * match["half"].astype(np.int64)
    edges = np.flatnonzero(np.diff(label)) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [n]))
    length = np.repeat(stops - starts, stops - starts).astype(np.float64)
    scaled = np.clip(np.sqrt(setting["reference"] / length), setting["low"], setting["high"])
    return np.where(alive, scaled, 1.0)


def _entry(match_id, profile, seed, config, first=None):
    match = load_match(match_id)
    recording = adapt.degrade_players(match, profile, seed)
    table = features.players_table(recording)
    players = table.matrix()
    rows = np.arange(config["step"] // 2, len(players), config["step"])
    entry = {"players": players[rows], "swap_p": table.swap_order(),
             "home": match["owner"][rows] == 0, "alive": match["alive"][rows],
             "weight": possession_weight(match, config)[rows]}
    if first is not None:
        recording = adapt.add_ball(recording, match, profile,
                                   opinion(first, recording, players), seed)
        table = features.ball_table(recording)
        entry.update(ball=table.matrix()[rows], swap_b=table.swap_order(),
                     seen=np.isfinite(recording["bx"][rows]))
    return entry


def _fit(matrix, home, alive, swap, task, seed, config, weight=None):
    if task == "team":
        subset = matrix[alive]
        target = home[alive].astype(np.float64)
        both = np.vstack([subset, subset[:, swap]])
        target = np.concatenate([target, 1.0 - target])
        weight = None if weight is None else np.concatenate([weight[alive], weight[alive]])
    else:
        both = np.vstack([matrix, matrix[:, swap]])
        target = np.concatenate([alive, alive]).astype(np.float64)
        weight = None
    binner = Binner(config["max_bins"]).fit(both[::3])
    booster = Booster(n_trees=config["n_trees"], depth=config["depth"],
                      learning_rate=config["learning_rate"],
                      min_child_weight=config["min_child_weight"],
                      column_fraction=config["column_fraction"],
                      row_fraction=config["row_fraction"],
                      n_bins=config["max_bins"] + 1, seed=seed).fit(binner.transform(both), target,
                                                                    weight)
    return {"binner": binner, "booster": booster, "swap": swap, "task": task}


def _member(job):
    key, k, match_ids, profile, config, names, first = job
    need_ball = any(source == "ball" for name, source, _ in TASKS if name in names)
    entries = [_entry(match_id, profile,
                      config["degrade_seed"] + 1000 * k + number_of(match_id), config,
                      first if need_ball else None)
               for match_id in match_ids]
    home = np.concatenate([entry["home"] for entry in entries])
    alive = np.concatenate([entry["alive"] for entry in entries])
    weight = np.concatenate([entry["weight"] for entry in entries])
    players = np.vstack([entry["players"] for entry in entries])
    swap_p = entries[0]["swap_p"]
    if need_ball:
        seen = np.concatenate([entry["seen"] for entry in entries])
        ball = np.vstack([entry["ball"] for entry in entries])
        swap_all = np.concatenate([swap_p, len(swap_p) + entries[0]["swap_b"]])
    del entries

    member = {}
    for name, source, task in TASKS:
        if name not in names:
            continue
        seed = config["seeds"][name] + 10 * k
        if source == "players":
            member[name] = _fit(players, home, alive, swap_p, task, seed, config, weight)
        else:
            member[name] = _fit(with_ball(players[seen], ball[seen]), home[seen], alive[seen],
                                swap_all, task, seed, config, weight[seen])
    return key, k, member


def train_stage(groups, profile, config=CONFIG, jobs=1, names=ALL_MODELS, first=None,
                verbose=True):
    work = [(key, k, list(match_ids), profile, config, tuple(names),
             None if first is None else first[key])
            for key, match_ids in groups.items() for k in range(config["members"])]
    if verbose:
        print("  training {} model sets: {} ({} at a time) ...".format(
            len(work), ", ".join(names), jobs), flush=True)
    if jobs > 1 and len(work) > 1:
        import multiprocessing
        for match_id in sorted({m for job in work for m in job[2]}):
            load_match(match_id)
        with multiprocessing.get_context("spawn").Pool(min(jobs, len(work))) as pool:
            done = pool.map(_member, work, chunksize=1)
    else:
        done = [_member(job) for job in work]
    out = {key: [None] * config["members"] for key in groups}
    for key, k, member in done:
        out[key][k] = member
    return out


def merged(first, second):
    return [dict(a, **b) for a, b in zip(first, second)]


def save(members, match_ids, config=CONFIG, path=TRAINED):
    arrays = {"meta": np.array(json.dumps({
        "config": config, "matches": list(match_ids),
        "source": "DFL / IDSSE open dataset, doi:10.1038/s41597-025-04505-y, CC-BY 4.0",
    }, sort_keys=True))}
    for k, member in enumerate(members):
        for name, _, _ in TASKS:
            model, key = member[name], "m{}__{}__".format(k, name)
            edges = model["binner"].edges
            arrays[key + "edges"] = np.concatenate(edges)
            arrays[key + "edge_stops"] = np.cumsum([len(e) for e in edges]).astype(np.int64)
            booster = model["booster"]
            arrays[key + "features"] = np.stack(
                [np.concatenate(nodes) for nodes, _, _ in booster.trees]).astype(np.int32)
            arrays[key + "thresholds"] = np.stack(
                [np.concatenate(nodes) for _, nodes, _ in booster.trees]).astype(np.int32)
            arrays[key + "values"] = np.stack([values for _, _, values in booster.trees])
            arrays[key + "base"] = np.array(booster.base)
            arrays[key + "swap"] = np.asarray(model["swap"], dtype=np.int64)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load(path=TRAINED):
    tasks = {name: task for name, _, task in TASKS}
    with np.load(path, allow_pickle=False) as stored:
        meta = json.loads(str(stored["meta"]))
        config = meta["config"]
        level_stops = np.cumsum([1 << level for level in range(config["depth"])])[:-1]
        members = []
        for k in range(config["members"]):
            member = {}
            for name in tasks:
                key = "m{}__{}__".format(k, name)
                binner = Binner(config["max_bins"])
                binner.edges = np.split(stored[key + "edges"], stored[key + "edge_stops"][:-1])
                booster = Booster(n_trees=config["n_trees"], depth=config["depth"],
                                  learning_rate=config["learning_rate"],
                                  min_child_weight=config["min_child_weight"],
                                  column_fraction=config["column_fraction"],
                                  row_fraction=config["row_fraction"],
                                  n_bins=config["max_bins"] + 1,
                                  seed=config["seeds"][name] + 10 * k)
                booster.base = float(stored[key + "base"])
                booster.trees = [
                    (np.split(nodes.astype(np.int64), level_stops),
                     np.split(cuts.astype(np.int64), level_stops), values)
                    for nodes, cuts, values in zip(stored[key + "features"],
                                                   stored[key + "thresholds"],
                                                   stored[key + "values"])]
                member[name] = {"binner": binner, "booster": booster,
                                "swap": stored[key + "swap"], "task": tasks[name]}
            members.append(member)
    return members, meta


def _one(model, matrix, index):
    out = np.empty(len(index))
    sign = -1.0 if model["task"] == "team" else 1.0
    for start in range(0, len(index), 40000):
        chunk = matrix[index[start:start + 40000]]
        direct = model["booster"].decision_function(model["binner"].transform(chunk))
        mirrored = model["booster"].decision_function(
            model["binner"].transform(chunk[:, model["swap"]]))
        out[start:start + 40000] = 0.5 * (direct + sign * mirrored)
    return out


def log_odds(members, name, matrix, rows):
    index = np.flatnonzero(rows)
    out = np.full(len(matrix), np.nan)
    if len(index):
        out[index] = np.mean([_one(member[name], matrix, index) for member in members], axis=0)
    return out


def streams(members, players, ball, seen, rows=None):
    rows = np.ones(len(players), dtype=bool) if rows is None else rows
    return {
        "players_team": log_odds(members, "players_team", players, rows),
        "players_alive": log_odds(members, "players_alive", players, rows),
        "ball_team": log_odds(members, "ball_team", with_ball(players, ball), rows & seen),
    }
