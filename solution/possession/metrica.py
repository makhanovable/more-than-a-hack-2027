import io
import json
import os
import shutil
import subprocess
import urllib.request
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(SOLUTION_DIR, "external", "metrica")
CACHE_DIR = os.path.join(SOLUTION_DIR, "cache")

BASE_URL = "https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/"
FILES = {
    "MS1": ("Sample_Game_1/Sample_Game_1_RawTrackingData_Home_Team.csv",
            "Sample_Game_1/Sample_Game_1_RawTrackingData_Away_Team.csv",
            "Sample_Game_1/Sample_Game_1_RawEventsData.csv"),
    "MS2": ("Sample_Game_2/Sample_Game_2_RawTrackingData_Home_Team.csv",
            "Sample_Game_2/Sample_Game_2_RawTrackingData_Away_Team.csv",
            "Sample_Game_2/Sample_Game_2_RawEventsData.csv"),
    "MS3": ("Sample_Game_3/Sample_Game_3_tracking.txt",
            "Sample_Game_3/Sample_Game_3_metadata.xml",
            "Sample_Game_3/Sample_Game_3_events.json"),
}
MATCHES = tuple(FILES)
PITCH = (105.0, 68.0)
OWNING = ("SET PIECE", "PASS", "CARRY", "SHOT", "RECOVERY", "BALL LOST", "BALL OUT",
          "FAULT RECEIVED")
KEEPER_DEPTH = 0.33
GAP_FRAMES = 50
RESTART_SLACK = 5
MAX_STOPPAGE = 7500


def _is_restart(event):
    return event[1] == "SET PIECE" or (event[1] == "PASS" and "GOAL KICK" in event[4])


def _local(name):
    return os.path.join(DATA_DIR, os.path.basename(name))


def available():
    return [match_id for match_id in MATCHES
            if all(os.path.isfile(_local(name)) and os.path.getsize(_local(name)) > 0
                   for name in FILES[match_id])]


def fetch_all(verbose=True):
    os.makedirs(DATA_DIR, exist_ok=True)
    for match_id in MATCHES:
        for name in FILES[match_id]:
            target = _local(name)
            if os.path.isfile(target) and os.path.getsize(target) > 0:
                continue
            if verbose:
                print("  {} ...".format(os.path.basename(name)), flush=True)
            partial = target + ".part"
            if shutil.which("curl"):
                subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "3", "-o", partial,
                                BASE_URL + name], check=True)
            else:
                with urllib.request.urlopen(BASE_URL + name) as response, \
                        open(partial, "wb") as handle:
                    shutil.copyfileobj(response, handle, 1 << 20)
            os.replace(partial, target)
    return available()


def _table(path):
    with open(path, "rb") as handle:
        lines = handle.read().decode("ascii").splitlines()
    names = [name for name in lines[2].split(",")[3:] if name]
    body = "\n".join(lines[3:]).replace("NaN", "nan")
    values = np.loadtxt(io.StringIO(body), delimiter=",", dtype=np.float64)
    return names, values


def _csv_match(match_id):
    home_path, away_path, events_path = (_local(name) for name in FILES[match_id])
    home_names, home = _table(home_path)
    away_names, away = _table(away_path)
    if len(home) != len(away) or not np.array_equal(home[:, 1], away[:, 1]):
        raise ValueError("{}: the two tracking files do not cover the same frames".format(match_id))
    frames = home[:, 1].astype(np.int64)
    x = np.hstack([home[:, 3:-2:2], away[:, 3:-2:2]])
    y = np.hstack([home[:, 4:-2:2], away[:, 4:-2:2]])
    team = np.array([0] * (len(home_names) - 1) + [1] * (len(away_names) - 1), dtype=np.int8)
    ball = home[:, -2:]

    events = []
    with open(events_path, "r") as handle:
        header = handle.readline().strip().split(",")
        column = {name: index for index, name in enumerate(header)}
        for line in handle:
            fields = line.rstrip("\n").split(",")
            events.append((int(fields[column["Start Frame"]]), fields[column["Type"]],
                           0 if fields[column["Team"]] == "Home" else 1,
                           int(fields[column["End Frame"]]), fields[column["Subtype"]]))
    return frames, home[:, 0].astype(np.int8), x, y, team, None, ball, events


def _epts_match(match_id):
    tracking_path, meta_path, events_path = (_local(name) for name in FILES[match_id])
    root = ET.parse(meta_path).getroot()
    teams = [team.get("id") for team in root.iter("Team")]
    slots, team, keeper = {}, [], []
    for player in root.iter("Player"):
        slots[player.get("id")] = len(slots)
        team.append(teams.index(player.get("teamId")))
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        keeper.append((params.get("position_type") or "").lower() == "goalkeeper")
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    limits = {p.findtext("Name"): p.findtext("Value") for p in root.iter("ProviderParameter")}

    with open(tracking_path, "rb") as handle:
        lines = handle.read().decode("ascii").splitlines()
    n = len(lines)
    frames = np.arange(1, n + 1, dtype=np.int64)
    x = np.full((n, len(slots)), np.nan)
    y = np.full((n, len(slots)), np.nan)
    ball = np.full((n, 2), np.nan)
    for spec in root.iter("DataFormatSpecification"):
        first, last = int(spec.get("startFrame")), int(spec.get("endFrame"))
        order = [slots[channel[ref.get("playerChannelId")]]
                 for ref in spec.iter("PlayerChannelRef")][::2]
        body = "\n".join(lines[first - 1:last]).replace(":", ",").replace(";", ",") \
            .replace("NaN", "nan")
        values = np.loadtxt(io.StringIO(body), delimiter=",", dtype=np.float64, ndmin=2)
        if values.shape[1] != 1 + 2 * len(order) + 2 or values[0, 0] != first:
            raise ValueError("{}: frames {}..{} do not match their specification".format(
                match_id, first, last))
        x[first - 1:last, order] = values[:, 1:-2:2]
        y[first - 1:last, order] = values[:, 2:-2:2]
        ball[first - 1:last] = values[:, -2:]

    period = np.zeros(n, dtype=np.int8)
    period[int(limits["first_half_start"]) - 1:int(limits["first_half_end"])] = 1
    period[int(limits["second_half_start"]) - 1:int(limits["second_half_end"])] = 2
    with open(events_path, "r") as handle:
        events = [(int(event["start"]["frame"]), event["type"]["name"],
                   teams.index(event["team"]["id"]), int(event["end"]["frame"]),
                   (event["subtypes"] or {}).get("name", "") if isinstance(event["subtypes"], dict)
                   else " ".join(sub["name"] for sub in (event["subtypes"] or [])))
                  for event in json.load(handle)["data"]]
    return frames, period, x, y, np.array(team, dtype=np.int8), np.array(keeper), ball, events


def parse_match(match_id):
    parse = _epts_match if FILES[match_id][0].endswith(".txt") else _csv_match
    frames, period, x, y, team, keeper, ball, events = parse(match_id)
    inside = period > 0
    frames, period, x, y, ball = frames[inside], period[inside], x[inside], y[inside], ball[inside]

    if keeper is None:
        with np.errstate(all="ignore"):
            depth = np.nanmedian(np.abs(x - 0.5), axis=0)
        keeper = np.nan_to_num(depth, nan=0.0) > KEEPER_DEPTH

    owner = np.zeros(len(frames), dtype=np.int8)
    index_of = {int(frame): index for index, frame in enumerate(frames)}
    changes = [(index_of[event[0]], event[2]) for event in sorted(events, key=lambda e: e[0])
               if event[1] in OWNING and event[0] in index_of]
    if changes:
        owner[:changes[0][0]] = changes[0][1]
        for (start, side), (stop, _) in zip(changes, changes[1:] + [(len(frames), 0)]):
            owner[start:stop] = side

    alive = np.isfinite(ball[:, 0])
    restarts = np.array(sorted(index_of[event[0]] for event in events
                               if _is_restart(event) and event[0] in index_of))
    edges = np.flatnonzero(np.diff(np.concatenate(([0], (~alive).astype(np.int8), [0]))))
    for start, stop in zip(edges[::2], edges[1::2]):
        nearest = np.searchsorted(restarts, stop - RESTART_SLACK)
        restarted = nearest < len(restarts) and restarts[nearest] <= stop + RESTART_SLACK
        if stop - start > GAP_FRAMES or restarted or start == 0 or stop == len(alive) \
                or period[start - 1] != period[stop]:
            continue
        alive[start:stop] = True
        for axis in (0, 1):
            ball[start:stop, axis] = np.interp(np.arange(start, stop), [start - 1, stop],
                                               [ball[start - 1, axis], ball[stop, axis]])

    ordered = sorted(events, key=lambda e: e[0])
    for position, event in enumerate(ordered):
        if not _is_restart(event) or event[0] not in index_of:
            continue
        before = [other for other in ordered[:position]
                  if other[0] < event[0] and other[1] != "CARD" and not _is_restart(other)]
        if not before:
            continue
        last = before[-1]
        stopped = max(last[0], last[3])
        if stopped in index_of and period[index_of[stopped]] == period[index_of[event[0]]] \
                and 0 < event[0] - stopped <= MAX_STOPPAGE:
            alive[index_of[stopped] + 1:index_of[event[0]]] = False

    used = np.isfinite(x).any(axis=0)
    return {
        "px": (x[:, used] * PITCH[0]).astype(np.float32),
        "py": (y[:, used] * PITCH[1]).astype(np.float32),
        "team": team[used], "is_gk": np.asarray(keeper, dtype=bool)[used],
        "bx": (ball[:, 0] * PITCH[0]).astype(np.float32),
        "by": (ball[:, 1] * PITCH[1]).astype(np.float32),
        "bz": np.full(len(frames), np.nan, dtype=np.float32),
        "owner": owner, "alive": alive, "half": period, "pitch": np.array(PITCH),
    }


def load_match(match_id, use_cache=True):
    cache = os.path.join(CACHE_DIR, "metrica_{}.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    match = parse_match(match_id)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        partial = os.path.join(CACHE_DIR, "metrica_{}.{}.part.npz".format(match_id, os.getpid()))
        np.savez(partial, **match)
        os.replace(partial, cache)
    return match
