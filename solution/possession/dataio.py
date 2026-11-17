import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(HERE)
ROOT = os.path.dirname(SOLUTION_DIR)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import games

CACHE_DIR = os.path.join(SOLUTION_DIR, "cache")

HOME, AWAY = 0, 1
FPS = games.GAMES[games.GAME]["fps"]
N_FRAMES = games.GAMES[games.GAME]["n_video_frames"]
PITCH_LENGTH, PITCH_WIDTH = games.GAMES[games.GAME]["pitch"]

HALVES = games.half_video_blocks(games.GAME)


def read_atd_meta(meta_path):
    root = ET.parse(meta_path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2:
        raise ValueError("{}: expected 2 teams, found {}".format(meta_path, len(teams)))
    home_id = teams[0].get("id")

    slot_team, slot_is_gk = [], []
    for player in root.iter("Player"):
        slot_team.append(HOME if player.get("teamId") == home_id else AWAY)
        params = {p.findtext("Name"): p.findtext("Value") for p in player.iter("ProviderParameter")}
        slot_is_gk.append((params.get("position_type") or "").lower() == "goalkeeper")

    directions = []
    for team in teams:
        params = {p.findtext("Name"): p.findtext("Value") for p in team.iter("ProviderParameter")}
        directions.append(params.get("attack_direction_first_half"))

    return {
        "team_names": [team.findtext("Name") for team in teams],
        "slot_team": np.array(slot_team, dtype=np.int8),
        "slot_is_gk": np.array(slot_is_gk, dtype=bool),
        "attack_direction_first_half": directions,
    }


def parse_atd_raw(raw_path, n_slots):
    with open(raw_path, "rb") as handle:
        blob = handle.read()
    blob = blob.replace(b"\r", b"").replace(b":", b",").replace(b";", b",").replace(b"\n", b",")
    values = np.array(blob.decode("ascii").rstrip(",").split(","), dtype=np.float64)

    per_line = 1 + 3 * n_slots + 3
    if len(values) % per_line:
        raise ValueError("{}: {} values is not a multiple of {} per line".format(
            raw_path, len(values), per_line))
    table = values.reshape(-1, per_line)
    frame_ids = table[:, 0].astype(np.int64)
    players = table[:, 1:1 + 3 * n_slots].reshape(-1, n_slots, 3).astype(np.float32)
    ball = table[:, 1 + 3 * n_slots:].astype(np.float32)
    return frame_ids, players, ball


def load_atd(source="tactical", use_cache=True):
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"], games.GAME, None, source)
    meta = read_atd_meta(meta_path)
    n_slots = len(meta["slot_team"])
    n_frames = games.source_frames(source)

    cache_path = os.path.join(CACHE_DIR, "atd_{}.npz".format(source))
    if use_cache and os.path.isfile(cache_path) \
            and os.path.getmtime(cache_path) >= os.path.getmtime(raw_path):
        cached = np.load(cache_path)
        players, ball = cached["players"], cached["ball"]
    else:
        frame_ids, players, ball = parse_atd_raw(raw_path, n_slots)
        if len(frame_ids) != n_frames or not np.array_equal(frame_ids, np.arange(1, n_frames + 1)):
            raise ValueError("{}: frame ids are not 1..{}".format(raw_path, n_frames))
        if use_cache:
            os.makedirs(CACHE_DIR, exist_ok=True)
            partial = os.path.join(CACHE_DIR, "atd_{}.{}.part.npz".format(source, os.getpid()))
            np.savez(partial, players=players, ball=ball)
            os.replace(partial, cache_path)

    return {
        "px": players[:, :, 0], "py": players[:, :, 1], "ps": players[:, :, 2],
        "bx": ball[:, 0], "by": ball[:, 1], "bs": ball[:, 2],
        "slot_team": meta["slot_team"], "slot_is_gk": meta["slot_is_gk"], "meta": meta,
    }


def load_smart_tagging():
    path = games.require(["smart_tagging"], games.GAME, None, "tactical")[0]
    meta_path = games.require(["atd_meta"], games.GAME, None, "tactical")[0]
    home_name = read_atd_meta(meta_path)["team_names"][0].strip().lower()

    out = []
    for instance in ET.parse(path).iter("instance"):
        tags = {label.findtext("group"): label.findtext("text")
                for label in instance.findall("label")}
        team_text = (tags.get("Team") or "").strip().lower()
        if team_text in ("", "n/a"):
            team = -1
        elif team_text == home_name:
            team = HOME
        else:
            team = AWAY
        out.append({
            "code": (instance.findtext("code") or "").strip().lower(),
            "start": int(round(float(instance.findtext("start")) * FPS)),
            "end": int(round(float(instance.findtext("end")) * FPS)),
            "team": team,
            "half": tags.get("Half"),
            "type": (tags.get("Type") or "").strip().lower(),
            "tags": tags,
        })
    out.sort(key=lambda event: (event["start"], event["end"]))
    return out


LABELS = np.array(["H", "A", "D", "X"])
L_H, L_A, L_D, L_X = 0, 1, 2, 3


def write_submission(path, codes):
    folder = os.path.dirname(path)
    if folder and not os.path.isdir(folder):
        os.makedirs(folder)
    labels = LABELS[np.asarray(codes, dtype=np.int64)]
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))
    return path


def graded_mask(n_frames=N_FRAMES):
    mask = np.zeros(n_frames, dtype=bool)
    for _, first, last in HALVES:
        mask[first:last + 1] = True
    return mask
