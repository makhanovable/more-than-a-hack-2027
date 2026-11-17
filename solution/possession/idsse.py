import os
import re
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SOLUTION_DIR = os.path.dirname(HERE)
DATA_DIR = os.path.join(SOLUTION_DIR, "external", "idsse")
CACHE_DIR = os.path.join(SOLUTION_DIR, "cache")

MATCHES = {
    "J03WMX": "DFL-COM-000001",
    "J03WN1": "DFL-COM-000001",
    "J03WOH": "DFL-COM-000002",
    "J03WOY": "DFL-COM-000002",
    "J03WPY": "DFL-COM-000002",
    "J03WQQ": "DFL-COM-000002",
    "J03WR9": "DFL-COM-000002",
}
BASE_URL = "https://huggingface.co/datasets/pysport/idsse-data/resolve/main/"
SECTIONS = {b"firstHalf": 1, b"secondHalf": 2}

FRAMESET = re.compile(rb'<FrameSet GameSection="(\w+)" MatchId="[^"]*" TeamId="([^"]*)" '
                      rb'PersonId="([^"]*)">')
PLAYER_FRAME = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)"')
BALL_FRAME = re.compile(rb'<Frame N="(\d+)" T="[^"]*" X="([-\d.]+)" Y="([-\d.]+)" Z="([-\d.]+)"'
                        rb'[^>]*? BallPossession="(\d)" BallStatus="(\d)"')


def file_names(match_id):
    stem = "{}_DFL-MAT-{}.xml".format(MATCHES[match_id], match_id)
    return ("DFL_02_01_matchinformation_" + stem, "DFL_04_03_positions_raw_observed_" + stem)


def read_match_information(path):
    root = ET.parse(path).getroot()
    environment = next(root.iter("Environment"))
    side_of, keeper = {}, {}
    for team in root.iter("Team"):
        side = 0 if team.get("Role") == "home" else 1
        for player in team.iter("Player"):
            side_of[player.get("PersonId")] = side
            keeper[player.get("PersonId")] = player.get("PlayingPosition") == "TW"
    return {
        "pitch": (float(environment.get("PitchX")), float(environment.get("PitchY"))),
        "side_of": side_of, "keeper": keeper,
    }


def parse_match(match_id, data_dir=DATA_DIR):
    info_name, positions_name = file_names(match_id)
    info = read_match_information(os.path.join(data_dir, info_name))
    length, width = info["pitch"]
    with open(os.path.join(data_dir, positions_name), "rb") as handle:
        blob = handle.read()

    headers = list(FRAMESET.finditer(blob))
    chunks = {}
    for index, header in enumerate(headers):
        stop = headers[index + 1].start() if index + 1 < len(headers) else len(blob)
        section, team_id, person = header.group(1), header.group(2), header.group(3).decode()
        if section not in SECTIONS or team_id == b"referee":
            continue
        chunks.setdefault(SECTIONS[section], []).append(
            (team_id == b"BALL", person, blob[header.end():stop]))

    slots = sorted({person for half in chunks.values() for is_ball, person, _ in half
                    if not is_ball and person in info["side_of"]})
    slot_of = {person: i for i, person in enumerate(slots)}

    parts = []
    for half in sorted(chunks):
        ball = next(chunk for is_ball, _, chunk in chunks[half] if is_ball)
        table = np.array(BALL_FRAME.findall(ball)).astype(np.float64)
        first = int(table[:, 0].min())
        n = int(table[:, 0].max()) - first + 1
        row = table[:, 0].astype(np.int64) - first
        part = {
            "px": np.full((n, len(slots)), np.nan, dtype=np.float32),
            "py": np.full((n, len(slots)), np.nan, dtype=np.float32),
            "bx": np.full(n, np.nan, dtype=np.float32), "by": np.full(n, np.nan, dtype=np.float32),
            "bz": np.full(n, np.nan, dtype=np.float32),
            "owner": np.zeros(n, dtype=np.int8), "alive": np.zeros(n, dtype=bool),
            "half": np.full(n, half, dtype=np.int8),
        }
        part["bx"][row] = table[:, 1] + length / 2.0
        part["by"][row] = table[:, 2] + width / 2.0
        part["bz"][row] = table[:, 3]
        part["owner"][row] = (table[:, 4] != 1).astype(np.int8)
        part["alive"][row] = table[:, 5] == 1
        for is_ball, person, chunk in chunks[half]:
            if is_ball or person not in slot_of:
                continue
            frames = np.array(PLAYER_FRAME.findall(chunk)).astype(np.float64)
            if not len(frames):
                continue
            row = frames[:, 0].astype(np.int64) - first
            inside = (row >= 0) & (row < n)
            part["px"][row[inside], slot_of[person]] = frames[inside, 1] + length / 2.0
            part["py"][row[inside], slot_of[person]] = frames[inside, 2] + width / 2.0
        parts.append(part)

    out = {key: np.concatenate([part[key] for part in parts]) for key in parts[0]}
    out["team"] = np.array([info["side_of"][person] for person in slots], dtype=np.int8)
    out["is_gk"] = np.array([info["keeper"][person] for person in slots], dtype=bool)
    out["pitch"] = np.array(info["pitch"])
    return out


def load_match(match_id, data_dir=DATA_DIR, use_cache=True):
    cache = os.path.join(CACHE_DIR, "idsse_{}.npz".format(match_id))
    if use_cache and os.path.isfile(cache):
        with np.load(cache) as stored:
            return {key: stored[key] for key in stored.files}
    match = parse_match(match_id, data_dir)
    if use_cache:
        os.makedirs(CACHE_DIR, exist_ok=True)
        partial = os.path.join(CACHE_DIR, "idsse_{}.{}.part.npz".format(match_id, os.getpid()))
        np.savez(partial, **match)
        os.replace(partial, cache)
    return match


def _complete(path):
    if not os.path.isfile(path):
        return False
    with open(path, "rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - 200))
        return b"</PutDataRequest>" in handle.read()


def available(data_dir=DATA_DIR):
    return [match_id for match_id in MATCHES
            if all(_complete(os.path.join(data_dir, name)) for name in file_names(match_id))]
