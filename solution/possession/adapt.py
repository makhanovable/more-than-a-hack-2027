import json
import os

import numpy as np

from . import dataio

PROFILE = os.path.join(dataio.SOLUTION_DIR, "trained", "target_profile.json")

SPEED_HALF_WINDOW = 3
SPEED_CAP = 12.0
MIN_PLAYERS = 6
SPEED_BINS = (0.6, 1.0, 1.4, 1.8, 2.5)
MOVING_BALL = 2.0
MAX_SMOOTH = 16
PRIORITY_BLOCK = 250
MAX_COUNT = 12
OPINION_HALF_WIDTH = 25
EDGE = 75
N_PHASES = 6
MIN_CELL = 1000


def runs_of(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def smooth(values, half_width):
    if half_width <= 0:
        return values
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    pad = [(1, 0)] + [(0, 0)] * (values.ndim - 1)
    total = np.cumsum(np.pad(filled, pad), axis=0)
    count = np.cumsum(np.pad(finite.astype(np.float64), pad), axis=0)
    n = len(values)
    index = np.arange(n)
    low = np.maximum(0, index - half_width)
    high = np.minimum(n, index + half_width + 1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (total[high] - total[low]) / (count[high] - count[low])
    return np.where(finite, out, np.nan)


def median_speed(px, py):
    k = SPEED_HALF_WINDOW
    vx = np.full(px.shape, np.nan)
    vy = np.full(px.shape, np.nan)
    dt = 2.0 * k / dataio.FPS
    vx[k:-k] = (px[2 * k:] - px[:-2 * k]) / dt
    vy[k:-k] = (py[2 * k:] - py[:-2 * k]) / dt
    speed = np.hypot(vx, vy)
    speed[speed > SPEED_CAP] = np.nan
    out = np.full(len(px), np.nan)
    enough = np.isfinite(speed).sum(axis=1) >= MIN_PLAYERS
    with np.errstate(all="ignore"):
        out[enough] = np.nanmedian(speed[enough], axis=1)
    return out


def speed_class(vmed):
    return np.where(np.isfinite(vmed), np.digitize(np.nan_to_num(vmed), SPEED_BINS),
                    len(SPEED_BINS) + 1).astype(np.int64)


def ball_acceleration(bx, by):
    vx, vy = np.diff(bx), np.diff(by)
    acceleration = np.hypot(np.diff(vx), np.diff(vy))
    speed = np.hypot(vx, vy)[1:] * dataio.FPS
    use = np.isfinite(acceleration) & (speed > MOVING_BALL)
    return float(np.percentile(acceleration[use], 90))


def player_jitter(px, py):
    ax = px[2:] - 2.0 * px[1:-1] + px[:-2]
    ay = py[2:] - 2.0 * py[1:-1] + py[:-2]
    size = np.hypot(ax, ay)
    return float(np.median(size[np.isfinite(size)]))


def measure(tactical):
    graded = dataio.graded_mask()
    px = tactical["px"].astype(np.float64) * dataio.PITCH_LENGTH
    py = tactical["py"].astype(np.float64) * dataio.PITCH_WIDTH
    bx = tactical["bx"].astype(np.float64) * dataio.PITCH_LENGTH
    by = tactical["by"].astype(np.float64) * dataio.PITCH_WIDTH
    pair = graded[1:] & graded[:-1]

    present = np.isfinite(px)
    change = np.zeros((MAX_COUNT + 1, MAX_COUNT + 1))
    for side in (dataio.HOME, dataio.AWAY):
        count = np.minimum(present[:, tactical["slot_team"] == side].sum(axis=1), MAX_COUNT)
        np.add.at(change, (count[:-1][pair], count[1:][pair]), 1.0)
    change = change + change.T
    count_share = change.sum(axis=1) / change.sum()
    rows = change.sum(axis=1, keepdims=True)
    change = np.where(rows > 0, change / np.maximum(rows, 1.0), np.eye(MAX_COUNT + 1))
    return {
        "ball_acceleration": ball_acceleration(bx[graded], by[graded]),
        "player_jitter": player_jitter(px[graded], py[graded]),
        "count_share": count_share, "count_change": change,
    }


def steady_opinion(logit, half):
    out = np.zeros(len(logit), dtype=bool)
    for number in (1, 2):
        rows = half == number
        if rows.any():
            out[rows] = smooth(np.nan_to_num(logit[rows], nan=-1.0), OPINION_HALF_WIDTH) > 0.0
    return out


def phases(live, half):
    live = np.asarray(live, dtype=bool)
    phase = np.full(len(live), -1, dtype=np.int64)
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        first, last = rows[0], rows[-1] + 1
        for state in (False, True):
            for start, stop in runs_of(live[first:last] == state):
                length = stop - start
                edge = min(EDGE, length // 2)
                part = np.full(length, 4 if state else 1, dtype=np.int64)
                part[:edge] -= 1
                part[length - edge:] += 1
                phase[first + start:first + stop] = part
    return phase


def measure_visibility(tactical, live):
    half = np.zeros(dataio.N_FRAMES, dtype=np.int8)
    for number, first, last in dataio.HALVES:
        half[first:last + 1] = number
    px = tactical["px"].astype(np.float64) * dataio.PITCH_LENGTH
    py = tactical["py"].astype(np.float64) * dataio.PITCH_WIDTH
    seen = np.isfinite(tactical["bx"])
    classes = speed_class(median_speed(px, py))
    n_classes = len(SPEED_BINS) + 2
    phase = phases(live, half)

    def rate(values, default):
        return float(values.mean()) if len(values) else default

    keep = np.zeros((N_PHASES, n_classes))
    regain = np.zeros((N_PHASES, n_classes))
    share = np.zeros((N_PHASES, n_classes))
    pair = (half[1:] > 0) & (half[1:] == half[:-1])
    for p in range(N_PHASES):
        here = pair & (phase[1:] == p)
        whole = (rate(seen[1:][here & seen[:-1]], 1.0), rate(seen[1:][here & ~seen[:-1]], 0.0),
                 rate(seen[phase == p], 0.0))
        for c in range(n_classes):
            cell = here & (classes[1:] == c)
            was_seen, was_lost = cell & seen[:-1], cell & ~seen[:-1]
            if cell.sum() >= MIN_CELL and was_seen.any() and was_lost.any():
                keep[p, c] = seen[1:][was_seen].mean()
                regain[p, c] = seen[1:][was_lost].mean()
                share[p, c] = seen[(phase == p) & (classes == c)].mean()
            else:
                keep[p, c], regain[p, c], share[p, c] = whole
    return {"keep": keep, "regain": regain, "seen_share": share}


def save(profile, path=PROFILE):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plain = {key: (np.asarray(value).tolist() if np.ndim(value) else float(value))
             for key, value in profile.items()}
    with open(path, "w") as handle:
        json.dump(plain, handle, indent=1, sort_keys=True)
        handle.write("\n")
    return load(path)


def load(path=PROFILE):
    with open(path, "r") as handle:
        plain = json.load(handle)
    return {key: (np.array(value) if isinstance(value, list) else value)
            for key, value in plain.items()}


def matching_smooth(statistic, target, tracks):
    gaps = [abs(np.log(statistic(*[smooth(track, half_width) for track in tracks]) / target))
            for half_width in range(MAX_SMOOTH + 1)]
    return int(np.argmin(gaps))


def count_series(n, profile, rng):
    cumulative = np.cumsum(profile["count_change"], axis=1)
    draws = rng.random_sample(n)
    series = np.empty(n, dtype=np.int64)
    state = int(np.searchsorted(np.cumsum(profile["count_share"]), rng.random_sample()))
    last = cumulative.shape[1] - 1
    rows = [row.tolist() for row in cumulative]
    for t, draw in enumerate(draws.tolist()):
        row = rows[min(state, last)]
        state = 0
        while state < last and row[state] < draw:
            state += 1
        series[t] = state
    return series


def hide_players(px, py, team, profile, rng):
    n = len(px)
    target = np.stack([count_series(n, profile, rng) for _ in (0, 1)], axis=1)
    px, py = px.copy(), py.copy()
    for side in (0, 1):
        slots = np.flatnonzero(team == side)
        for start in range(0, n, PRIORITY_BLOCK):
            stop = min(n, start + PRIORITY_BLOCK)
            priority = rng.permutation(len(slots))
            present = np.isfinite(px[start:stop][:, slots])
            excess = present.sum(axis=1) - target[start:stop, side]
            rank = np.where(present, priority[None, :], len(slots) + 1)
            position = rank.argsort(axis=1).argsort(axis=1)
            hide = present & (position < excess[:, None])
            block_x, block_y = px[start:stop][:, slots], py[start:stop][:, slots]
            block_x[hide] = np.nan
            block_y[hide] = np.nan
            px[start:stop, slots] = block_x
            py[start:stop, slots] = block_y
    return px, py


def visibility(classes, phase, profile, rng):
    n = len(classes)
    draws = rng.random_sample(n).tolist()
    inside = phase >= 0
    row = np.where(inside, phase, 0)
    keep = profile["keep"][row, classes].tolist()
    regain = profile["regain"][row, classes].tolist()
    first = profile["seen_share"][row, classes].tolist()
    seen = np.zeros(n, dtype=bool)
    state, started = False, False
    for t, is_inside in enumerate(inside.tolist()):
        if not is_inside:
            started = False
            continue
        if started:
            state = draws[t] < (keep[t] if state else regain[t])
        else:
            state, started = draws[t] < first[t], True
        seen[t] = state
    return seen


def smoothing_of(match, profile):
    px, py = match["px"].astype(np.float64), match["py"].astype(np.float64)
    bx, by = match["bx"].astype(np.float64), match["by"].astype(np.float64)
    return (matching_smooth(ball_acceleration, profile["ball_acceleration"], (bx, by)),
            matching_smooth(player_jitter, profile["player_jitter"], (px, py)))


def degrade_players(match, profile, seed, smoothing=None):
    rng = np.random.RandomState(seed)
    ball_width, player_width = smoothing if smoothing is not None else smoothing_of(match, profile)
    px = smooth(match["px"].astype(np.float64), player_width)
    py = smooth(match["py"].astype(np.float64), player_width)
    px, py = hide_players(px, py, match["team"], profile, rng)
    return {
        "px": px, "py": py, "team": match["team"], "outfield": ~match["is_gk"],
        "is_gk": match["is_gk"], "half": match["half"], "pitch": match["pitch"],
        "bx": np.full(len(px), np.nan), "by": np.full(len(px), np.nan),
        "ball_smooth": ball_width, "player_smooth": player_width,
    }


def add_ball(recording, match, profile, live, seed):
    rng = np.random.RandomState([seed, 1])
    bx = smooth(match["bx"].astype(np.float64), recording["ball_smooth"])
    by = smooth(match["by"].astype(np.float64), recording["ball_smooth"])
    classes = speed_class(median_speed(recording["px"], recording["py"]))
    seen = visibility(classes, phases(live, recording["half"]), profile, rng)
    return dict(recording, bx=np.where(seen, bx, np.nan), by=np.where(seen, by, np.nan))
