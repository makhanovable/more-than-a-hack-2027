import warnings

import numpy as np

FPS = 25
SPEED_HALF_WINDOW = 3
SPEED_CAP = 12.0
FAR = 40.0
HORIZON = 250
OFFSETS = (-100, -50, -25, 25, 50, 100)


def shift(values, frames):
    out = np.full(values.shape, np.nan, dtype=np.float64)
    if frames > 0:
        out[:-frames] = values[frames:]
    elif frames < 0:
        out[-frames:] = values[:frames]
    else:
        out[:] = values
    return out


def velocity(x, y, half_window=SPEED_HALF_WINDOW):
    vx = np.full(x.shape, np.nan)
    vy = np.full(x.shape, np.nan)
    dt = 2.0 * half_window / FPS
    vx[half_window:-half_window] = (x[2 * half_window:] - x[:-2 * half_window]) / dt
    vy[half_window:-half_window] = (y[2 * half_window:] - y[:-2 * half_window]) / dt
    return vx, vy


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


def masked_mean(values, mask):
    count = mask.sum(axis=1)
    total = np.where(mask, values, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(count > 0, total / count, np.nan)


def masked_std(values, mask):
    mean = masked_mean(values, mask)
    squares = masked_mean((values - mean[:, None]) ** 2, mask)
    return np.sqrt(squares)


def since(event):
    index = np.where(event, np.arange(len(event)), -10 ** 9)
    return np.minimum(np.arange(len(event)) - np.maximum.accumulate(index), HORIZON)


def until(event):
    return since(event[::-1])[::-1]


def team_matrix(team, shape):
    team = np.asarray(team)
    return team if team.ndim == 2 else np.broadcast_to(team, shape)


def home_attacks_positive(recording):
    px = recording["px"]
    team = team_matrix(recording["team"], px.shape)
    outfield = recording["outfield"][None, :] & np.isfinite(px)
    out = np.ones(len(px), dtype=bool)
    for half in (1, 2):
        rows = recording["half"] == half
        if not rows.any():
            continue
        home = px[rows][outfield[rows] & (team[rows] == 0)]
        away = px[rows][outfield[rows] & (team[rows] == 1)]
        if len(home) and len(away):
            out[rows] = home.mean() < away.mean()
    return out


class Columns:
    def __init__(self):
        self.names, self.values, self.partner = [], [], {}

    def both(self, name, home, away):
        self.partner["home_" + name] = "away_" + name
        self.partner["away_" + name] = "home_" + name
        self.names += ["home_" + name, "away_" + name]
        self.values += [home, away]

    def one(self, name, value):
        self.partner[name] = name
        self.names.append(name)
        self.values.append(value)

    def matrix(self):
        return np.stack([np.asarray(v, dtype=np.float32) for v in self.values], axis=1)

    def swap_order(self):
        index = {name: i for i, name in enumerate(self.names)}
        return np.array([index[self.partner[name]] for name in self.names], dtype=np.int64)


def players_table(recording):
    length, width = float(recording["pitch"][0]), float(recording["pitch"][1])
    px, py = recording["px"].astype(np.float64), recording["py"].astype(np.float64)
    team = team_matrix(recording["team"], px.shape)
    seen = np.isfinite(px) & recording["outfield"][None, :]
    home_positive = home_attacks_positive(recording)

    vx, vy = velocity(px, py)
    speed = np.sqrt(vx * vx + vy * vy)
    bad = ~(speed <= SPEED_CAP)
    moving = seen & ~bad
    ax, ay = velocity(np.where(bad, np.nan, vx), np.where(bad, np.nan, vy), 5)
    accel = np.sqrt(ax * ax + ay * ay)

    columns = Columns()
    blocks = {}
    for name, code in (("home", 0), ("away", 1)):
        mine = seen & (team == code)
        forward = home_positive if code == 0 else ~home_positive
        own_x = np.where(forward[:, None], px, length - px)
        own_vx = np.where(forward[:, None], vx, -vx)
        mine_moving = mine & moving

        ordered = np.sort(np.where(mine, own_x, np.nan), axis=1)
        count = mine.sum(axis=1)
        reverse = np.sort(np.where(mine, -own_x, np.nan), axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            deepest2 = np.nanmean(ordered[:, :2], axis=1)
            deepest4 = np.nanmean(ordered[:, :4], axis=1)
            advanced2 = -np.nanmean(reverse[:, :2], axis=1)
            advanced3 = -np.nanmean(reverse[:, :3], axis=1)
        y_high = np.where(mine, py, -np.inf).max(axis=1)
        y_low = np.where(mine, py, np.inf).min(axis=1)
        in_box = np.abs(py - width / 2.0) < 20.16
        blocks[name] = {
            "centroid": masked_mean(own_x, mine),
            "spread_x": masked_std(own_x, mine),
            "spread_y": masked_std(py, mine),
            "depth": advanced2 - deepest2,
            "width": np.where(count > 0, y_high - y_low, np.nan),
            "back_line": deepest4,
            "front_line": advanced3,
            "speed": masked_mean(speed, mine_moving),
            "slow": masked_mean((speed < 1.0).astype(np.float64), mine_moving),
            "fast": masked_mean((speed > 4.0).astype(np.float64), mine_moving),
            "forward": masked_mean(own_vx, mine_moving),
            "accel": masked_mean(accel, mine & np.isfinite(accel)),
            "count": count.astype(np.float64),
            "in_own_box": (mine & (own_x < 16.5) & in_box).sum(axis=1).astype(np.float64),
            "in_far_box": (mine & (own_x > length - 16.5) & in_box).sum(axis=1).astype(np.float64),
        }
    with np.errstate(invalid="ignore"):
        for block in blocks.values():
            block["forward_change"] = shift(block["forward"], 10) - shift(block["forward"], -10)
    for key in blocks["home"]:
        columns.both(key, blocks["home"][key], blocks["away"][key])

    everyone = seen & moving
    columns.one("all_speed", masked_mean(speed, everyone))
    columns.one("all_slow", masked_mean((speed < 1.0).astype(np.float64), everyone))
    columns.one("all_accel", masked_mean(accel, seen & np.isfinite(accel)))
    columns.one("overlap", blocks["home"]["centroid"] + blocks["away"]["centroid"] - length)

    for key in ("centroid", "spread_x", "spread_y", "depth", "forward", "speed", "back_line"):
        for offset in OFFSETS:
            columns.both("{}_{:+d}".format(key, offset),
                         shift(blocks["home"][key], offset), shift(blocks["away"][key], offset))
    for key in ("all_speed", "all_slow"):
        base = columns.values[columns.names.index(key)]
        for offset in OFFSETS:
            columns.one("{}_{:+d}".format(key, offset), shift(base, offset))
    return columns


def ball_table(recording, ball_smooth=0):
    length, width = float(recording["pitch"][0]), float(recording["pitch"][1])
    px, py = recording["px"].astype(np.float64), recording["py"].astype(np.float64)
    team = team_matrix(recording["team"], px.shape)
    bx = smooth(recording["bx"].astype(np.float64), ball_smooth)
    by = smooth(recording["by"].astype(np.float64), ball_smooth)
    seen_ball = np.isfinite(bx)
    home_positive = home_attacks_positive(recording)
    n = len(bx)
    rows = np.arange(n)

    pvx, pvy = velocity(px, py)
    bvx, bvy = velocity(bx, by)
    ball_speed = np.sqrt(bvx * bvx + bvy * bvy)
    k = 4
    change = np.full(n, np.nan)
    change[k:-k] = np.sqrt((bvx[2 * k:] - bvx[:-2 * k]) ** 2 + (bvy[2 * k:] - bvy[:-2 * k]) ** 2)

    distance = np.sqrt((px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2)
    distance = np.where(np.isfinite(distance), distance, np.inf)

    columns = Columns()
    blocks = {}
    for name, code in (("home", 0), ("away", 1)):
        mine = np.where(team == code, distance, np.inf)
        ordered = np.sort(mine, axis=1)
        nearest = mine.argmin(axis=1)
        d1 = np.where(seen_ball, np.minimum(ordered[:, 0], FAR), np.nan)
        d2 = np.where(seen_ball, np.minimum(ordered[:, 1], FAR), np.nan)
        dvx = bvx - pvx[rows, nearest]
        dvy = bvy - pvy[rows, nearest]
        player_speed = np.sqrt(pvx[rows, nearest] ** 2 + pvy[rows, nearest] ** 2)
        forward = home_positive if code == 0 else ~home_positive
        blocks[name] = {
            "d1": d1,
            "d2": d2,
            "rel": np.where(seen_ball, np.sqrt(dvx * dvx + dvy * dvy), np.nan),
            "player_speed": np.where(seen_ball, player_speed, np.nan),
            "closing": shift(d1, 5) - shift(d1, -5),
            "within2": np.where(seen_ball, (mine <= 2.0).sum(axis=1), np.nan),
            "within5": np.where(seen_ball, (mine <= 5.0).sum(axis=1), np.nan),
            "within10": np.where(seen_ball, (mine <= 10.0).sum(axis=1), np.nan),
            "ball_from_goal": np.where(forward, bx, length - bx),
            "ball_forward": np.where(forward, bvx, -bvx),
            "since_touch": np.where(seen_ball, since(d1 <= 1.0), np.nan),
            "until_touch": np.where(seen_ball, until(d1 <= 1.0), np.nan),
        }
    for key in blocks["home"]:
        columns.both(key, blocks["home"][key], blocks["away"][key])
    for key in ("d1", "rel"):
        for offset in (-50, -25, -12, -6, 6, 12, 25, 50):
            columns.both("{}_{:+d}".format(key, offset),
                         shift(blocks["home"][key], offset), shift(blocks["away"][key], offset))

    columns.one("ball_speed", ball_speed)
    columns.one("ball_change", change)
    columns.one("ball_to_touchline", np.minimum(by, width - by))
    columns.one("ball_to_goal_line", np.minimum(bx, length - bx))
    columns.one("ball_seen", seen_ball.astype(np.float64))
    columns.one("since_seen", since(seen_ball))
    columns.one("until_seen", until(seen_ball))
    for offset in (-25, -12, 12, 25):
        columns.one("ball_speed_{:+d}".format(offset), shift(ball_speed, offset))
    return columns
