import numpy as np

from . import adapt, features, labelling, penalty

SPEED_HALF_WINDOW = 3


def velocity(x, y):
    k = SPEED_HALF_WINDOW
    vx = np.full(x.shape, np.nan)
    vy = np.full(x.shape, np.nan)
    dt = 2.0 * k / features.FPS
    vx[k:-k] = (x[2 * k:] - x[:-2 * k]) / dt
    vy[k:-k] = (y[2 * k:] - y[:-2 * k]) / dt
    return vx, vy


def ball_view(bx, by):
    bvx, bvy = velocity(bx, by)
    return {"bx": bx, "by": by, "bvx": bvx, "bvy": bvy, "bspeed": np.hypot(bvx, bvy)}


def nearest_player(px, py, bx, by):
    with np.errstate(invalid="ignore"):
        distance = np.hypot(px - bx[:, None], py - by[:, None])
    return np.where(np.isfinite(distance), distance, np.inf).min(axis=1)


def left_team(recording):
    home_positive = features.home_attacks_positive(recording)
    return np.where(recording["half"] > 0, np.where(home_positive, 0, 1), -1).astype(np.int8)


def team_logit(streams):
    return np.where(np.isfinite(streams["ball_team"]), streams["ball_team"],
                    streams["players_team"])


def precompute(scene):
    scene["proofs"] = labelling.ball_proofs(scene)
    scene["kicks"] = penalty.detect(scene["px"], scene["py"], scene["is_gk"], scene["vmed"],
                                    tuple(scene["pitch"]))
    return scene


def external(data):
    px, py = data["px"].astype(np.float64), data["py"].astype(np.float64)
    bx, by = data["bx"].astype(np.float64), data["by"].astype(np.float64)
    recording = {"px": px, "py": py, "team": data["team"], "outfield": ~data["is_gk"],
                 "half": data["half"], "pitch": data["pitch"]}
    view = ball_view(bx, by)
    scene = {
        "half": data["half"], "pitch": data["pitch"],
        "team_logit": team_logit(data), "alive_logit": data["players_alive"].astype(np.float64),
        "ball": view, "nearest": nearest_player(px, py, bx, by),
        "keeper_views": [dict(view, gx=px[:, data["is_gk"]], gy=py[:, data["is_gk"]])],
        "px": px, "py": py, "is_gk": data["is_gk"], "vmed": adapt.median_speed(px, py),
        "left_team": left_team(recording),
    }
    truth = np.where(data["alive"], data["owner"], labelling.D).astype(np.int8)
    return precompute(scene), truth


def split_halves(codes, half):
    first, second = codes[half == 1], codes[half == 2]
    return np.concatenate([first, [labelling.X], second]).astype(np.int8)
