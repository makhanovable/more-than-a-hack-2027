import numpy as np

from . import decode, keeper, penalty

H, A, D, X = 0, 1, 2, 3
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16
ALL_RULES = ("ball", "keeper", "penalty")

RULES = {
    "certain": 5.0,
    "off_pitch": 1.0,
    "rest_speed": 0.5,
    "rest_frames": 50,
    "rest_alone": 1.5,
    "goal_area_margin": 1.5,
    "goal_kick_alone": 0.25,
}


def rest_runs(ball, rules=RULES):
    slow = np.isfinite(ball["bx"]) & (np.nan_to_num(ball["bspeed"], nan=np.inf) < rules["rest_speed"])
    return [(start, stop) for start, stop in decode.runs_of(slow)
            if stop - start >= rules["rest_frames"]]


def ball_proofs(scene, rules=RULES):
    length, width = scene["pitch"]
    x, y = scene["ball"]["bx"], scene["ball"]["by"]
    seen = np.isfinite(x)
    margin = rules["off_pitch"]
    with np.errstate(invalid="ignore"):
        dead = seen & ((x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin))
    goal_area = np.zeros(len(x), dtype=bool)
    reach = rules["goal_area_margin"]
    for start, stop in rest_runs(scene["ball"], rules):
        centre_x, centre_y = np.median(x[start:stop]), np.median(y[start:stop])
        if not (0.0 <= centre_x <= length and 0.0 <= centre_y <= width):
            continue
        nearest = scene["nearest"][start:stop]
        in_goal_area = min(centre_x, length - centre_x) <= GOAL_AREA_DEPTH + reach \
            and abs(centre_y - width / 2.0) <= GOAL_AREA_HALF_WIDTH + reach
        goal_area[start:stop] = in_goal_area
        if np.median(nearest) > rules["rest_alone"] or (
                in_goal_area and (nearest > rules["rest_alone"]).mean() >= rules["goal_kick_alone"]):
            dead[start:stop] = True
    return dead, goal_area


def dead_path(alive, half, cost):
    is_dead = np.ones(len(alive), dtype=bool)
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if len(rows):
            first, last = rows[0], rows[-1] + 1
            is_dead[first:last] = decode.viterbi2(alive[first:last], cost, cost)
    return is_dead


def lead_stoppages(is_dead, alive, half, lead, certain):
    if lead <= 0:
        return is_dead
    out = is_dead.copy()
    proven_dead = alive <= -certain
    proven_alive = alive >= certain
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        first, last = rows[0], rows[-1] + 1
        for start, stop in decode.runs_of(~is_dead[first:last]):
            start, stop = start + first, stop + first
            if stop >= last or proven_dead[stop:stop + 5].any():
                continue
            earlier = max(start + 1, stop - lead)
            held = np.flatnonzero(proven_alive[earlier:stop])
            if len(held):
                earlier += int(held[-1]) + 1
            out[earlier:stop] = True
    return out


def team_path(team, is_dead, half, params):
    codes = np.where(half > 0, D, X).astype(np.int8)
    cost = params["team_penalty"]
    for number in (1, 2):
        rows = np.flatnonzero(half == number)
        if not len(rows):
            continue
        first, last = rows[0], rows[-1] + 1
        for start, stop in decode.runs_of(~is_dead[first:last]):
            start, stop = start + first, stop + first
            path = decode.viterbi2(team[start:stop], cost, cost).astype(np.int8)
            spells = decode.segments_of(path)
            if len(spells) >= 2 and stop < last:
                last_start, last_stop, last_code = spells[-1]
                sign = 1.0 if last_code == H else -1.0
                net = sign * team[start + last_start:start + last_stop].sum()
                if net < params["end_factor"] * cost:
                    path[last_start:last_stop] = spells[-2][2]
            codes[start:stop] = path
    return codes


def alive_stage(scene, params, rules=RULES, use=ALL_RULES):
    half = scene["half"]
    inside = half > 0
    certain = rules["certain"]
    alive = np.clip(np.nan_to_num(scene["alive_logit"]), -params["alive_clip"], params["alive_clip"])
    proofs = scene.get("proofs") or ball_proofs(scene, rules)
    if "ball" in use:
        alive[proofs[0]] = -certain
    if scene.get("dead") is not None:
        alive[scene["dead"]] = -certain
    first_dead = dead_path(alive, half, params["alive_penalty"])
    out = {"first_dead": first_dead, "episodes": [], "penalties": [],
           "hold": np.zeros(len(half), dtype=bool)}

    team_known = np.full(len(half), np.nan)
    left = scene["left_team"]
    if "keeper" in use:
        hold, hold_end, episodes = keeper.detect(
            scene["keeper_views"], ~first_dead, tuple(scene["pitch"]), goal_kick=proofs[1])
        usable = hold & inside & (left >= 0) & (alive > -certain)
        alive[usable] = certain
        his = usable & (hold_end >= 0)
        keeper_team = np.where(hold_end == 0, left, 1 - left)
        team_known[his] = np.where(keeper_team[his] == H, certain, -certain)
        for episode in episodes:
            if episode["kept"]:
                frame = min(episode["first"], episode["stop"] - 1)
                episode["team"] = int(left[frame] if episode["end"] == 0 else 1 - left[frame])
        out["episodes"], out["hold"] = episodes, usable

    if "penalty" in use:
        kicks = scene.get("kicks")
        if kicks is None:
            kicks = penalty.detect(scene["px"], scene["py"], scene["is_gk"], scene["vmed"],
                                   tuple(scene["pitch"]))
        for kick in kicks:
            start, stop = kick["kick"], kick["stop"]
            if not (inside[start] and inside[stop - 1] and left[start] >= 0):
                continue
            kicking = 1 - left[start] if kick["end"] == 0 else left[start]
            alive[start:stop] = certain
            team_known[start:stop] = certain if kicking == H else -certain
            out["penalties"].append(dict(kick, team=int(kicking)))

    is_dead = dead_path(alive, half, params["alive_penalty"])
    out["is_dead"] = lead_stoppages(is_dead, alive, half, int(params.get("dead_lead", 0)), certain)
    out["team_known"], out["alive"] = team_known, alive
    return out


def team_stage(scene, state, params):
    team = np.clip(np.nan_to_num(scene["team_logit"]), -params["team_clip"], params["team_clip"])
    known = np.isfinite(state["team_known"])
    team[known] = state["team_known"][known]
    return team_path(team, state["is_dead"], scene["half"], params), team


def label(scene, params, rules=RULES, use=ALL_RULES):
    state = alive_stage(scene, params, rules, use)
    codes, team = team_stage(scene, state, params)
    return codes, dict(state, team=team)
