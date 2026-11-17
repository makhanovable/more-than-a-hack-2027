import numpy as np

AREA_DEPTH = 16.5
AREA_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16

PARAMS = {
    "catch_distance": 2.0,
    "hold_distance": 2.5,
    "bridge": 12,
    "set_down_distance": 6.0,
    "set_down_speed": 4.0,
    "leave_time": 0.6,
    "leave_speed": 4.0,
    "area_margin": 1.0,
    "keeper_max_depth": 30.0,
    "keeper_gap": 25,
    "before_frames": 25,
    "after_frames": 75,
    "play_share": 0.5,
    "min_frames": 25,
    "max_frames": 750,
    "flight_reach": 40.0,
    "rest_speed": 0.5,
    "rest_frames": 50,
    "rest_within": 25,
    "goal_area_margin": 1.5,
    "free_depth": 7.0,
    "long_hold": 150,
    "out_margin": 0.5,
    "out_lookback": 12,
}


def runs_of(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def keeper_at(gx, gy, end, length, max_depth):
    depth = gx if end == 0 else length - gx
    depth = np.where(np.isfinite(depth), depth, np.inf)
    column = depth.argmin(axis=1)
    rows = np.arange(len(gx))
    found = depth[rows, column] <= max_depth
    return (np.where(found, gx[rows, column], np.nan), np.where(found, gy[rows, column], np.nan),
            np.where(found, depth[rows, column], np.nan))


def one_ball(views):
    n = len(views[0]["bx"])
    out = {key: np.full(n, np.nan) for key in ("bx", "by", "bvx", "bvy", "bspeed")}
    for view in views:
        use = ~np.isfinite(out["bx"]) & np.isfinite(view["bx"])
        for key in out:
            out[key][use] = view[key][use]
    return out


def detect(views, alive, pitch, goal_kick=None, params=PARAMS):
    length, width = pitch
    n = len(alive)
    hold = np.zeros(n, dtype=bool)
    hold_end = np.full(n, -1, dtype=np.int8)
    episodes = []

    ball = one_ball(views)
    x, y, speed = ball["bx"], ball["by"], ball["bspeed"]
    seen = np.isfinite(x)
    margin = params["out_margin"]
    out = seen & ((x < -margin) | (x > length + margin) | (y < -margin) | (y > width + margin))

    rest_start = np.zeros(n, dtype=bool)
    rests = []
    in_goal_area = np.zeros(n, dtype=bool) if goal_kick is None else goal_kick.copy()
    reach = params["goal_area_margin"]
    for start, stop in runs_of(seen & (np.nan_to_num(speed, nan=np.inf) < params["rest_speed"])):
        if stop - start < params["rest_frames"]:
            continue
        rest_start[start] = True
        rests.append((start, stop))
        rest_x, rest_y = np.median(x[start:stop]), np.median(y[start:stop])
        depth = min(rest_x, length - rest_x)
        if 0.0 <= depth <= GOAL_AREA_DEPTH + reach \
                and abs(rest_y - width / 2.0) <= GOAL_AREA_HALF_WIDTH + reach:
            in_goal_area[start:stop] = True

    for end in (0, 1):
        distance = np.full(n, np.inf)
        leaving = np.zeros(n, dtype=bool)
        in_area = np.zeros(n, dtype=bool)
        tracked = np.zeros(n, dtype=bool)
        for view in views:
            kx, ky, depth = keeper_at(view["gx"], view["gy"], end, length,
                                      params["keeper_max_depth"])
            dx, dy = view["bx"] - kx, view["by"] - ky
            d = np.hypot(dx, dy)
            distance = np.where(np.isfinite(d), np.minimum(distance, d), distance)
            with np.errstate(invalid="ignore", divide="ignore"):
                radial = (dx * view["bvx"] + dy * view["bvy"]) / d
                recently = (d - params["catch_distance"]) / radial <= params["leave_time"]
                just_played = recently & (radial >= params["leave_speed"])
            leaving |= np.nan_to_num(just_played, nan=False).astype(bool)
            with np.errstate(invalid="ignore"):
                inside = (depth <= AREA_DEPTH + params["area_margin"]) \
                    & (np.abs(ky - width / 2.0) <= AREA_HALF_WIDTH + params["area_margin"])
            in_area |= inside
            tracked |= np.isfinite(kx)
        home = in_area.copy()
        for start, stop in runs_of(~tracked):
            if stop - start <= params["keeper_gap"] and start > 0 and in_area[start - 1]:
                home[start:stop] = True

        at_him = seen & (distance <= params["catch_distance"]) & home
        with_him = (~seen | (distance <= params["hold_distance"])) & home
        for start, stop in runs_of(~with_him & home & seen):
            if stop - start <= params["bridge"] and start > 0 and with_him[start - 1] \
                    and at_him[stop:stop + params["bridge"]].any():
                with_him[start:stop] = True
        set_down = seen & home & ~with_him & (distance <= params["set_down_distance"]) \
            & ~(np.nan_to_num(speed, nan=0.0) >= params["set_down_speed"])
        for start, stop in runs_of(set_down):
            if start > 0 and with_him[start - 1]:
                with_him[start:stop] = True
        anchor = at_him | (seen & leaving & home)

        for start, stop in runs_of(with_him):
            if not (at_him[start:stop].any() or (stop < n and anchor[stop])):
                continue
            blind = not at_him[start:min(stop, start + 3)].any()
            episode = {"start": int(start), "stop": int(stop), "end": end, "kept": False,
                       "blind": bool(blind)}
            episodes.append(episode)
            at = np.flatnonzero(at_him[start:stop])
            first = start + int(at[0]) if len(at) else stop
            episode["first"] = int(first)

            if stop - start < params["min_frames"]:
                episode["why"] = "short"
                continue
            if blind and (start == 0 or not seen[start - 1]):
                episode["why"] = "no ball before"
                continue
            if blind and not distance[start - 1] <= params["flight_reach"]:
                episode["why"] = "lost far from him"
                continue
            if stop - start > params["max_frames"] or stop >= n or not seen[stop]:
                episode["why"] = "no release"
                continue
            if out[max(0, start - params["out_lookback"]):stop + params["rest_within"]].any():
                episode["why"] = "ball off the pitch"
                continue
            if in_goal_area[start:stop + params["rest_within"]].any():
                episode["why"] = "goal kick"
                continue
            if rest_start[stop:stop + params["rest_within"]].any() or any(
                    first_rest < stop and last_rest > start and np.median(
                        distance[max(first_rest, start):min(last_rest, stop)])
                    > params["hold_distance"] for first_rest, last_rest in rests):
                episode["why"] = "placed ball"
                continue
            after = alive[stop:stop + params["after_frames"]]
            if not len(after) or after.mean() < params["play_share"]:
                episode["why"] = "no play afterwards"
                continue
            before = alive[max(0, start - params["before_frames"]):start]
            was_alive = len(before) > 0 and before.mean() >= params["play_share"]
            release_depth = x[stop] if end == 0 else length - x[stop]
            outside_goal_area = release_depth >= params["free_depth"]
            if not was_alive and (blind or not outside_goal_area):
                episode["why"] = "ball not in play before"
                continue
            if not outside_goal_area and (blind or stop - start >= params["long_hold"]):
                episode["why"] = "released from the goal area"
                continue
            episode["kept"] = True
            episode["why"] = "hold" if was_alive else "hold after a restart"
            hold[start:stop] = True
            hold_end[first:stop] = end
    return hold, hold_end, episodes
