import numpy as np

AREA_DEPTH = 16.5
AREA_HALF_WIDTH = 20.16
SPOT_DEPTH = 11.0

PARAMS = {
    "line_depth": 1.5,
    "line_half_width": 4.5,
    "inside_max": 1,
    "inside_margin": 1.0,
    "others_distance": 8.5,
    "waiting_depth": 35.0,
    "waiting_min": 10,
    "still_speed": 0.6,
    "line_up_frames": 50,
    "smooth": 12,
    "back_distance": 3.0,
    "kick_distance": 2.0,
    "kick_delay": 6,
    "kick_search": 200,
    "flight_frames": 14,
}


def runs_of(mask):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return list(zip(edges[::2], edges[1::2]))


def detect(px, py, is_gk, vmed, pitch, params=PARAMS):
    length, width = pitch
    n = len(px)
    seen = np.isfinite(px)
    lateral = np.abs(py - width / 2.0)
    slow = np.nan_to_num(vmed, nan=np.inf) < params["still_speed"]
    kernel = np.ones(2 * params["smooth"] + 1) / (2 * params["smooth"] + 1)
    found = []
    for end in (0, 1):
        depth = px if end == 0 else length - px
        with np.errstate(invalid="ignore"):
            inside = seen & (depth <= AREA_DEPTH) & (depth >= 0.0) & (lateral <= AREA_HALF_WIDTH)
            on_line = (inside & is_gk[None, :] & (depth <= params["line_depth"])
                       & (lateral <= params["line_half_width"])).any(axis=1)
            waiting = (seen & ~inside & (depth <= params["waiting_depth"])).sum(axis=1)
        margin = params["inside_margin"]
        with np.errstate(invalid="ignore"):
            well_inside = inside & (depth <= AREA_DEPTH - margin) \
                & (lateral <= AREA_HALF_WIDTH - margin)
        outfield_inside = (well_inside & ~is_gk[None, :]).sum(axis=1)
        with np.errstate(invalid="ignore"):
            to_spot = np.hypot(depth - SPOT_DEPTH, py - width / 2.0)
        ordered = np.sort(np.where(seen & ~is_gk[None, :], to_spot, np.inf), axis=1)
        spot_distance, others = ordered[:, 0], ordered[:, 1]
        picture = on_line & (outfield_inside <= params["inside_max"]) \
            & (waiting >= params["waiting_min"]) & slow & (others >= params["others_distance"])
        picture = np.convolve(picture.astype(np.float64), kernel, mode="same") > 0.5
        for start, stop in runs_of(picture):
            if stop - start < params["line_up_frames"]:
                continue
            if np.median(spot_distance[start:stop]) < params["back_distance"]:
                continue
            back = np.flatnonzero(spot_distance[start:stop] >= params["back_distance"])
            search_from = start + int(back[-1])
            search_to = min(n, stop + params["kick_search"])
            arrival = np.flatnonzero(spot_distance[search_from:search_to] <= params["kick_distance"])
            if not len(arrival):
                continue
            kick = search_from + int(arrival[0]) + params["kick_delay"]
            found.append({"end": end, "line_up": (int(start), int(stop)), "kick": int(kick),
                          "stop": int(kick + params["flight_frames"])})
    return sorted(found, key=lambda item: item["kick"])
