import numpy as np

from . import adapt, dataio, labelling, models, nn, scenes, tags, tune


def recording():
    tactical = dataio.load_atd("tactical")
    pitch = np.array([dataio.PITCH_LENGTH, dataio.PITCH_WIDTH])
    half = np.zeros(dataio.N_FRAMES, dtype=np.int8)
    for number, first, last in dataio.HALVES:
        half[first:last + 1] = number
    px = tactical["px"].astype(np.float64) * pitch[0]
    py = tactical["py"].astype(np.float64) * pitch[1]
    is_gk = tactical["slot_is_gk"]
    out = {
        "px": px, "py": py, "outfield": ~is_gk, "is_gk": is_gk, "half": half, "pitch": pitch,
        "team": tactical["slot_team"],
        "bx": tactical["bx"].astype(np.float64) * pitch[0],
        "by": tactical["by"].astype(np.float64) * pitch[1],
    }
    left = scenes.left_team(out)
    team = np.broadcast_to(tactical["slot_team"], px.shape).copy()
    for slot in np.flatnonzero(is_gk):
        by_goal = np.where(px[:, slot] < pitch[0] / 2.0, left, 1 - left)
        use = (left >= 0) & np.isfinite(px[:, slot])
        team[use, slot] = by_goal[use]
    out["team"], out["left_team"] = team, left
    return out


def live_estimate(members):
    return models.opinion(members, recording())


def before_kick_off(tagging):
    dead = np.zeros(dataio.N_FRAMES, dtype=bool)
    for event in tagging:
        if event["code"] == "set pieces" and "kick off (start)" in event["type"]:
            for _, first, last in dataio.HALVES:
                if first < event["start"] <= last:
                    dead[first:event["start"]] = True
    return dead


def scene(members, trained, seq_weight, tagging):
    rec = recording()
    graded = rec["half"] > 0
    players, ball, _, _ = models.tables(rec)
    seen = np.isfinite(rec["bx"])
    streams = models.streams(members, players, ball, seen, rows=graded)
    network = nn.predict(trained, players, ball, rec["half"])
    view = scenes.ball_view(rec["bx"], rec["by"])
    px, py = rec["px"], rec["py"]
    return {
        "half": rec["half"], "pitch": rec["pitch"],
        "team_logit": nn.blend(scenes.team_logit(streams), network, seq_weight),
        "alive_logit": streams["players_alive"],
        "ball": view, "nearest": scenes.nearest_player(px, py, rec["bx"], rec["by"]),
        "keeper_views": [dict(view, gx=px[:, rec["is_gk"]], gy=py[:, rec["is_gk"]])],
        "px": px, "py": py, "is_gk": rec["is_gk"], "vmed": adapt.median_speed(px, py),
        "left_team": rec["left_team"], "dead": before_kick_off(tagging),
    }


def run(verbose=True):
    if verbose:
        print("models, networks and decoder settings (trained and tuned on external matches) ...")
    members, _ = models.load()
    trained = nn.load()
    params = tune.load()
    if verbose:
        print("competition match: features, boosted models, networks ...")
    tagging = dataio.load_smart_tagging()
    match = scene(members, trained, params["seq_weight"], tagging)
    if verbose:
        print("labelling ...")
    codes, details = labelling.label(match, params)
    events = tags.set_pieces(tagging, match["half"] > 0)
    extra, repaired = tags.repair(codes, events, details["episodes"])
    if extra.any():
        match["dead"] = match["dead"] | extra
        codes, details = labelling.label(match, params)
    details.update(scene=match, params=params, set_pieces=events, repaired=repaired)
    return codes, details


def describe(codes):
    from . import decode
    graded = dataio.graded_mask()
    segments = [s for _, first, last in dataio.HALVES
                for s in decode.segments_of(codes[first:last + 1], first)]
    spells = [s for s in segments if s[2] in (labelling.H, labelling.A)]
    stoppages = [s for s in segments if s[2] == labelling.D]
    lengths = np.array([stop - start for start, stop, _ in spells], dtype=np.float64)
    n = float(graded.sum())
    home = (codes[graded] == labelling.H).sum()
    away = (codes[graded] == labelling.A).sum()
    dead = (codes[graded] == labelling.D).sum()
    return [
        "graded frames  H {:.1f}%  A {:.1f}%  D {:.1f}%".format(
            100 * home / n, 100 * away / n, 100 * dead / n),
        "home share of the ball in play  {:.1f}%".format(100.0 * home / max(1, home + away)),
        "possessions  {}  (mean {:.1f} s, median {:.1f} s)".format(
            len(spells), lengths.mean() / dataio.FPS if len(lengths) else 0.0,
            np.median(lengths) / dataio.FPS if len(lengths) else 0.0),
        "stoppages    {}".format(len(stoppages)),
    ]
