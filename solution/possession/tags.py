import numpy as np

D = 2
SET_PIECES = ("throw-in", "corner kick", "goal kick", "free kick")

BEFORE = 125
AFTER = 200
DEAD = 87


def set_pieces(tagging, inside):
    out = []
    for event in tagging:
        if event["code"] != "set pieces":
            continue
        start = int(event["start"])
        kind = next((k for k in SET_PIECES if k in event["type"]), None)
        if kind is not None and 0 <= start < len(inside) and inside[start]:
            out.append((start, kind))
    return sorted(out)


def repair(codes, events, episodes=()):
    n = len(codes)
    extra = np.zeros(n, dtype=bool)
    repaired = []
    for start, kind in events:
        if (codes[max(0, start - BEFORE):start + AFTER] == D).any():
            continue
        extra[start:min(n, start + DEAD)] = True
        what = "{:.1f} s made dead".format(DEAD / 25.0)
        if kind == "goal kick":
            for episode in episodes:
                if episode.get("kept") and episode["start"] < start + AFTER \
                        and episode["stop"] > start - BEFORE:
                    extra[episode["start"]:episode["stop"]] = True
                    what += "; goalkeeper episode {}..{} is a goal kick, not a hold".format(
                        episode["start"], episode["stop"] - 1)
        repaired.append((start, kind, what))
    return extra, repaired
