import os

import numpy as np

from . import idsse, models

EVALUATION_SEED = 500000
KEEP = ("players_team", "players_alive", "ball_team")


def path_of(match_id):
    return os.path.join(idsse.CACHE_DIR, "held_out_{}.npz".format(match_id))


def evaluation_recording(match_id, profile, members):
    match, recording = models.degraded(match_id, profile,
                                       EVALUATION_SEED + models.number_of(match_id), members)
    players, ball, _, _ = models.tables(recording)
    return match, recording, players, ball


def _write(match_id, members, profile, n_members):
    match, recording, players, ball = evaluation_recording(match_id, profile, members)
    seen = np.isfinite(recording["bx"])
    out = models.streams(members, players, ball, seen)
    target = path_of(match_id)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    partial = target[:-4] + ".{}.part.npz".format(os.getpid())
    np.savez(partial, seen=seen, alive=match["alive"], owner=match["owner"],
             half=match["half"], pitch=match["pitch"], team=match["team"],
             is_gk=match["is_gk"], px=recording["px"].astype(np.float32),
             py=recording["py"].astype(np.float32), bx=recording["bx"].astype(np.float32),
             by=recording["by"].astype(np.float32), members=np.array(n_members),
             **{key: out[key].astype(np.float32) for key in KEEP})
    os.replace(partial, target)
    return target


def unseen(members, profile, match_ids=None, verbose=True):
    from . import metrica
    match_ids = metrica.available() if match_ids is None else list(match_ids)
    paths = []
    for match_id in match_ids:
        paths.append(_write(match_id, members, profile, len(members)))
        if verbose:
            print("  {}: read by the final models".format(match_id), flush=True)
    return paths


def load(match_id):
    with np.load(path_of(match_id)) as stored:
        return {key: stored[key] for key in stored.files}
