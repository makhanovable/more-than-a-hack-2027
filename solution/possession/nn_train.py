import numpy as np

from . import nn


def _scaler(paths):
    total = count = square = None
    for path in paths:
        with np.load(path) as stored:
            x = stored["x"].astype(np.float64)
        finite = np.isfinite(x)
        x = np.where(finite, x, 0.0)
        if total is None:
            total, square, count = x.sum(axis=0), (x * x).sum(axis=0), finite.sum(axis=0)
        else:
            total, square, count = total + x.sum(axis=0), square + (x * x).sum(axis=0), \
                count + finite.sum(axis=0)
    count = np.maximum(count, 1)
    mean = total / count
    std = np.sqrt(np.maximum(square / count - mean * mean, 0.0))
    return mean, np.where(std > 1e-6, std, 1.0)


def train(job):
    key, number, match_ids, config = job
    import torch
    seed = config["seed"] + number
    torch.set_num_threads(config["threads"])
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    rng = np.random.RandomState(seed)

    sequences = []
    for match_id in match_ids:
        for copy in range(config["copies"]):
            with np.load(nn.sequence_path(match_id, copy)) as stored:
                sequences.append({k: stored[k] for k in ("x", "home", "weight")})
                swap, n_players = stored["swap"], int(stored["n_players"])
    mean, std = _scaler([nn.sequence_path(match_id, 0) for match_id in match_ids])
    net = {"mean": mean.astype(np.float32), "std": std.astype(np.float32), "clip": config["clip"],
           "n_players": n_players, "dilations": list(config["dilations"])}

    channels, n_in = config["channels"], len(mean) + 2

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inp = torch.nn.Conv1d(n_in, channels, 1)
            self.blocks = torch.nn.ModuleList(
                [torch.nn.Conv1d(channels, channels, 3, dilation=d, padding=d)
                 for d in config["dilations"]])
            self.out = torch.nn.Conv1d(channels, 1, 1)
            self.drop = torch.nn.Dropout(config["dropout"])

        def forward(self, x):
            h = self.inp(x)
            for conv in self.blocks:
                h = h + self.drop(conv(torch.relu(h)))
            return self.out(torch.relu(h)).squeeze(1)

    model = Net()
    optimiser = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    lengths = np.array([len(s["x"]) for s in sequences], dtype=np.float64)
    chance = lengths / lengths.sum()
    window, edge, span = config["window"], config["edge"], nn.STEP * config["window"]
    steps = nn.STEP * np.arange(window)
    model.train()
    for iteration in range(config["iterations"]):
        rate = config["learning_rate"] * 0.5 * (1.0 + np.cos(np.pi * iteration / config["iterations"]))
        for group in optimiser.param_groups:
            group["lr"] = rate
        xs, ys, ws = [], [], []
        for _ in range(config["batch"]):
            sequence = sequences[rng.choice(len(sequences), p=chance)]
            index = rng.randint(0, len(sequence["x"]) - span) + steps
            part = sequence["x"][index]
            target = sequence["home"][index].astype(np.float32)
            if rng.random_sample() < 0.5:
                part, target = part[:, swap], 1.0 - target
            weight = sequence["weight"][index].copy()
            weight[:edge] = 0.0
            weight[-edge:] = 0.0
            xs.append(nn.standardise(part, net).T)
            ys.append(target)
            ws.append(weight)
        x = torch.from_numpy(np.stack(xs))
        y = torch.from_numpy(np.stack(ys))
        w = torch.from_numpy(np.stack(ws))
        loss = (w * torch.nn.functional.binary_cross_entropy_with_logits(
            model(x), y, reduction="none")).sum() / w.sum().clamp(min=1.0)
        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()

    state = {name: value.detach().numpy() for name, value in model.state_dict().items()}
    net.update(inp_w=state["inp.weight"][:, :, 0], inp_b=state["inp.bias"],
               out_w=state["out.weight"][0, :, 0], out_b=state["out.bias"][0],
               blk_w=[state["blocks.{}.weight".format(j)] for j in range(len(config["dilations"]))],
               blk_b=[state["blocks.{}.bias".format(j)] for j in range(len(config["dilations"]))])

    model.eval()
    probe = nn.standardise(sequences[0]["x"][:nn.STEP * 600:nn.STEP], net)
    with torch.no_grad():
        expected = model(torch.from_numpy(probe.T[None])).numpy()[0]
    assert float(np.abs(nn.forward(net, probe) - expected).max()) < 1e-3, "numpy forward differs"
    return key, number, net, swap, float(loss.detach())


def train_many(groups, config, jobs=1, verbose=True):
    work = [(key, number, list(match_ids), config)
            for key, match_ids in groups.items() for number in range(config["nets"])]
    if verbose:
        print("  training {} networks ({} at a time) ...".format(len(work), jobs), flush=True)
    if jobs > 1 and len(work) > 1:
        import multiprocessing
        with multiprocessing.get_context("spawn").Pool(min(jobs, len(work))) as pool:
            done = pool.map(train, work, chunksize=1)
    else:
        done = [train(job) for job in work]
    out = {key: ([None] * config["nets"], None) for key in groups}
    for key, number, net, swap, loss in done:
        out[key][0][number] = net
        out[key] = (out[key][0], swap)
        if verbose:
            print("    {} net {}: final loss {:.4f}".format(key, number, loss), flush=True)
    return out
