"""Inference with a frozen TimeRCD encoder and a four-head, 11-specialist router."""
from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler

from HP_list import Optimal_Uni_algo_HP_dict
import model_wrapper
from models.router import SoftLabelMLP
from models.time_rcd.ts_encoder_bi_bias import TimeSeriesEncoder
from utils.slidingWindows import find_length_rank


ROOT = Path(__file__).resolve().parent


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def zscore(data, zero_std=1.0):
    data = np.asarray(data)
    std = np.std(data, axis=0, keepdims=True)
    return (data - np.mean(data, axis=0, keepdims=True)) / np.where(std == 0, zero_std, std)


def minmax(scores):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("A specialist returned non-finite scores")
    low, high = np.min(scores), np.max(scores)
    return (scores - low) / (high - low) if high > low else np.zeros_like(scores)


def mean_fuse(score_columns):
    lengths = {len(column) for column in score_columns}
    if len(lengths) != 1:
        raise ValueError("Specialist score lengths differ: %s" % sorted(lengths))
    matrix = np.stack(score_columns, axis=1)
    if not np.isfinite(matrix).all():
        raise ValueError("A specialist returned non-finite scores")
    return minmax(StandardScaler().fit_transform(matrix).mean(axis=1))


def run_specialist(name, train, test):
    """Preserve the source implementation's detector-specific fitting protocol."""
    hp = dict(Optimal_Uni_algo_HP_dict.get(name, {}))
    fn = getattr(model_wrapper, "run_" + name)
    if name in {"PCA", "OCSVM"}:
        period = hp.pop("periodicity", 1)
        score = fn(train, test, slidingWindow=find_length_rank(test, rank=period), **hp)
    elif name in {"Sub_PCA", "Sub_OCSVM"}:
        score = fn(train, test, **hp)
    else:
        score = fn(test, **hp)
    score = np.asarray(score, dtype=float).ravel()
    if len(score) != len(test) or not np.isfinite(score).all():
        raise ValueError("Invalid scores from %s: length=%d expected=%d" % (name, len(score), len(test)))
    return score


class TSRouter:
    def __init__(self, checkpoint_dir=ROOT / "checkpoints", device="cpu"):
        self.device = torch.device(device)
        directory = Path(checkpoint_dir)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        for item in self.manifest["files"]:
            if sha256(directory / item["name"]) != item["sha256"]:
                raise ValueError("Checkpoint checksum mismatch: " + item["name"])
        cfg = self.manifest["encoder"]["config"]
        self.encoder = TimeSeriesEncoder(**cfg)
        state = {}
        for name in self.manifest["encoder"]["shards"]:
            shard = torch.load(directory / name, map_location="cpu", weights_only=True)
            if state.keys() & shard.keys():
                raise ValueError("Duplicate tensors in encoder shards")
            state.update(shard)
        self.encoder.load_state_dict(state, strict=True)
        self.encoder.to(self.device).eval().requires_grad_(False)
        router_cfg = self.manifest["router"]
        self.experts = router_cfg["expert_algorithms"]
        self.heads = router_cfg["target_metrics"]
        self.router = SoftLabelMLP(router_cfg["input_dim"], tuple(router_cfg["hidden_layers"]),
                                   len(self.experts), len(self.heads), dropout=0.0)
        self.router.load_state_dict(torch.load(directory / router_cfg["file"],
                                              map_location="cpu", weights_only=True), strict=True)
        self.router.to(self.device).eval().requires_grad_(False)

    @torch.inference_mode()
    def embed(self, series):
        series = np.asarray(series, dtype=np.float32).reshape(-1, 1)
        normalized = zscore(series, zero_std=1e-8)
        tensor = torch.from_numpy(normalized).unsqueeze(0).to(self.device)
        mask = torch.ones((1, len(series)), dtype=torch.bool, device=self.device)
        local = self.encoder(tensor, mask)
        feature = local.mean(dim=2).mean(dim=1)
        return torch.nan_to_num(feature, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.inference_mode()
    def predict(self, data, train_index, top_k=3, head="VUS-PR", route_span="original", seed=42):
        """Labels are not accepted by the scoring API. All outputs cover the test suffix."""
        data = np.asarray(data, dtype=float)
        if data.ndim == 1:
            data = data[:, None]
        if data.ndim != 2 or not np.isfinite(data).all():
            raise ValueError("Expected finite data with shape [time, variables]")
        if not 0 < train_index < len(data):
            raise ValueError("train_index must leave a nonempty prefix and suffix")
        if not 1 <= top_k <= len(self.experts):
            raise ValueError("top_k must be between 1 and %d" % len(self.experts))
        heads = self.heads if head == "all" else [head]
        if any(h not in self.heads for h in heads):
            raise ValueError("Unknown router head: " + str(head))
        if route_span not in {"original", "full", "test"}:
            raise ValueError("Unknown route_span")
        multi = data.shape[1] > 1
        # Match the source entries: full univariate observations, first 10k test points for multivariate routing.
        route = data
        if route_span == "test" or (route_span == "original" and multi):
            route = data[train_index:]
            if route_span == "original":
                route = route[:10000]
        np.random.seed(seed)
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        columns = {h: [] for h in heads}
        selections = {h: [] for h in heads}
        probabilities = []
        for dim in range(data.shape[1]):
            feature = self.embed(route[:, dim])
            probs = self.router(feature).softmax(dim=2)[0].cpu().numpy()
            probabilities.append(probs.tolist())
            train = data[:train_index, dim:dim+1][:10000].copy()
            test = data[train_index:, dim:dim+1].copy()
            if multi:
                for arr in (train, test):
                    if np.all(np.std(arr, axis=0, keepdims=True) <= 1e-12):
                        arr += rng.normal(0.0, 1e-6, size=arr.shape)
            train, test = zscore(train), zscore(test)
            for h in heads:
                indices = np.argsort(-probs[self.heads.index(h)])[:top_k]
                names = [self.experts[i] for i in indices]
                scores = [run_specialist(name, train, test) for name in names]
                columns[h].append(mean_fuse(scores) if top_k > 1 else minmax(scores[0]))
                selections[h].append(names)
        return {
            "scores": {h: mean_fuse(columns[h]) if multi else columns[h][0] for h in heads},
            "selected_experts": selections,
            "probabilities": probabilities,
            "expert_order": self.experts,
            "head_order": self.heads,
            "train_index": int(train_index),
            "route_span": route_span,
            "route_points": len(route),
        }
