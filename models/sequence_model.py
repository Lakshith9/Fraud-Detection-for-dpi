"""LSTM over a payer's last SEQ_LEN-1 transactions + the current one; exported to ONNX for serving."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from pipeline.features import SEQ_DIM, SEQ_LEN


def build_model(hidden: int = 48):
    import torch
    from torch import nn

    class SeqNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(SEQ_DIM, hidden, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden + SEQ_DIM, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):  # x: [B, SEQ_LEN, SEQ_DIM] -> logit [B, 1]
            out, _ = self.lstm(x)
            return self.head(torch.cat([out[:, -1, :], x[:, -1, :]], dim=1))

    return SeqNet()


def train_sequence_model(
    X: np.ndarray,
    y: np.ndarray,
    Xv: np.ndarray,
    yv: np.ndarray,
    seed: int = 42,
    epochs: int = 6,
    neg_keep: float = 0.15,
    batch: int = 512,
    log=print,
):
    """Train with negative subsampling + positive class weighting; select the epoch by validation PR-AUC."""
    import torch
    from sklearn.metrics import average_precision_score
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    model = build_model()
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    pos = np.where(y == 1)[0]
    best_state, best_ap = None, -1.0
    Xv_t = torch.from_numpy(Xv.astype(np.float32))
    for ep in range(epochs):
        neg = np.where(y == 0)[0]
        neg = neg[rng.random(len(neg)) < neg_keep]
        idx = np.concatenate([pos, neg])
        rng.shuffle(idx)
        pw = torch.tensor([min(50.0, len(neg) / max(1, len(pos)))])
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        model.train()
        t = time.time()
        tot = 0.0
        for b in range(0, len(idx), batch):
            bi = np.sort(idx[b : b + batch])
            xb = torch.from_numpy(X[bi].astype(np.float32))
            yb = torch.from_numpy(y[bi].astype(np.float32)).unsqueeze(1)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * len(bi)
        model.eval()
        with torch.no_grad():
            pv = np.concatenate([model(Xv_t[i : i + 8192]).numpy().ravel() for i in range(0, len(Xv_t), 8192)])
        ap = average_precision_score(yv, pv)
        log(f"  seq epoch {ep + 1}/{epochs}  loss={tot / len(idx):.4f}  val PR-AUC={ap:.4f}  ({time.time() - t:.0f}s)")
        if ap > best_ap:
            best_ap = ap
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, best_ap


def predict_logits(model, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    import torch

    out = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out.append(model(torch.from_numpy(X[i : i + batch].astype(np.float32))).numpy().ravel())
    return np.concatenate(out) if out else np.zeros(0)


def export_onnx(model, path: Path) -> None:
    import torch

    dummy = torch.zeros(1, SEQ_LEN, SEQ_DIM, dtype=torch.float32)
    torch.onnx.export(
        model,
        dummy,
        str(path),
        input_names=["x"],
        output_names=["logit"],
        dynamic_axes={"x": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )


class OnnxSequenceScorer:
    """Batch-size-1 ONNX Runtime inference with warm-up (serving path)."""

    def __init__(self, path: Path, threads: int = 1):
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
        self.name = self.sess.get_inputs()[0].name
        for _ in range(20):
            self.logit(np.zeros((SEQ_LEN, SEQ_DIM), dtype=np.float32))

    def logit(self, seq: np.ndarray) -> float:
        x = seq.reshape(1, SEQ_LEN, SEQ_DIM).astype(np.float32, copy=False)
        return float(self.sess.run(None, {self.name: x})[0][0, 0])

    def logits(self, X: np.ndarray, batch: int = 4096) -> np.ndarray:
        out = []
        for i in range(0, len(X), batch):
            out.append(self.sess.run(None, {self.name: X[i : i + batch].astype(np.float32)})[0].ravel())
        return np.concatenate(out) if out else np.zeros(0)
