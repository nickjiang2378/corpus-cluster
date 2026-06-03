"""Train a TopK SAE (sae.py) on extracted DINOv2 activations (CLS or patch).

Usage:
  python train_sae.py --which cls   --d-sae 12288 --k 64 --steps 20000
  python train_sae.py --which patch --d-sae 12288 --k 32 --steps 30000

Reads activations as memmaps from --data-dir (or a single .npy via --acts-path, e.g. the
precomputed 13M CLS embeddings). CLS loads fully into RAM; patch block-shuffles. Checkpoints
frequently (resume-safe / preemption-proof). Logs FVU / L0 / dead-fraction on a held-out split.
"""
import os
import sys
import time
import json
import signal
import argparse
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
from sae import TopKSAE  # noqa: E402

DIM = 768


def load_acts(data_dir, which, acts_path=None):
    """Return a 2D memmap view [N_tokens, 768] and N_tokens.
    If acts_path is given, load that .npy directly (e.g. precomputed 13M CLS embeddings)."""
    if acts_path:
        arr = np.load(acts_path, mmap_mode="r")
        if arr.ndim == 3:
            arr = arr.reshape(-1, DIM)
        return arr, arr.shape[0]
    if which == "cls":
        arr = np.load(os.path.join(data_dir, "cls_acts.fp16.npy"), mmap_mode="r")
    else:
        arr = np.load(os.path.join(data_dir, "patch_acts.fp16.npy"), mmap_mode="r")
        arr = arr.reshape(-1, DIM)  # view, contiguous
    return arr, arr.shape[0]


class BlockShuffleStream:
    """Yields shuffled minibatches. CLS: full in-RAM shuffle. Patch: block-shuffle."""

    def __init__(self, arr, n_train, batch_size, block_size, device, seed=0,
                 full_in_ram=False):
        self.arr = arr
        self.n_train = n_train
        self.bs = batch_size
        self.block = block_size
        self.device = device
        self.rng = np.random.default_rng(seed)
        self.full_in_ram = full_in_ram
        self.ram = None
        if full_in_ram:
            self.ram = torch.from_numpy(np.ascontiguousarray(arr[:n_train])).to(device)

    def __iter__(self):
        while True:
            if self.full_in_ram:
                perm = torch.randperm(self.n_train, device=self.device)
                for i in range(0, self.n_train - self.bs, self.bs):
                    yield self.ram[perm[i:i + self.bs]].float()
            else:
                start = int(self.rng.integers(0, max(1, self.n_train - self.block)))
                blk = np.ascontiguousarray(self.arr[start:start + self.block])
                blk = torch.from_numpy(blk).to(self.device)
                perm = torch.randperm(blk.shape[0], device=self.device)
                for i in range(0, blk.shape[0] - self.bs, self.bs):
                    yield blk[perm[i:i + self.bs]].float()


@torch.no_grad()
def evaluate(model, arr, n_train, n_total, device, scale, n_eval=200_000, bs=8192):
    """Global FVU / L0 / dead-fraction on held-out tokens."""
    idx = np.arange(n_train, n_total)
    if len(idx) > n_eval:
        idx = idx[:: len(idx) // n_eval][:n_eval]
    # global mean over eval set
    sse = 0.0
    sst = 0.0
    l0s = []
    fired = torch.zeros(model.d_sae, dtype=torch.bool, device=device)
    # first pass mean
    msum = torch.zeros(DIM, device=device)
    cnt = 0
    for i in range(0, len(idx), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[idx[i:i + bs]])).to(device).float() * scale
        msum += x.sum(0); cnt += x.shape[0]
    gmean = msum / cnt
    for i in range(0, len(idx), bs):
        x = torch.from_numpy(np.ascontiguousarray(arr[idx[i:i + bs]])).to(device).float() * scale
        z = model.encode(x)
        recon = model.decode(z)
        sse += (recon - x).pow(2).sum().item()
        sst += (x - gmean).pow(2).sum().item()
        l0s.append((z > 0).float().sum(-1).mean().item())
        fired |= (z > 0).any(0)
    return {
        "val_fvu": sse / (sst + 1e-8),
        "val_l0": float(np.mean(l0s)),
        "val_dead_frac": 1.0 - float(fired.float().mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["cls", "patch"], required=True)
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--acts-path", default=None, help="explicit .npy of activations (e.g. precomputed 13M CLS)")
    ap.add_argument("--block-shuffle", action="store_true", help="force block-shuffle (don't load all to GPU)")
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--tag", default=None, help="run tag for ckpt/out naming")
    ap.add_argument("--d-sae", type=int, default=12288)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--k-aux", type=int, default=512)
    ap.add_argument("--aux-coef", type=float, default=1.0 / 32.0)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.02)
    ap.add_argument("--block-size", type=int, default=4_000_000)
    ap.add_argument("--dead-steps-threshold", type=int, default=200)
    ap.add_argument("--ckpt-every", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=1000)
    args = ap.parse_args()

    tag = args.tag or f"{args.which}_d{args.d_sae}_k{args.k}"
    ckpt_dir = args.ckpt_dir or os.path.join(args.data_dir, "ckpt", tag)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")
    out_path = os.path.join(args.data_dir, f"sae_{tag}.pt")
    log_path = os.path.join(ROOT, "results", f"train_{tag}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    arr, n_total = load_acts(args.data_dir, args.which, args.acts_path)
    n_train = int(n_total * (1 - args.val_frac))
    print(f"[{tag}] n_total={n_total} n_train={n_train} dim={DIM} device={device}", flush=True)

    model = TopKSAE(DIM, args.d_sae, args.k, k_aux=args.k_aux, aux_coef=args.aux_coef,
                    dead_steps_threshold=args.dead_steps_threshold).to(device)

    # normalization on a sample of train
    samp = np.ascontiguousarray(arr[np.linspace(0, n_train - 1, 50000).astype(int)])
    samp_t = torch.from_numpy(samp).to(device).float()
    scale = float(np.sqrt(DIM) / samp_t.norm(dim=-1).mean().item())
    model.set_norm(scale)
    model.set_bias_to_mean((samp_t * scale).mean(0))
    print(f"[{tag}] in_scale={scale:.4f}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    start_step = 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        start_step = ck["step"]
        print(f"[{tag}] resumed from step {start_step}", flush=True)

    def save_ckpt(step):
        tmp = ckpt_path + ".tmp"
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                    "config": vars(args), "scale": scale}, tmp)
        os.replace(tmp, ckpt_path)

    # save on preemption
    def handler(signum, frame):
        print(f"[{tag}] caught signal {signum}, checkpointing...", flush=True)
        save_ckpt(cur_step)
        sys.exit(0)
    signal.signal(signal.SIGTERM, handler)

    full_in_ram = (args.which == "cls") and not args.block_shuffle and args.acts_path is None
    stream = BlockShuffleStream(
        arr, n_train, args.batch_size, args.block_size, device,
        seed=0, full_in_ram=full_in_ram)
    it = iter(stream)

    cur_step = start_step
    t0 = time.time()
    logf = open(log_path, "a")
    for step in range(start_step, args.steps):
        cur_step = step
        lr = args.lr * min(1.0, (step + 1) / args.warmup)
        for g in opt.param_groups:
            g["lr"] = lr
        x = next(it) * scale
        out = model(x)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
        model.normalize_decoder()

        if step % 200 == 0:
            rate = (step - start_step + 1) * args.batch_size / max(1e-6, time.time() - t0)
            print(f"[{tag}] step {step} fvu={out['fvu']:.4f} l0={out['l0']:.1f} "
                  f"dead={model.dead_fraction():.3f} aux={out['aux_loss']:.4f} "
                  f"{rate:.0f} tok/s", flush=True)
        if step > 0 and step % args.eval_every == 0:
            ev = evaluate(model, arr, n_train, n_total, device, scale)
            rec = {"step": step, "train_fvu": out["fvu"], **ev}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            print(f"[{tag}] EVAL step {step} {ev}", flush=True)
        if step > 0 and step % args.ckpt_every == 0:
            save_ckpt(step)

    save_ckpt(args.steps)
    ev = evaluate(model, arr, n_train, n_total, device, scale)
    logf.write(json.dumps({"step": args.steps, "final": True, **ev}) + "\n"); logf.flush()
    logf.close()
    # final standalone artifact
    torch.save({"model": model.state_dict(), "config": vars(args), "scale": scale,
                "tag": tag, "final_eval": ev}, out_path)
    print(f"[{tag}] FINAL {ev} -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
