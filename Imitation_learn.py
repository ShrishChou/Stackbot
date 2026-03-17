# train_bc.py
import os
import glob
import math
import random
import argparse
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BCDataset(Dataset):
    """
    Loads all trajectory .npz files and flattens them into transition-level samples.
    Splitting is done outside this class using trajectory indices.
    """
    def __init__(self, files, normalize=True, stats=None, predict_delta=False):
        if len(files) == 0:
            raise ValueError("No dataset files provided.")

        self.files = files
        self.predict_delta = predict_delta

        obs_list = []
        act_list = []
        traj_ids = []

        for traj_idx, f in enumerate(files):
            data = np.load(f)

            obs = data["obs"].astype(np.float32)         # [T, obs_dim]
            actions = data["actions"].astype(np.float32) # [T, act_dim]

            if obs.ndim != 2 or actions.ndim != 2:
                raise ValueError(f"{f} has invalid shapes: obs {obs.shape}, actions {actions.shape}")

            if obs.shape[0] != actions.shape[0]:
                raise ValueError(f"{f} has mismatched lengths: obs {obs.shape[0]}, actions {actions.shape[0]}")

            if self.predict_delta:
                # action[:7] becomes q_des - current_q
                actions = actions.copy()
                actions[:, :7] = actions[:, :7] - obs[:, :7]

            obs_list.append(obs)
            act_list.append(actions)
            traj_ids.extend([traj_idx] * obs.shape[0])

        self.obs_raw = np.concatenate(obs_list, axis=0)
        self.act_raw = np.concatenate(act_list, axis=0)
        self.traj_ids = np.array(traj_ids, dtype=np.int32)

        self.obs_dim = self.obs_raw.shape[1]
        self.act_dim = self.act_raw.shape[1]

        if stats is None:
            self.obs_mean = self.obs_raw.mean(axis=0)
            self.obs_std = self.obs_raw.std(axis=0)
            self.obs_std[self.obs_std < 1e-6] = 1.0

            self.act_mean = self.act_raw.mean(axis=0)
            self.act_std = self.act_raw.std(axis=0)
            self.act_std[self.act_std < 1e-6] = 1.0
        else:
            self.obs_mean = stats["obs_mean"].astype(np.float32)
            self.obs_std = stats["obs_std"].astype(np.float32)
            self.act_mean = stats["act_mean"].astype(np.float32)
            self.act_std = stats["act_std"].astype(np.float32)

        self.normalize = normalize
        if normalize:
            self.obs = (self.obs_raw - self.obs_mean) / self.obs_std
            self.act = (self.act_raw - self.act_mean) / self.act_std
        else:
            self.obs = self.obs_raw.copy()
            self.act = self.act_raw.copy()

        self.obs = self.obs.astype(np.float32)
        self.act = self.act.astype(np.float32)

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(self.obs[idx]), torch.from_numpy(self.act[idx])

    def get_stats(self):
        return {
            "obs_mean": self.obs_mean.astype(np.float32),
            "obs_std": self.obs_std.astype(np.float32),
            "act_mean": self.act_mean.astype(np.float32),
            "act_std": self.act_std.astype(np.float32),
        }


class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, act_dim),
        )

    def forward(self, x):
        return self.net(x)


def split_trajectory_files(files, val_frac=0.1, seed=42):
    files = list(files)
    rng = np.random.default_rng(seed)
    rng.shuffle(files)

    n_val = max(1, int(len(files) * val_frac))
    val_files = files[:n_val]
    train_files = files[n_val:]

    if len(train_files) == 0:
        raise ValueError("Not enough trajectories for train/val split.")

    return train_files, val_files


def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_n = 0

    with torch.no_grad():
        for obs, act in loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = loss_fn(pred, act)

            bs = obs.shape[0]
            total_loss += loss.item() * bs
            total_n += bs

    return total_loss / max(total_n, 1)


@dataclass
class Config:
    data_dir: str = "datasets"
    out_dir: str = "bc_runs2"
    batch_size: int = 512
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-5
    hidden_dim: int = 256
    dropout: float = 0.0
    val_frac: float = 0.1
    seed: int = 42
    normalize: bool = True
    predict_delta: bool = False
    grad_clip: float = 1.0
    num_workers: int = 0


def save_training_artifacts(out_dir, model, stats, cfg, obs_dim, act_dim, best_val):
    os.makedirs(out_dir, exist_ok=True)

    ckpt_path = os.path.join(out_dir, "bc_policy_best.pt")
    stats_path = os.path.join(out_dir, "normalization_stats.npz")
    info_path = os.path.join(out_dir, "train_info.txt")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "hidden_dim": cfg.hidden_dim,
            "dropout": cfg.dropout,
            "normalize": cfg.normalize,
            "predict_delta": cfg.predict_delta,
        },
        ckpt_path,
    )

    np.savez(
        stats_path,
        obs_mean=stats["obs_mean"],
        obs_std=stats["obs_std"],
        act_mean=stats["act_mean"],
        act_std=stats["act_std"],
    )

    with open(info_path, "w") as f:
        f.write(f"best_val_loss: {best_val:.8f}\n")
        f.write(f"obs_dim: {obs_dim}\n")
        f.write(f"act_dim: {act_dim}\n")
        f.write(f"hidden_dim: {cfg.hidden_dim}\n")
        f.write(f"dropout: {cfg.dropout}\n")
        f.write(f"normalize: {cfg.normalize}\n")
        f.write(f"predict_delta: {cfg.predict_delta}\n")
        f.write(f"lr: {cfg.lr}\n")
        f.write(f"weight_decay: {cfg.weight_decay}\n")
        f.write(f"batch_size: {cfg.batch_size}\n")
        f.write(f"epochs: {cfg.epochs}\n")
        f.write(f"seed: {cfg.seed}\n")

    return ckpt_path, stats_path, info_path


def train(cfg: Config):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(cfg.data_dir, "traj_*.npz")))
    if len(files) == 0:
        raise FileNotFoundError(f"No traj_*.npz files found in {cfg.data_dir}")

    train_files, val_files = split_trajectory_files(files, val_frac=cfg.val_frac, seed=cfg.seed)

    print(f"Found {len(files)} trajectories total")
    print(f"Train trajectories: {len(train_files)}")
    print(f"Val trajectories:   {len(val_files)}")

    # Fit normalization only on training set
    train_dataset = BCDataset(
        train_files,
        normalize=cfg.normalize,
        stats=None,
        predict_delta=cfg.predict_delta,
    )
    stats = train_dataset.get_stats()

    val_dataset = BCDataset(
        val_files,
        normalize=cfg.normalize,
        stats=stats,
        predict_delta=cfg.predict_delta,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")
    print(f"obs_dim: {train_dataset.obs_dim}")
    print(f"act_dim: {train_dataset.act_dim}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        drop_last=False,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = MLPPolicy(
        obs_dim=train_dataset.obs_dim,
        act_dim=train_dataset.act_dim,
        hidden_dim=cfg.hidden_dim,
        dropout=cfg.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    loss_fn = nn.MSELoss()

    best_val = math.inf
    best_state = None

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_loss = 0.0
        n_seen = 0

        for obs, act in train_loader:
            obs = obs.to(device)
            act = act.to(device)

            pred = model(obs)
            loss = loss_fn(pred, act)

            optimizer.zero_grad()
            loss.backward()

            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()

            bs = obs.shape[0]
            running_loss += loss.item() * bs
            n_seen += bs

        train_loss = running_loss / max(n_seen, 1)
        val_loss = evaluate(model, val_loader, device, loss_fn)
        scheduler.step()

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={val_loss:.6f} | "
            f"lr={optimizer.param_groups[0]['lr']:.6e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("Training finished but no best model was recorded.")

    model.load_state_dict(best_state)

    ckpt_path, stats_path, info_path = save_training_artifacts(
        out_dir=cfg.out_dir,
        model=model,
        stats=stats,
        cfg=cfg,
        obs_dim=train_dataset.obs_dim,
        act_dim=train_dataset.act_dim,
        best_val=best_val,
    )

    print("\nTraining complete.")
    print(f"Best val loss: {best_val:.6f}")
    print(f"Saved model: {ckpt_path}")
    print(f"Saved stats: {stats_path}")
    print(f"Saved info:  {info_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="datasets")
    parser.add_argument("--out_dir", type=str, default="bc_runs")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_normalize", action="store_true")
    parser.add_argument("--predict_delta", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = Config(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        val_frac=args.val_frac,
        seed=args.seed,
        normalize=not args.no_normalize,
        predict_delta=True,
        grad_clip=args.grad_clip,
        num_workers=args.num_workers,
    )

    train(cfg)