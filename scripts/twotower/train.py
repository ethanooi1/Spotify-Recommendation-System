# Trains the two-tower model on the cached playlists from dataset.py.
# Auto-detects the device, so the same command runs on CPU locally and on a GPU on Kaggle.
#
#     python3 scripts/twotower/train.py --dim 128 --hidden-dims 256 --batch-size 4096 --epochs 30 --logq

# Import Libraries
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1])) # so "twotower" resolves when run as a script
from twotower.dataset import PlaylistDataset, make_dataloader  # noqa: E402
from twotower.model import build_model_from_vocab_sizes  # noqa: E402


def select_device(requested):
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# "256" gives one hidden layer, "256,256" gives two, empty leaves the towers linear.
def parse_hidden_dims(spec):
    if not spec or spec.strip().lower() in {"none", "0"}:
        return None
    return [int(part) for part in spec.split(",") if part.strip()]


def load_vocab_sizes(vocab_dir):
    meta = json.loads((Path(vocab_dir) / "vocab_metadata.json").read_text())
    return {e: meta["entities"][e]["vocab_size"] for e in ("track", "artist", "album")}


# How often each track appears in train, which is what the logQ correction divides out.
def compute_item_counts(cache_path, num_tracks):
    counts = np.bincount(np.load(cache_path)["track_idx"], minlength=num_tracks)
    return torch.from_numpy(counts)


# Fraction of playlists whose own positive wins its row. A quick "is it learning" readout,
# the real metrics come from evaluate.py.
def in_batch_accuracy(playlist_vec, item_vec):
    logits = F.normalize(playlist_vec, dim=-1) @ F.normalize(item_vec, dim=-1).t()
    targets = torch.arange(logits.size(0), device=logits.device)
    return (logits.argmax(dim=1) == targets).float().mean().item()


# One pass over the loader. Returns (avg_loss, avg_acc, updated_global_step).
def train_one_epoch(model, loader, optimizer, device, epoch, log_every=50, max_steps=0, global_step=0):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for step, batch in enumerate(loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        playlist_vec, item_vec = model(batch)
        loss = model.in_batch_softmax_loss(playlist_vec, item_vec, item_indices=batch["pos_track"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            acc = in_batch_accuracy(playlist_vec, item_vec)
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1
        global_step += 1

        if step % log_every == 0:
            print(f"  epoch {epoch}  step {step:>6d}  loss {loss.item():.4f}  in-batch acc {acc:.3f}")
        if max_steps and global_step >= max_steps:
            break

    return total_loss / max(n_batches, 1), total_acc / max(n_batches, 1), global_step


# Model weights only, no optimizer state, since I never resume a run.
def save_checkpoint(path, model, epoch, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "config": config}, path)


def parse_args():
    parser = argparse.ArgumentParser(description="Train the two-tower retrieval model.")
    parser.add_argument("--cache", default="artifacts/twotower/train_playlists.npz")
    parser.add_argument("--vocab-dir", default="artifacts/vocab")
    parser.add_argument("--out", default="artifacts/twotower/checkpoints")
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--hidden-dims", default="") # "256" for one MLP layer, empty for linear towers
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=1024) # also the in-batch negative count
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max-context-len", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--logq", action="store_true") # debias negatives by track popularity
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = select_device(args.device)
    vocab_sizes = load_vocab_sizes(args.vocab_dir)
    hidden_dims = parse_hidden_dims(args.hidden_dims)

    model = build_model_from_vocab_sizes(
        vocab_sizes, embedding_dim=args.dim, temperature=args.temperature, hidden_dims=hidden_dims,
    ).to(device)

    if args.logq:
        model.set_item_log_q(compute_item_counts(args.cache, vocab_sizes["track"]).to(device))

    dataset = PlaylistDataset(args.cache, max_context_len=args.max_context_len)
    loader = make_dataloader(dataset, args.batch_size, num_workers=args.num_workers, seed=args.seed)
    print(f"{device}, dim {args.dim}, head {hidden_dims or 'linear'}, logQ {bool(args.logq)}, "
          f"{len(dataset):,} playlists")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # evaluate.py rebuilds the model from this, so it rides along in the checkpoint.
    config = {
        "vocab_sizes": vocab_sizes,
        "embedding_dim": args.dim,
        "temperature": args.temperature,
        "hidden_dims": hidden_dims,
        "logq": bool(args.logq),
    }

    best_acc, global_step = -1.0, 0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss, avg_acc, global_step = train_one_epoch(
            model, loader, optimizer, device, epoch, max_steps=args.max_steps, global_step=global_step,
        )
        print(f"epoch {epoch} done  avg loss {avg_loss:.4f}  avg in-batch acc {avg_acc:.3f}  ({time.time() - start:.1f}s)")

        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint(Path(args.out) / "best.pt", model, epoch, config)

        if args.max_steps and global_step >= args.max_steps:
            break

    print(f"done, best in-batch acc {best_acc:.4f}, saved {Path(args.out) / 'best.pt'}")


if __name__ == "__main__":
    main()
