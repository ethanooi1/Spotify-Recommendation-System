# Trains the two-tower model on the cached playlists from dataset.py.
# Full train done on Kaggle with T4 GPU

# Import Libraries
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twotower.dataset import PlaylistDataset, make_dataloader
from twotower.model import build_model_from_vocab_sizes

CACHE_PATH = 'artifacts/twotower/train_playlists.npz'
VOCAB_DIR = 'artifacts/vocab'
CHECKPOINT_DIR = 'artifacts/twotower/checkpoints'

SEED = 42
MAX_CONTEXT_LEN = 100
NUM_WORKERS = 2
LOG_EVERY = 50

def select_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')

def load_vocab_sizes():
    return json.loads((Path(VOCAB_DIR) / 'vocab_metadata.json').read_text())['vocab_sizes']

# How often each track appears in train, which is what the logQ correction divides out
def compute_item_counts(num_tracks):
    counts = np.bincount(np.load(CACHE_PATH)['track_idx'], minlength=num_tracks)

    return torch.from_numpy(counts)

# Proportion of playlists in-batch whose "positive" track was identified. In-training metric to follow progress, not a final eval metric
def in_batch_accuracy(playlist_vec, item_vec):
    logits = F.normalize(playlist_vec, dim=-1) @ F.normalize(item_vec, dim=-1).t()
    targets = torch.arange(logits.size(0), device=logits.device)

    return (logits.argmax(dim=1) == targets).float().mean().item()

def train_one_epoch(model, loader, optimizer, device, epoch):
    model.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    # Loop through dataloader, batches of B (4096) playlists
    for step, batch in enumerate(loader):
        for k in batch:
            batch[k] = batch[k].to(device)
            
        # calls model.forward(), shapes: playlist_vec [B, dim], item_vec [B, dim]
        playlist_vec, item_vec = model(batch)
        loss = model.in_batch_softmax_loss(playlist_vec, item_vec, item_indices=batch['pos_track'])

        optimizer.zero_grad() # clears previous gradients
        loss.backward() # calculates gradients for params in-batch
        optimizer.step() # updates params with gradients

        with torch.no_grad():
            acc = in_batch_accuracy(playlist_vec, item_vec)
        total_loss += loss.item()
        total_acc += acc
        n_batches += 1

        if step % LOG_EVERY == 0:
            print(f'  epoch: {epoch}  step: {step:>6d}  loss: {loss.item():.4f}  in-batch acc: {acc:.3f}')

    avg_total_loss = total_loss / n_batches
    avg_total_acc = total_acc / n_batches
    
    return avg_total_loss, avg_total_acc

# Saves model weights, epoch, and config to a .pt file
def save_checkpoint(path, model, epoch, config):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'config': config}, path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--hidden-dims', type=int, nargs='*', default=[]) 
    parser.add_argument('--temperature', type=float, default=0.05)
    parser.add_argument('--batch-size', type=int, default=1024) 
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--logq', action='store_true') # debias negatives by track popularity

    return parser.parse_args()

def main():
    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    device = select_device()
    vocab_sizes = load_vocab_sizes()

    model = build_model_from_vocab_sizes(vocab_sizes, embedding_dim=args.dim, temperature=args.temperature, hidden_dims=args.hidden_dims).to(device)

    if args.logq:
        model.set_item_log_q(compute_item_counts(vocab_sizes['track']).to(device))

    # Build dataset & dataloader 
    dataset = PlaylistDataset(CACHE_PATH, max_context_len=MAX_CONTEXT_LEN)
    loader = make_dataloader(dataset, args.batch_size, NUM_WORKERS, SEED)
    print(f'device: {device}, dim: {args.dim}, hidden: {args.hidden_dims or "linear"}, logQ: {bool(args.logq)}, {len(dataset):,} playlists')

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # evaluate.py uses same hyperparams
    config = {
        'vocab_sizes': vocab_sizes,
        'embedding_dim': args.dim,
        'temperature': args.temperature,
        'hidden_dims': args.hidden_dims,
        'logq': bool(args.logq)
    }

    checkpoint_path = Path(CHECKPOINT_DIR) / 'best.pt'
    best_acc = -1.0
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        avg_loss, avg_acc = train_one_epoch(model, loader, optimizer, device, epoch)
        print(f'epoch {epoch} done, avg loss {avg_loss:.4f},  avg in-batch acc {avg_acc:.3f}  ({time.time() - start:.1f}s)')

        # If the in-batch accuracy is better than the best so far, save the model weights to best.pt
        if avg_acc > best_acc:
            best_acc = avg_acc
            save_checkpoint(checkpoint_path, model, epoch, config)

    print(f'done, best in-batch acc {best_acc:.4f}, saved to {checkpoint_path}')

if __name__ == '__main__':
    main()
