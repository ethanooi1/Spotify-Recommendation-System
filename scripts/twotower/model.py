# The two-tower model.
# One tower averages a playlist's context tracks into a single vector, the other tower encodes a single candidate track.
# Both towers read from the same 3 embedding tables (artist, album, track) so their vectors are directly comparable and live in the same N-dimensional space

# Import Libraries
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from twotower.vocab import Vocabulary

PAD_INDEX = Vocabulary.PAD_INDEX

# Hidden layers [dim, *hidden_dims, dim], *hidden_dims are ReLU activated, first and last dim are linear
def build_mlp(dim, hidden_dims):
    if not hidden_dims:
        return nn.Identity()

    layers = []
    prev = dim
    for hidden in hidden_dims:
        layers += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden

    layers.append(nn.Linear(prev, dim)) # back to dim, no activation

    # nn.Sequential on init takes a list of layers (nn.Linear, nn.ReLu, etc.).
    # On call it passes an input tensor (batch_size, dim) through each layer in order, returning the final output tensor (batch_size, dim)
    return nn.Sequential(*layers)

class TwoTowerModel(nn.Module):
    def __init__(self, track_vocab_size, artist_vocab_size, album_vocab_size, embedding_dim, temperature, hidden_dims):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature

        # The three embedding tables are summed to give one vector per token
        # padding_idx ensures the PAD_INDEX row never learns an embedding vector
        self.track_emb = nn.Embedding(track_vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.artist_emb = nn.Embedding(artist_vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.album_emb = nn.Embedding(album_vocab_size, embedding_dim, padding_idx=PAD_INDEX)

        # Initializes a tensor of size (track_vocab_size, ) with 0s for logQ correction
        self.register_buffer('item_log_q', torch.zeros(track_vocab_size), persistent=False)

        self.playlist_mlp = build_mlp(embedding_dim, hidden_dims)
        self.item_mlp = build_mlp(embedding_dim, hidden_dims)

        # Initialize the embedding weights with a smaller std than default (0, 1)
        for emb in [self.track_emb, self.artist_emb, self.album_emb]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)
            # Ensures PAD_INDEX's row is 0 again after initializing embeddings
            with torch.no_grad():
                emb.weight[PAD_INDEX].zero_()

    # Fills the logQ tensor with the log of frequency distribution of training tracks
    def set_item_log_q(self, item_counts):
        q = (item_counts / item_counts.sum()).clamp(min=1e-12) # if Q = 0, log(0) = -inf, .clamp() avoids that
        self.item_log_q.copy_(torch.log(q))

    # Summing the three embeddings gives one vector per token, still at dimension D
    def embed_tokens(self, track, artist, album):
        return self.track_emb(track) + self.artist_emb(artist) + self.album_emb(album)

    # Playlist tower. Averages the context tracks into one vector out.
    def encode_playlist(self, track, artist, album, mask):
        tokens = self.embed_tokens(track, artist, album) # [batch_size, context_length, dim]
        mask = mask.unsqueeze(-1).to(tokens.dtype)
        # Average across context_length, collapsing into [batch_size, dim]
        pooled = ((tokens * mask).sum(dim=1)) / (mask.sum(dim=1).clamp(min=1.0))

        return self.playlist_mlp(pooled) # [batch_size, dim]

    # Item tower. One "positive" track in, one vector out.
    def encode_item(self, track, artist, album):
        return self.item_mlp(self.embed_tokens(track, artist, album)) # [batch_size, dim]

    def forward(self, batch):
        playlist_vec = self.encode_playlist(batch['context_track'], batch['context_artist'], batch['context_album'], batch['context_mask'])
        item_vec = self.encode_item(batch['pos_track'], batch['pos_artist'], batch['pos_album'])

        return playlist_vec, item_vec

    # Scores batch_size (B) playlists against their positives, gives a [B, B] matrix where each (i, i) pair is the positive match
    # All other (i, j) pairs are in-batch negatives, B-1 negatives per playlist, 1 positive per playlist
    def in_batch_softmax_loss(self, playlist_vec, item_vec, item_indices):
        # Build unit vectors for playlist & item vector
        playlist_vec = F.normalize(playlist_vec, dim=-1)
        item_vec = F.normalize(item_vec, dim=-1)

        # Cosine similarity calculation, logits shape: (batch_size, batch_size)
        logits = playlist_vec @ item_vec.t() / self.temperature

        # logQ correction
        logits = logits - self.item_log_q[item_indices].unsqueeze(0)

        targets = torch.arange(logits.size(0), device=logits.device)

        # Runs softmax across each row of logits, computes cross entropy loss with the targets & returns the avg loss across the batch.
        return F.cross_entropy(logits, targets)

def build_model_from_vocab_sizes(vocab_sizes, embedding_dim, temperature, hidden_dims):
    return TwoTowerModel(vocab_sizes['track'], vocab_sizes['artist'], vocab_sizes['album'], embedding_dim, temperature, hidden_dims)
