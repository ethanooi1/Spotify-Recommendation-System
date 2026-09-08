# The two-tower retrieval model. One tower pools a playlist's context tracks into a vector,
# the other encodes a single candidate track, and both read the same three embedding tables
# so their vectors are directly comparable.

# Import Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_INDEX = 0 # index 0 is PAD in every vocab, see vocab.py


# The head on top of each tower. Empty hidden_dims gives a no-op, so the towers stay linear.
def build_mlp(dim, hidden_dims):
    if not hidden_dims:
        return nn.Identity()
    layers = []
    prev = dim
    for hidden in hidden_dims:
        layers += [nn.Linear(prev, hidden), nn.ReLU()]
        prev = hidden
    layers.append(nn.Linear(prev, dim)) # back to D, no activation, the output gets normalized
    return nn.Sequential(*layers)


class TwoTowerModel(nn.Module):
    def __init__(self, track_vocab_size, artist_vocab_size, album_vocab_size,
                 embedding_dim=128, temperature=0.05, hidden_dims=None):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.temperature = temperature
        self.hidden_dims = hidden_dims

        # padding_idx pins row 0 to zeros and keeps it from ever getting gradient.
        self.track_emb = nn.Embedding(track_vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.artist_emb = nn.Embedding(artist_vocab_size, embedding_dim, padding_idx=PAD_INDEX)
        self.album_emb = nn.Embedding(album_vocab_size, embedding_dim, padding_idx=PAD_INDEX)

        # Per-track log sampling probability for the logQ correction. Zeros mean no
        # correction, train.py fills it in when --logq is on.
        self.register_buffer("item_log_q", torch.zeros(track_vocab_size), persistent=False)

        self.playlist_mlp = build_mlp(embedding_dim, hidden_dims)
        self.item_mlp = build_mlp(embedding_dim, hidden_dims)

        for emb in (self.track_emb, self.artist_emb, self.album_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=0.05)
            with torch.no_grad():
                emb.weight[PAD_INDEX].zero_() # nn.init clobbered the PAD row, put it back

    # Fills the logQ table from how often each track appears in train.
    def set_item_log_q(self, item_counts):
        counts = item_counts.to(self.item_log_q.device, dtype=torch.float64)
        q = (counts / counts.sum()).clamp(min=1e-12)
        self.item_log_q.copy_(torch.log(q).to(self.item_log_q.dtype))

    # Summing the three embeddings gives one vector per token, still at dimension D.
    def embed_tokens(self, track, artist, album):
        return self.track_emb(track) + self.artist_emb(artist) + self.album_emb(album)

    # Playlist tower. Averages the context tokens, ignoring padded slots.
    def encode_playlist(self, track, artist, album, mask):
        tokens = self.embed_tokens(track, artist, album) # [B, L, D]
        mask = mask.unsqueeze(-1).to(tokens.dtype)
        pooled = (tokens * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return self.playlist_mlp(pooled) # [B, D]

    # Item tower. One candidate track in, one vector out.
    def encode_item(self, track, artist, album):
        return self.item_mlp(self.embed_tokens(track, artist, album)) # [B, D]

    def forward(self, batch):
        playlist_vec = self.encode_playlist(
            batch["context_track"], batch["context_artist"],
            batch["context_album"], batch["context_mask"],
        )
        item_vec = self.encode_item(batch["pos_track"], batch["pos_artist"], batch["pos_album"])
        return playlist_vec, item_vec

    # In-batch-negative softmax. Scoring B playlists against their B positives gives a [B, B]
    # matrix where row i should pick column i, and the other B-1 columns are free negatives.
    # Passing item_indices subtracts log(Q) per column, which undoes the popularity bias.
    def in_batch_softmax_loss(self, playlist_vec, item_vec, item_indices=None):
        playlist_vec = F.normalize(playlist_vec, dim=-1)
        item_vec = F.normalize(item_vec, dim=-1)
        logits = playlist_vec @ item_vec.t() / self.temperature

        if item_indices is not None:
            logits = logits - self.item_log_q[item_indices].unsqueeze(0)

        targets = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, targets)


def build_model_from_vocab_sizes(vocab_sizes, embedding_dim=128, temperature=0.05, hidden_dims=None):
    return TwoTowerModel(
        vocab_sizes["track"], vocab_sizes["artist"], vocab_sizes["album"],
        embedding_dim=embedding_dim, temperature=temperature, hidden_dims=hidden_dims,
    )
