# Spotify Recommendation System
#### By: Ethan Ooi

A two-tower retrieval model built on the [Spotify Million Playlist Dataset](https://engineering.atspotify.com/2018/5/introducing-the-million-playlist-dataset-and-recsys-challenge-2018) (MPD). It contains 1M playlists, 66M playlist-track interactions, and 33GB of raw JSON. I created 2 baselines using popularity and track co-occurrence, and compared the results against my trained two-tower model. The data pipeline is built with PySpark, the model is built with PyTorch, and the training was run on Kaggle using a T4 GPU.

## Results

The models (two-tower, co-occurrence baseline, and popularity baseline), were scored the same way on the same 50K validation & 50K test splits. I ranked every track in the 2.1M track catalog for each held-out playlist, then checked the hidden tracks against a top K. The two-tower and popularity baseline are built from all 900K train playlists. Co-occurrence baseline is built from a 50K sample of the 900K train playlists due to compute limits where the full split would generate 6.5 billion pairs to compute. 

Validation split (50,110 playlists):

| model | Recall@10 | Recall@100 | NDCG@100 |
|---|---|---|---|
| two-tower | 0.121 | 0.359 | 0.215 |
| co-occurrence baseline | 0.071 | 0.232 | 0.134 |
| popularity baseline | 0.008 | 0.045 | 0.020 |

Test split (49,747 playlists):

| model | Recall@10 | Recall@100 | NDCG@100 |
|---|---|---|---|
| two-tower | 0.120 | 0.360 | 0.215 |
| co-occurrence baseline | 0.071 | 0.233 | 0.135 |
| popularity baseline | 0.008 | 0.045 | 0.021 |

Test is within 0.001 of validation on every metric, so the tuning didn't overfit.

How it got there:

- plain in-batch softmax actually **lost** to co-occurrence (Recall@100 = 0.182)
- adding a logQ sampling-bias correction brought it to 0.328
- adding a small MLP head on each tower brought it to 0.359

I also tried a deeper [256, 256] head and a bigger 8192 batch. Both were measured against the final config and neither beat it, so neither is in it.

## How it works

- [`scripts/ingest_mpd.py`](scripts/ingest_mpd.py) - flattens the 33GB of nested MPD JSON into 3 Parquet tables (playlists, tracks, playlist_tracks) with PySpark for scalability
- [`scripts/build_splits.py`](scripts/build_splits.py) - deterministic 90/5/5 train/val/test split, for each val/test playlist it hides 20% of the tracks, capped at 10, as the positive targets to predict
- [`scripts/run_baselines.py`](scripts/run_baselines.py) - the popularity and co-occurrence baselines, plus the Recall/Precision/NDCG@K definitions everything else reuses
- [`scripts/build_vocab.py`](scripts/build_vocab.py) - maps 2.1M track, 283K artist, and 705K album string IDs to integer indices, with index 0 reserved for padding and 1 for unknown IDs
- [`scripts/twotower/vocab.py`](scripts/twotower/vocab.py) - defines the Vocabulary class used across the repo so all IDs encode the same way
- [`scripts/twotower/dataset.py`](scripts/twotower/dataset.py) - encodes the 60M row training table once into a flat `.npz` cache, each epoch samples one held-out positive per playlist and uses the rest as context
- [`scripts/twotower/model.py`](scripts/twotower/model.py) - Defines the two-tower model. Both towers share track/artist/album embedding tables, the playlist tower mean-pools its context, and the loss is in-batch softmax with the logQ correction
- [`scripts/twotower/train.py`](scripts/twotower/train.py) - training loop with hyperparams: dim, hidden_dims, temperature, epochs, batch_size, lr, logq
- [`scripts/twotower/evaluate.py`](scripts/twotower/evaluate.py) - full-catalog evaluation, scores each held-out playlist against all 2.1M tracks and prints metrics

## Design choices
- **In-batch negatives with a logQ correction.** Each batch contains 4096 playlists and each playlist contains a single random positive sample, while the other 4095 songs in the batch are a free negative in training. Using this system of randomly sampled positives, it disproportionately samples popular tracks, so the model over-penalizes popular tracks. Subtracting logQ, log(track frequency) from the logits, we undo the bias. This design choice improved Recall@100 by 80%.
- **Shared embedding tables.** A track has one vector whether it appears as playlist context or as the candidate being scored, so the two towers live in the same space.
- **PAD at zero, UNK at one.** Padding can never leak into the mean-pool, and tracks in val/test, but never seen in training still encode to a shared "unknown" vector instead of breaking retrieval.
- **A cache instead of Spark at train time.** Training reads one `.npz` of flat arrays with offsets, so epochs never touch Spark, massively reducing training time and memory.
- **One hidden layer, not two.** A single MLP head, [256] beat both the plain linear towers and [256, 256].

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Spark 4.0 needs Java 17 (`export JAVA_HOME=$(/usr/libexec/java_home -v 17)` on macOS). 

`data/bronze` expects the raw MPD JSON slices, Spotify removed the MPD dataset from public view, but I have it locally.

The Spark stages run locally:

```bash
python3 scripts/ingest_mpd.py
python3 scripts/build_splits.py
python3 scripts/run_baselines.py
python3 scripts/build_vocab.py
python3 scripts/twotower/dataset.py
```

Training and evaluation need an external GPU at this size. I ran both on Kaggle with a T4, in [this notebook](https://www.kaggle.com/code/ethanooi1/spotify-recommender-system-full-train):

```bash
python3 scripts/twotower/train.py --dim 128 --hidden-dims 256 --batch-size 4096 --epochs 30 --temperature 0.10 --logq
python3 scripts/twotower/evaluate.py --split validation
python3 scripts/twotower/evaluate.py --split test
```

## What I'd add next

- a FAISS/ANN index over the item vectors, so retrieval doesn't need a full 2.1M track matrix multiplication
- playlist titles as a model feature
- a second-stage ranker over the top few hundred retrieved candidates
