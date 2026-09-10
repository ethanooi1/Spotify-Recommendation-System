# Import Libraries
import json
from pathlib import Path

# Maps track/artist/album string IDs to contiguous integers which become embedding table rows.
# Index 0 is PAD, an empty row for each entity's embedding table to pad short playlists to the same length as the longest playlist in the batch
# Index 1 is UNK, a row for each entity's embedding table to hold any ID we didn't see in training. UNK will learn its own embedding vector, but it will be shared by all unseen IDs. Real IDs start at 2.
class Vocabulary:
    PAD_INDEX = 0
    UNK_INDEX = 1
    NUM_RESERVED = 2

    def __init__(self, id_to_index):
        self.id_to_index = id_to_index

    # sorted so the same IDs always get the same indices
    @classmethod
    def build(cls, ids):
        id_to_index = {}
        for i, string_id in enumerate(sorted(set(ids))):
            id_to_index[string_id] = i + cls.NUM_RESERVED

        return cls(id_to_index)

    # Anything the vocab hasn't seen falls back to UNK and shares its embedding with all other unseen IDs
    def encode_batch(self, string_ids):
        indices = []
        for string_id in string_ids:
            indices.append(self.id_to_index.get(string_id, self.UNK_INDEX))

        return indices

    @property
    def size(self):
        return len(self.id_to_index) + self.NUM_RESERVED

    def save(self, path):
        Path(path).write_text(json.dumps({'id_to_index': self.id_to_index}, indent=2))

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text())['id_to_index'])
