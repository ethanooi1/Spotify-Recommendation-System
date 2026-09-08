# Import Libraries
import json
from pathlib import Path


# Maps track/artist/album string IDs to contiguous integers, which become embedding table rows.
# Index 0 is PAD, the empty slot short playlists get padded with and masked out during pooling.
# Index 1 is UNK, where any ID we didn't see in training lands. Real IDs start at 2.
class Vocabulary:
    PAD_INDEX = 0
    UNK_INDEX = 1
    NUM_RESERVED = 2

    def __init__(self, id_to_index, index_to_id):
        self.id_to_index = id_to_index
        self.index_to_id = index_to_id

    @classmethod
    def build(cls, ids):
        id_to_index, index_to_id = {}, {}
        for idx, string_id in enumerate(sorted(set(ids))):
            index = idx + cls.NUM_RESERVED
            id_to_index[string_id] = index
            index_to_id[index] = string_id
        return cls(id_to_index, index_to_id)

    def encode(self, string_id):
        return self.id_to_index.get(string_id, self.UNK_INDEX)

    def encode_batch(self, string_ids):
        return [self.id_to_index.get(sid, self.UNK_INDEX) for sid in string_ids]

    def decode(self, index):
        return self.index_to_id.get(index, None)

    @property
    def size(self):
        return len(self.id_to_index) + self.NUM_RESERVED

    def save(self, path):
        Path(path).write_text(json.dumps({"id_to_index": self.id_to_index}, indent=2))

    @classmethod
    def load(cls, path):
        id_to_index = json.loads(Path(path).read_text())["id_to_index"]
        index_to_id = {int(idx): sid for sid, idx in id_to_index.items()} # JSON keys come back as strings
        return cls(id_to_index, index_to_id)
