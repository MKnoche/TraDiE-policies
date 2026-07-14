# Copyright (c) 2026, Markus Knoche. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import io
import pickle
import zlib
from pathlib import Path

import numpy as np


DatasetData = dict[str, dict[str, dict[str, bytes]]]


def pack_npy(a: np.ndarray, *, compress: bool = True) -> bytes:
    """Serialize and compress a NumPy array."""
    bio = io.BytesIO()
    np.save(bio, a, allow_pickle=False)
    raw = bio.getvalue()
    return zlib.compress(raw, level=3) if compress else raw


def unpack_npy(b: bytes, *, compressed: bool = True) -> np.ndarray:
    """Deserialize a packed NumPy array."""
    raw = zlib.decompress(b) if compressed else b
    return np.load(io.BytesIO(raw), allow_pickle=False)


def load_data(path: Path) -> DatasetData:
    """Load a pickled scenario database from disk."""
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data
