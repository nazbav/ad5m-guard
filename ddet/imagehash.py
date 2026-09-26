"""Перцептивный хеш для поиска почти одинаковых картинок."""
from __future__ import annotations

import numpy as np
from PIL import Image

HASH_SIZE = 16  # 16×16 = 256 бит


def dhash(img: Image.Image, size: int = HASH_SIZE) -> np.ndarray:
    """Разностный хеш: знак горизонтального градиента на уменьшенной серой картинке."""
    g = img.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    a = np.asarray(g, dtype=np.int16)
    return (a[:, 1:] > a[:, :-1]).flatten()


def pack(hashes: list[np.ndarray]) -> np.ndarray:
    """Список bool-хешей → матрица uint64 (n, bits/64) для быстрого сравнения."""
    bits = np.stack(hashes)
    packed = np.packbits(bits, axis=1)
    pad = (-packed.shape[1]) % 8
    if pad:
        packed = np.pad(packed, ((0, 0), (0, pad)))
    return packed.view(np.uint64)


def near_duplicate_pairs(packed: np.ndarray, max_distance: int, chunk: int = 512):
    """Пары индексов (i < j), у которых расстояние Хэмминга не больше max_distance."""
    n = packed.shape[0]
    for start in range(0, n, chunk):
        block = packed[start:start + chunk]
        dist = np.bitwise_count(block[:, None, :] ^ packed[None, :, :]).sum(axis=2)
        ii, jj = np.nonzero(dist <= max_distance)
        for i, j in zip(ii + start, jj):
            if i < j:
                yield int(i), int(j)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)
