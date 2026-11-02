"""Paged KV-cache allocation versus contiguous reservation (Section 15.2).

Teaching example for Chapter 15 of "Secrets of DeepSeek AI". Early serving
systems reserved a contiguous slab for the longest possible answer of every
request; PagedAttention [28] stores the cache in fixed-size blocks that are
allocated on demand, addressed through a per-sequence block table and shared
between requests with a common prefix (reference counts plus copy-on-write).

`PagedKVCache` is a toy allocator: it only does the bookkeeping (which block
holds which tokens), not the attention itself. The simulation compares how
many token slots each policy reserves for the same stream of requests.

Run:
    python paged_kv.py
"""
import numpy as np


class PagedKVCache:
    """Block allocator: block tables, reference counts, copy-on-write."""

    def __init__(self, n_blocks, block=16):
        self.block, self.free = block, list(range(n_blocks))
        self.refs = [0] * n_blocks
        self.tables, self.lens = {}, {}

    def _new_block(self):
        if not self.free:
            raise MemoryError("out of KV blocks: preempt a sequence")
        b = self.free.pop()
        self.refs[b] = 1
        return b

    def append(self, seq, n=1):
        """Reserve room for n more tokens of sequence seq."""
        table = self.tables.setdefault(seq, [])
        used = self.lens.get(seq, 0)
        if table and used % self.block and self.refs[table[-1]] > 1:
            self.refs[table[-1]] -= 1          # shared partial block:
            table[-1] = self._new_block()      # copy-on-write
        while len(table) * self.block < used + n:
            table.append(self._new_block())
        self.lens[seq] = used + n

    def fork(self, parent, child):
        """Let child reuse every block of parent's prefix."""
        self.tables[child] = list(self.tables[parent])
        self.lens[child] = self.lens[parent]
        for b in self.tables[child]:
            self.refs[b] += 1

    def release(self, seq):
        for b in self.tables.pop(seq):
            self.refs[b] -= 1
            if self.refs[b] == 0:
                self.free.append(b)
        del self.lens[seq]

    def used_blocks(self):
        return len(self.refs) - len(self.free)


def simulate(n_requests=64, system=512, max_len=4096, block=16, seed=0):
    """Token slots reserved by three policies for one batch of requests.

    Every request = a shared system prompt + its own question + an answer;
    `max_len` is the context limit a contiguous allocator must reserve.
    """
    rng = np.random.default_rng(seed)
    question = rng.integers(50, 600, n_requests)
    answer = np.minimum(rng.geometric(1 / 300, n_requests), 2000)
    lengths = system + question + answer
    used = int(lengths.sum())

    contiguous = n_requests * max_len

    paged = PagedKVCache(n_blocks=10 ** 5, block=block)
    for i, n in enumerate(lengths):
        paged.append(i, int(n))
    paged_slots = paged.used_blocks() * block

    shared = PagedKVCache(n_blocks=10 ** 5, block=block)
    shared.append("system", system)
    for i in range(n_requests):
        shared.fork("system", i)
        shared.append(i, int(question[i] + answer[i]))
    shared_slots = shared.used_blocks() * block
    unique = used - (n_requests - 1) * system   # tokens actually distinct

    return {"tokens in use": used, "distinct tokens": unique,
            "contiguous, max_len": contiguous,
            f"paged, block {block}": paged_slots,
            f"paged + shared prefix": shared_slots}


if __name__ == "__main__":
    res = simulate()
    used = res["tokens in use"]
    print("64 requests, 512-token shared system prompt, 4096-token limit\n")
    print(f"{'policy':26s} {'token slots':>12s} {'vs tokens in use':>17s}")
    for name, slots in res.items():
        print(f"{name:26s} {slots:12,d} {slots / used:16.1%}")

    # MLA cache of DeepSeek-V3: 70,272 bytes per token (Table 4.1)
    kv = 70_272
    for name in ("contiguous, max_len", "paged, block 16",
                 "paged + shared prefix"):
        print(f"{name:26s} {res[name] * kv / 2**30:8.2f} GiB of MLA cache")
