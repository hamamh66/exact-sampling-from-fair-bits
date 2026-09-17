"""Exact weighted sampling from a stream of fair bits.

Lazily-generated Knuth-Yao style DDG tree; see derivation.md.
Standard library only; exact integer arithmetic on the sampling path.
"""

import random


class BitSource:
    """Fair-bit stream backed by random.getrandbits (counts bits consumed)."""

    def __init__(self, seed=None):
        self._rng = random.Random(seed)
        self.count = 0

    def bit(self):
        self.count += 1
        return self._rng.getrandbits(1)


class ListBitSource:
    """Fair-bit stream replaying a fixed list of bits (for exhaustive testing)."""

    def __init__(self, bits):
        self._bits = bits
        self.count = 0

    def bit(self):
        if self.count >= len(self._bits):
            raise IndexError("bit stream exhausted")
        b = self._bits[self.count]
        self.count += 1
        return b


class ExactWeightedSampler:
    """Preprocess weights once, then sample indices with probability w_i / W."""

    def __init__(self, weights, kmax=64):
        w = [int(x) for x in weights]
        if any(x < 0 for x in w):
            raise ValueError("weights must be non-negative")
        W = sum(w)
        if W <= 0:
            raise ValueError("weights must not be all zero")
        self.w = w
        self.W = W
        self.n = len(w)
        self.kmax = kmax
        support = [i for i, x in enumerate(w) if x > 0]
        self.degenerate = support[0] if len(support) == 1 else None
        # residues r_i^(k) = 2^k w_i mod W, advanced level by level
        self._r = list(w)
        self._m = 1                     # m_0
        self._levels = []               # _levels[k-1] = (leaves_k, m_k)
        self._ckpt_r = None             # residues at level kmax
        self._ckpt_m = None

    # ---- level machinery (deterministic, no randomness) -------------------

    @staticmethod
    def _advance(r, m, W, n):
        """One long-division step: returns (leaves, new_m); mutates r in place."""
        leaves = []
        for i in range(n):
            t = r[i] << 1
            if t >= W:
                t -= W
                leaves.append(i)
            r[i] = t
        return leaves, (m << 1) - len(leaves)

    def _level(self, k):
        """Return (leaves_k, m_k) for 1 <= k <= kmax, memoised."""
        while len(self._levels) < k:
            leaves, self._m = self._advance(self._r, self._m, self.W, self.n)
            self._levels.append((leaves, self._m))
            if len(self._levels) == self.kmax:
                self._ckpt_r = list(self._r)
                self._ckpt_m = self._m
        return self._levels[k - 1]

    # ---- sampling --------------------------------------------------------

    def sample(self, src):
        if self.degenerate is not None:
            return self.degenerate            # 0 bits
        j = 0
        k = 0
        r = m = None                          # deep-tail scratch state
        while True:
            k += 1
            j = (j << 1) | src.bit()
            if k <= self.kmax:
                leaves, m_k = self._level(k)
            else:
                if r is None:                 # enter deep tail from checkpoint
                    self._level(self.kmax)
                    r = list(self._ckpt_r)
                    m = self._ckpt_m
                leaves, m = self._advance(r, m, self.W, self.n)
                m_k = m
            if j >= m_k:
                return leaves[j - m_k]

    def expected_bits_upper(self, depth=200):
        """Sum_{k<=depth} m_k 2^-k as an exact Fraction-free rational pair."""
        from fractions import Fraction
        if self.degenerate is not None:
            return Fraction(0)
        r = list(self.w)
        m = 1
        total = Fraction(m)                   # k = 0 term
        for k in range(1, depth + 1):
            _, m = self._advance(r, m, self.W, self.n)
            total += Fraction(m, 1 << k)
        return total


def enumerate_counts(sampler, depth):
    """Exhaustively run every bit string of length `depth`.

    Returns (counts dict index->#strings decided as i, undecided count).
    Done by recursion over the tree rather than 2^depth explicit strings.
    """
    counts = {}
    undecided = 0
    if sampler.degenerate is not None:
        return {sampler.degenerate: 1 << depth}, 0
    # frontier: list of node indices alive at current level -> all of them, so
    # we just track how many strings sit at each alive node (all equal weight).
    # Alive nodes at level k are 0..m_k-1, each carrying 2^(depth-k) strings.
    r = list(sampler.w)
    m = 1
    for k in range(1, depth + 1):
        prev_m = m
        leaves, m = sampler._advance(r, m, sampler.W, sampler.n)
        # nodes m..2*prev_m-1 are leaves, in order of `leaves`
        for idx, lab in enumerate(leaves):
            counts[lab] = counts.get(lab, 0) + (1 << (depth - k))
        undecided = m << (depth - k)
    if depth == 0:
        undecided = 1
    return counts, undecided


if __name__ == "__main__":
    s = ExactWeightedSampler([1, 2, 3])
    src = BitSource(0)
    from collections import Counter
    c = Counter(s.sample(src) for _ in range(60000))
    print(c, src.count / 60000.0)
