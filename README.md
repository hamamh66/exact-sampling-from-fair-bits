# Exact sampling from fair bits

Exact discrete samplers driven by a stream of independent fair coin flips, with proofs, reproducible verification, and a runnable notebook.

Two samplers are implemented. Both are **exact**: the output distribution is the target distribution, not an approximation of it. No floating-point arithmetic appears on any sampling path, and no acceptance test depends on a numerical tolerance.

| Sampler | Problem | Expected cost |
|---|---|---|
| Uniform | draw uniformly from `{0, …, n−1}` | minimal for a single fresh sample |
| Weighted | draw index `i` with probability `w_i / W` | between `H` and `H + 2` bits |

---

## The uniform sampler

The textbook approach draws `⌈log₂ n⌉` bits, accepts if the value is below `n`, and otherwise restarts — discarding everything the rejected draw revealed. This implementation keeps it instead. It maintains a range size `v` and a value `c` uniform within that range; each fresh bit maps `(v, c) → (2v, 2c + b)`. When the range reaches `n` the first `n` values are accepted, and otherwise `n` is subtracted from both, which retains the residual uniformity rather than throwing it away.

The number of unfinished length-`k` prefixes is exactly `2^k mod n`, so

```
E[bits] = Σ_{k ≥ 0} (2^k mod n) / 2^k
```

and this is minimal over all exact fair-bit algorithms for a single fresh sample: by depth `k`, each output can have been allocated at most `⌊2^k/n⌋` prefixes, leaving at least `2^k mod n` unfinished, a bound this sampler attains at every depth.

For six outcomes the expected cost is `11/3` bits against `4` for restarting; for ten outcomes `23/5` against `32/5`. For `n = 3` the two coincide, so retention does not improve every range.

## The weighted sampler

Each target probability `w_i / W` is expanded in binary by integer long division, keeping only the residues `r_i = 2^k w_i mod W`. These define a generating tree level by level: at level `k` the indices whose `k`-th binary digit is 1 become leaves, and the rest of the level stays internal. Sampling walks the tree by `j ← 2j + bit` and stops at the first level where `j ≥ m_k`.

The tree is infinite whenever some `w_i / W` has a non-terminating binary expansion, so it has no finite leaf count. What makes it implementable is that the **live-node count is bounded independently of the total weight**:

```
m_k = (1/W) · Σ_i (2^k w_i mod W)   ≤   n − 1      for k ≥ 1
```

Each residue is at most `W − 1` and the residues sum to a multiple of `W`, so the quotient is at most `n − 1`. The state is therefore the `n` residues, of `O(n log W)` bit complexity — no quantity proportional to the *value* of `W` is ever stored. Individual level widths may depend on `W`; their bound does not.

Proved: exactness, almost-sure termination (`P[D > k] = m_k 2^{-k} ≤ (n−1)2^{-k}`), and `H ≤ E[D] ≤ H + 2` where `H` is the Shannon entropy.

### Reading the expected-cost numbers

`expected_bits_upper` in `src/weighted_sampler.py` is **misnamed**. It returns

```
S_K = Σ_{k=0}^{K} m_k 2^{-k}
```

a truncation of a series with non-negative terms — a *lower* partial sum, never an upper bound, and not the exact infinite expectation when the residue sequence does not terminate. Exact rational arithmetic makes the partial sum exact; it does not remove the truncation.

`src/corrections.py` supplies the certified interval that the proved width bound licenses:

```
S_K  ≤  E[D]  ≤  S_K + (n − 1) 2^{-K}
```

and reports separately when the series terminates, which happens exactly when `W / gcd(w₁…wₙ, W)` is a power of two. In those cases the value is exact: `7/4` for weights `(3, 5)`, `3/2` for `(1, 3)`.

The file `src/weighted_sampler.py` is kept as generated, so the misnomer is documented rather than silently patched.

---

## Verification

```bash
python3 verification/uniform_sampler_checks.py    # uniform sampler
python3 verification/weighted_sampler_checks.py   # weighted sampler
python3 src/corrections.py                        # certified expected-cost intervals
python3 figures/make_figures.py                   # figures (needs matplotlib)
```

Only the Python standard library is required, except for the figures.

**Uniform sampler.** Every bit string of length 12 is fed to the sampler for each `n` from 1 to 32 — 131,072 executions. Each output must occur exactly `⌊4096/n⌋` times and exactly `4096 mod n` strings must remain unfinished. A separate frontier expansion checks depths 1–40 for `n` up to 128, giving 5,120 range–depth checkpoints.

**Weighted sampler.** Checks are written against the public entry point with explicit bit strings, rather than reusing the sampler's own enumeration helpers:

- Exhaustive enumeration to depth 12–14 on five weight vectors. Counts equal `⌊2^d w_i / W⌋` exactly: `(1,2,3)` at depth 14 gives 2730 / 5461 / 8192 with one string unfinished; `(2,7,11,4)` at depth 13 gives 682 / 2389 / 3754 / 1365 with two unfinished. Totals reconcile to `2^d`.
- The level-width identity recomputed from the weights alone and compared against the implementation, over 300 random weight vectors across levels 1–24 — 7,200 comparisons, all matching, with `max(m_k − (n−1)) = 0`.
- The entropy band on 400 random vectors: no violations of `E ≥ H`, none of `E ≤ H + 2`. The largest observed gap is 1.999892. For `(1, 10¹²)` the gap approaches 2 to the precision reported — numerical evidence that the bound is approached, not a proof of tightness.
- Monte Carlo over 200,000 draws: empirical probabilities within 0.002 of exact, measured bits within 0.003 of the expectation.
- Edge cases: a single non-zero weight returns its index consuming no bits; all-zero, empty and negative inputs are rejected.

The notebook reimplements both samplers from their descriptions rather than importing this source, so agreement between the notebook and these scripts is an additional check.

---

## Notebook

`notebook/exact_sampling_from_fair_bits.ipynb` runs everything end to end and plots the results. In Colab it writes outputs to `MyDrive/Outputs/exact-sampling-from-fair-bits/`; run locally it writes to `./Outputs/`.

---

## Scope

Neither sampler is new. The uniform construction corresponds to the Fast Dice Roller; the weighted construction is the Knuth–Yao generating tree in a lazily generated form. The `H ≤ E ≤ H + 2` bound is classical for exact trees in this model. Nothing here is offered as a priority claim, and no systematic prior-art search has been performed.

The Fast Loaded Dice Roller guarantees a sampler linear in the input encoding size with expected consumption at most six bits above optimal. That is a different guarantee with a different computational tradeoff — table-driven sampling in exchange for a weaker entropy bound, where this implementation keeps the exact tree and does `O(n)` work per level. Both figures are *upper bounds*, so their difference does not quantify any actual difference in expected consumption on a given distribution; establishing that would require a distribution-specific comparison, which has not been done.

Fairness and independence of the input bits are assumptions, not verified properties. The expected-cost metric counts fair bits and is not a wall-clock benchmark. The implementations use arbitrary-precision integers; a fixed-width port would need explicit bounds, since the range doubles before comparison.

## References

- D. E. Knuth and A. C. Yao. The complexity of nonuniform random number generation. In *Algorithms and Complexity: New Directions and Recent Results*, Academic Press, 1976, pp. 357–428.
- Y. Horibe. Entropy and an optimal random number transformation. *IEEE Transactions on Information Theory*, 27(1):135–137, 1981.
- L. Devroye. *Non-Uniform Random Variate Generation*. Springer, 1986. Chapter XV, The Random Bit Model.
- J. Lumbroso. Optimal discrete uniform generation from coin flips, and applications. Preprint, 2013.
- F. Saad, C. Freer, M. Rinard and V. Mansinghka. The Fast Loaded Dice Roller: a near-optimal exact sampler for discrete probability distributions. *AISTATS*, PMLR 108:1036–1046, 2020.

## License

MIT — see [LICENSE](LICENSE).
