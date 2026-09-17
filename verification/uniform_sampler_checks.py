"""Historical reconstruction pilot, not a blind rediscovery experiment.
Python 3 standard library only. Run this file to repeat exact checks.
Assumes independent fair input bits and arbitrary-precision integers.
"""
from fractions import Fraction
from itertools import product
import json
from pathlib import Path


def sample_uniform(n, flip):
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError('n must be a positive integer')
    if n == 1:
        return 0
    size, value = 1, 0
    while True:
        bit = flip()
        if bit not in (0, 1):
            raise ValueError('flip must return 0 or 1')
        size *= 2
        value = 2 * value + bit
        if size >= n:
            if value < n:
                return value
            size -= n
            value -= n


def exact_expected_bits(n):
    # Sum P(T > k) = (2**k mod n)/2**k using the finite remainder cycle.
    remainder, k, total, seen = 1 % n, 0, Fraction(0), {}
    while remainder and remainder not in seen:
        seen[remainder] = (k, total)
        total += Fraction(remainder, 2**k)
        remainder = (2 * remainder) % n
        k += 1
    if not remainder:
        return total
    start, prefix = seen[remainder]
    return prefix + (total - prefix) / (1 - Fraction(1, 2**(k-start)))


def verify_prefixes(n, depth=12):
    # Enumerate all fixed-length input strings, invoking the actual sampler.
    # Exhausted strings represent paths which have not terminated by depth.
    counts = [0] * n
    unfinished = 0
    for bits in product((0, 1), repeat=depth):
        it = iter(bits)
        try:
            value = sample_uniform(n, lambda: next(it))
        except StopIteration:
            unfinished += 1
        else:
            assert 0 <= value < n
            counts[value] += 1
    assert counts == [(2**depth)//n] * n, (n, counts)
    assert unfinished == (2**depth) % n, (n, unfinished)
    return 2**depth


def verify_tree(n, depth=40):
    # Expand only unfinished prefixes, keeping integer multiplicities.
    frontier = {(1, 0): 1} if n > 1 else {}
    completed = [0] * n
    if n == 1:
        completed[0] = 1
    for k in range(1, depth+1):
        completed = [2*x for x in completed]
        next_frontier = {}
        for (size, value), multiplicity in frontier.items():
            for bit in (0, 1):
                s, v = 2*size, 2*value+bit
                if s >= n and v < n:
                    completed[v] += multiplicity
                    continue
                if s >= n:
                    s, v = s-n, v-n
                assert 0 <= v < s < n
                next_frontier[(s,v)] = next_frontier.get((s,v),0)+multiplicity
        frontier = next_frontier
        assert completed == [(2**k)//n] * n
        assert sum(frontier.values()) == (2**k)%n


def main():
    cases = sum(verify_prefixes(n) for n in range(1,33))
    for n in range(1,129):
        verify_tree(n)
    for n in (0, -1, 1.5, True):
        try:
            sample_uniform(n, lambda: 0)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid n accepted')
    assert sample_uniform(1, lambda: (_ for _ in ()).throw(AssertionError())) == 0
    rows = []
    for n in (3,6,10,100,1000):
        k = (n-1).bit_length()
        baseline = Fraction(k * 2**k, n)
        candidate = exact_expected_bits(n)
        rows.append({'n':n, 'restart_bits':float(baseline),
                     'recycling_bits':float(candidate),
                     'recycling_exact':str(candidate),
                     'reduction_percent':float(100*(baseline-candidate)/baseline)})
    result = {'status':'passed', 'complete_input_strings_checked':cases,
              'input_string_length':12, 'n_range_exhaustive':[1,32],
              'tree_n_range':[1,128], 'tree_depth':40,
              'metric':'expected fair input bits per single output; not wall-clock speed',
              'results':rows}
    Path(__file__).with_name('verification_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
