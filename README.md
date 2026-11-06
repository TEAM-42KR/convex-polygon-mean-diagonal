# The mean diagonal of a random convex polygon of fixed perimeter: reproduction programs

Programs that recompute and check the numerical statements of the paper

> Sungsoo Na, *The mean diagonal of a random convex polygon of fixed perimeter*,
> version 1.0.0, Zenodo, 2026. https://doi.org/10.5281/zenodo.24026929

Every script prints one `PASS`/`FAIL` line per check, with the tolerance used,
and writes a JSON record `out/checks/<script>.json` (checks, values, runtime,
versions, seeds). Inputs are in `data/`; everything generated goes to `out/`,
which is created on demand.

## Model

A random closed polygon with perimeter 1 under the symmetric measure: take a
Haar-distributed orthonormal 2-frame Q in R^n (obtained from an n x 2 standard
Gaussian matrix G as Q = G (G^T G)^(-1/2)), set z_k = Q_k1 + i Q_k2 and use the
edges e_k = z_k^2/2. Then sum e_k = 0 and sum |e_k| = 1. Conditioning on
convexity amounts to traversing the edges in angular order, so sorting the
edges by arg(e_k) samples the convex polygon exactly.

- M_n: expected mean of the n(n-3)/2 diagonals.
- d(n,k): expected length of k consecutive sorted edges; M_n = (1/(n-3)) sum_{k=2}^{n-2} d(n,k).
- Large-n limit 2/pi^2, first correction c1 = (1 + 7 zeta(3)/2)/pi^2.

## Scripts

| script | what it checks | runtime |
|---|---|---|
| `sorted_sampler.py` | Monte Carlo M_n (n = 4..12, 16, 20, 30, 40) and d(n,k) (n <= 8) with the exact sorted sampler; z-scores against the exact values for n = 4..8 (pass: abs(z) < 4). Writes `out/mc_scaled.dat`. | ~1.5 min |
| `chord_selberg.py` | Unconditional chord mean (m0 m2 - m1^2)/S: exact values pi^2/32, 4/15, 3pi^2/128, 1/4 at 50 digits; symmetry k <-> n-k; Monte Carlo for n = 5, 6, 7, all k. | ~5 s |
| `gap_void.py` | Gap-void functions H_2..H_8 from the derivative recurrence; agreement with the closed forms of H_3 and H_4 (1e-12), H_m(pi/2) = 0, normalisation of H_m'', Monte Carlo at four interval lengths. Writes `out/Hm.dat`. | ~10 s |
| `finite_n_reduction.py` | Double-precision evaluation of the finite-n integral reduction for d(5,2), d(6,2), d(6,3), d(7,2), d(7,3) and M_5, M_6, M_7 against the closed forms (tolerance 1e-7; observed about 1e-12). | 1-4 min |
| `asymptotic_c1.py` | The c1 integral equals (1 + 7 zeta(3)/2)/pi^2, and int_0^pi a(pi-a)/sin a da = 7 zeta(3), both to 50 digits. | < 1 s |
| `asymptotic_c2c3.py` | Formal expansion 6.2. The coefficient functions k_0..k_3 of the bulk expansion at fixed alpha (delta method for the unconditioned chord of Lemma 6.1, exact finite polynomial algebra in six variables) are integrated over alpha with Gauss-Legendre rules of 96 and 192 nodes at 80 digits (the rules agree to about 1e-75). Checks: self-test of the machinery on tau/(tau+sigma), whose expansion is known exactly; agreement of the two rules; delta_0 = 2/pi^2; c1 against Theorem 1.3 and c2, c3 against the printed closed forms (tolerance 1e-60; observed 1e-75 to 1e-80); the printed decimals of c2, c3. | ~20 s |
| `exact_reductions.py` | Sections 7.4 and 7.5. Direct evaluation of the unreduced finite sums produced by the exact reductions: d(6,2) 4599 terms (weight <= 6), d(6,3) 6517 terms, d(7,2) 1232 terms, d(7,3) 3371 terms (weight <= 5), in 480-bit fixed-point arithmetic and without the relation tables used for the reduction. Checks: evaluator self-tests on classical values; d(6,2), d(6,3), d(7,2), d(7,3) against the printed reduced forms and M_6 = (2 d(6,2) + d(6,3))/3, M_7 = (d(7,2) + d(7,3))/2 against Theorem 1.5 (tolerance 1e-100; observed about 1e-140; imaginary parts below 1e-139); the 25 printed digits of M_6, M_7 in Table 1.1. | ~20 s |
| `quadrilateral_trichotomy.py` | For 10^5 closed 4-edge sets from the symmetric law and from an asymmetric exchangeable law, the three cyclic orders give exactly one convex, one simple non-convex and one self-intersecting quadrilateral; the sign criterion for which edge swap stays simple. | ~2 s |
| `simple_pentagon.py` | The integral J to 50 digits, P(simple pentagon) = 1/2 - 15J/(4pi^2) = 7/24 + (10 - 10 log 2 - (5/2) log^2 2)/pi^2, and a Monte Carlo check with 10^6 pentagons. | ~3 s |
| `figure_data.py` | Writes `out/polygon8.dat` (one sampled convex octagon) and `out/exact_scaled.dat` (n (M_n - 2/pi^2), n = 4..8, from the exact values). | < 1 s |
| `printed_formulas.py` | The closed forms as printed in the paper, evaluated at 140 digits: M_8 and d(8,2), d(8,3), d(8,4) against a 98-digit reference (>= 95 digits) and against each other; M_4..M_7, d(6,2), d(6,3), d(7,2), d(7,3) (>= 50 digits) with the relations M_6 = (2 d(6,2) + d(6,3))/3 and M_7 = (d(7,2) + d(7,3))/2; c_1, c_2, c_3; the simple-pentagon probability and the simple-quadrilateral mean; every printed decimal value. Alternating multiple zeta values from PARI/GP (`polylogmult`, `zetamult`) with the definitions of Appendix A, including the table of values there. | < 1 s |
| `alphabet_check.py` | Section 4.2 by computer algebra: the letters in y of the threefold integral (4.2) are {0, +-i, +-ix, +-i/x, +-is, +-i/s}; the singularity factors of the y-integration, factored over Q(i), are exactly {x, x+-1, x+-i, x+-s, sx+-1}; after the x-integration the factors are among {s, s+-1, s+-i}, all linear. | ~2 s |
| `fixed_dimension.py` | Section 5: the contour formula (5.1) for H_m against the closed forms of H_3 (three points) and H_4 (one point) at 25 digits; the change of variables of Lemma 5.2 at random points; the outer formula of Theorem 5.3 with exact H_2, H_3, H_3'' for d(4,2), d(5,2), d(6,3) (tolerance 1e-12); the integrand of Proposition 5.4 (bounds, O(eta^2) cancellation, n-series against the Psi form). | ~30 s |
| `fixed_dim_mini.py` | Section 5.3, end to end: runs the evaluator in `fixed_dim/` (H_m, H_k'' from (5.1) in ball arithmetic, then the outer quadrature of Theorem 5.3) with its one-core profile `mini` and checks M_4..M_8 against the closed forms and M_9, M_10 against Table 1.1 (tolerance 1e-15; reaches about 17 digits). Also checks the stored values `data/fixed_dim_values.json` (70 digits) against the references (54 / 60 digits). | ~4.5 min |

Helper modules: `common.py` (Haar frames, segment-intersection test, check
records, table writer) and `hm_core.py` (tabulation of H_m, H_m', H_m'' by
piecewise Chebyshev interpolation on a mesh graded toward pi/2). `fixed_dim/`
holds the evaluator of Section 5.3: `hcorner.py` (H_m and two x-derivatives
from (5.1) by dual numbers, polar coordinates at the corners), `outer.py` (the
quadrature of Theorem 5.3) and `prod.py` (Chebyshev tables, outer stage,
holdout check; profiles `prod`, `test`, `mini`).

## Data

- `data/reference_values.json`: closed-form values of M_4..M_8 and of the chord means d(n,k), n = 6, 7, 8, to 55 digits.
- `data/fixed_dim_values.json`: d(n,k) and M_n, 4 <= n <= 10, from the fixed-dimension representation (profile `prod` of `fixed_dim/prod.py`, 64 cores), with the digits of agreement.
- `data/d62_sum.txt.gz`, `data/d63_sum.txt.gz`, `data/d72_sum.txt.gz`, `data/d73_sum.txt.gz`: the finite sums obtained by exact iterated integration of the reduction integrals for d(6,2), d(6,3), d(7,2), d(7,3), before any shuffle factorization or reduction by relations (input of `exact_reductions.py`). One term per line, `Re(c) Im(c) w`: c in Q(i) as exact rationals and w a word in the letters 0, -1, i, -i, -1/2, (-1+i)/2, (-1-i)/2 (codes 0..6); the term value is c L(w), with L(w) the regularized value at +infinity of the hyperlogarithm G(w; z). The file headers state the format and the normalization of each chord mean.
- `data/ladder_values.dat`: high-precision values of M_n from the finite-n reduction, with the number of trusted digits (Table 1.1 of the paper).

## Output tables (`out/`, whitespace columns, one header line)

- `mc_scaled.dat`: `n scaled se`, scaled = n (M_n - 2/pi^2) from Monte Carlo, se = its standard error.
- `exact_scaled.dat`: `n scaled` from the exact values, n = 4..8.
- `Hm.dat`: `x H2 ... H8` on 121 equally spaced points of [0, pi/2].
- `polygon8.dat`: `i x y`, vertices of the octagon; the last row repeats the first vertex.

## Requirements

- Python 3 with numpy and mpmath (tested with Python 3.14, numpy 2.5, mpmath 1.3)
- SymPy for `alphabet_check.py` (tested with 1.14)
- python-flint for `fixed_dim_mini.py` (tested with 0.9)
- PARI/GP for `printed_formulas.py` (`gp` on the PATH; version 2.15 or later, tested with 2.17.4)

Everything runs single-threaded.

## Running

    ./run_all.sh            # all scripts, summary at the end; exit code 0 iff all checks pass
    python3 gap_void.py     # any script can also be run on its own

The total runtime is 8-13 minutes on one core, of which `fixed_dim_mini.py`
takes about 4.5 minutes. Monte Carlo checks use fixed seeds; their pass
criterion is abs(z) < 4 against the exact value, so a different platform may
give different but still passing z-scores. `SHA256SUMS` lists the checksums
of the distributed files.

## License

MIT, see `LICENSE`.
