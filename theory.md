# Theory and Measurement Bridge

Date: 2026-06-09

This note formalizes the bridge used by the experiments:

```text
local label roughness under a fixed metric
=> label mass outside the top kernel eigenspace
=> slow/static-kernel or low-margin consequences
```

It is not a final paper proof. It is the engineering version of the theorem that
explains exactly what should be measured.

## Setup

Let `K` be a symmetric positive semidefinite Gram matrix on `n` samples, with
eigendecomposition

```text
K = U diag(lambda_1, ..., lambda_n) U^T,
lambda_1 >= ... >= lambda_n >= 0.
```

Let `P_m = U[:, :m] U[:, :m]^T` be the projector onto the top `m` kernel
eigenvectors, and define the empirical spectral tail

```text
T_y(m) = ||(I - P_m)y||_2^2 / ||y||_2^2.
```

Build a weighted graph `G` from the fixed kernel metric `d_K`, for example a
mutual or directed kNN graph. Let `B` be the weighted incidence matrix, so

```text
||B f||_2^2 = sum_(i,j in E) w_ij (f_i - f_j)^2.
```

For binary labels `y_i in {-1, +1}`, the graph Dirichlet energy is

```text
D_G(y) = ||B y||_2^2 / ||y||_2^2.
```

If `G` contains only opposite-label local edges, then

```text
||B y||_2^2 = 4 * sum_(i,j in E) w_ij.
```

For a full kNN graph, `||B y||_2^2` is exactly four times the total
opposite-label edge weight.

## Finite-Sample Tail Lower Bound

Define the worst low-frequency graph roughness

```text
beta_m(G, K) = ||B P_m||_op^2
             = lambda_max(P_m B^T B P_m).
```

Also define `L_G = B^T B` and `rho_G = ||B||_op = sqrt(lambda_max(L_G))`.

Decompose `y = P_m y + (I - P_m)y`. Then

```text
||B y||_2
  <= ||B P_m y||_2 + ||B(I - P_m)y||_2
  <= sqrt(beta_m) ||y||_2 + rho_G ||(I - P_m)y||_2.
```

Therefore

```text
sqrt(T_y(m))
  >= max(0, sqrt(D_G(y)) - sqrt(beta_m(G,K))) / rho_G.
```

Equivalently,

```text
T_y(m)
  >= [max(0, sqrt(D_G(y)) - sqrt(beta_m(G,K))) / rho_G]^2.
```

This is the clean finite-sample bridge:

- high measured graph label roughness `D_G(y)` forces tail energy unless the top
  kernel eigenspace itself is already rough on the collision graph;
- the bound is non-vacuous when `D_G(y) > beta_m(G,K)`;
- `beta_m` is measurable from the same Gram matrix and graph, so it can be
  audited experimentally.

## Why the Original Collision Bound Was Conservative

The earlier disjoint-pair theorem is a sufficient obstruction. It tries to
control low-frequency variation using only isolated opposite-label pairs and
coarse similarity constants. That makes it easy to state but easy to make
vacuous.

The graph version above is sharper as a measurement principle because it uses:

- all local opposite-label edges rather than a small disjoint subset;
- edge weights rather than binary collision indicators;
- the actual top eigenspace via `beta_m(G,K)`;
- the graph operator norm `rho_G` rather than a worst-case pair constant.

The practical interpretation of experiment `001b` is therefore not "the idea
failed"; it is:

```text
the simple theorem is conservative, while graph roughness is the usable
finite-sample diagnostic.
```

## Multiclass Extension

For multiclass labels, use centered one-hot labels `Y_c in R^{n x C}` and
replace vector norms by Frobenius norms:

```text
T_Y(m) = ||(I - P_m)Y_c||_F^2 / ||Y_c||_F^2
D_G(Y_c) = ||B Y_c||_F^2 / ||Y_c||_F^2.
```

The same proof gives

```text
sqrt(T_Y(m))
  >= max(0, sqrt(D_G(Y_c)) - sqrt(beta_m(G,K))) / rho_G.
```

This justifies using kNN disagreement, normalized local entropy, and graph
Dirichlet energy as multiclass analogues of binary local mixing.

## Consequences

For kernel gradient flow on squared loss,

```text
r(t) = exp(-t K / n) y.
```

If `T_y(m)` is large for a cutoff where `lambda_m` is small, a large component
of the residual lies in slow directions, giving slow decay. This is measured
directly in experiment `009`.

For kernel ridge or minimum-norm interpolation, high tail often increases
regularized source/RKHS norm and reduces margin. This must be read within a
fixed kernel family and regularization context, as shown by experiment `012`.

## Taxonomy

- Local collision obstruction:
  high opposite-label graph roughness under the fixed metric.
- Global/nonlinear spectral misalignment:
  high spectral tail without local collisions, e.g. XOR under a linear kernel.
- Correctable metric mismatch:
  feature learning or a better frozen representation reduces held-out
  roughness/tail and improves margin/accuracy.
- Memorization:
  train roughness/tail improves but held-out geometry or clean accuracy does
  not.
- Intrinsic ambiguity:
  identical or semantically overlapping inputs have contradictory labels, so no
  metric can fully remove the obstruction without changing the label problem.

## Future Measurement

Add a small script that reports `beta_m(G,K)` and the graph lower bound for
experiments `001`, `008`, and `012`. This will quantify when the bridge is
non-vacuous and when graph energy should be treated only as a correlation
diagnostic.
