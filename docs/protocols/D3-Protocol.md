[中文版](D3-Protocol.zh-CN.md)

# D3 Mixed-Force Particle Dynamics Protocol

## 1. Objective

D3 tests whether the TaskKernel-GNN conclusion transfers from scalar
reaction–diffusion dynamics to vector-valued, second-order particle dynamics.
The gating method is frozen before D3: analytic RMS calibration, annealed
temperature softmax, `eta=0.1`, ADD aggregation, and no D3-specific gate
optimization. How to suppress irrelevant kernels is deliberately postponed
until the D3 generality result is known.

`Physics+Residual` is treated as an idealized TaskKernel-GNN with the relevant
analytic force set already selected. The full TaskKernel-GNN should be stable
and should outperform pure MLP and the best single-force model. A remaining gap
to `Physics+Residual` or the learned true force set is evidence of
irrelevant-kernel interference.

## 2. Ground-truth dynamics

The isolated 2D system contains gravity on every unordered particle pair and
spring plus dashpot forces on a sparse undirected bond graph. For
`r_ij = r_j-r_i`, `d_ij=||r_ij||`, `rhat_ij=r_ij/d_ij`,

$$
F_{i\leftarrow j}^{\mathrm{grav}}
=Gm_im_j\frac{r_{ij}}{(d_{ij}^2+\epsilon^2)^{3/2}},
$$

$$
F_{i\leftarrow j}^{\mathrm{spring}}
=\mathbf 1[(i,j)\in E_s],k_{ij}(d_{ij}-\ell_{ij}^0)\widehat r_{ij},
$$

$$
F_{i\leftarrow j}^{\mathrm{dashpot}}
=\mathbf 1[(i,j)\in E_s],\gamma_{ij}
\big[(v_j-v_i)\cdot\widehat r_{ij}\big]\widehat r_{ij}.
$$

Every pair contribution is applied with the opposite sign to particle `j`.
All three message mechanisms conserve total linear momentum. The dashpot
dissipates mechanical energy, so energy is expected to decrease rather than
remain constant.

The node update uses semi-implicit Euler:

$$
v_i^{t+1}=v_i^t+\Delta t\frac{F_i^t}{m_i},
\qquad
r_i^{t+1}=r_i^t+\Delta t,v_i^{t+1}.
$$

The reference time step is chosen by a preflight sweep and then frozen. Initial
positions enforce a minimum separation, total initial momentum is removed, and
the center of mass is translated to the origin. There are no walls or external
forces.

## 3. Candidate kernels

Message candidates are softened inverse-square gravity, Hooke spring, pairwise
dashpot, a Gaussian radial distractor, and an E(2)-equivariant MLP force. The
MLP predicts scalar coefficients from invariant pair features and multiplies
equivariant vector bases. Pair outputs are antisymmetrized so that
`F_ij=-F_ji`. This task enables the optional action–reaction symmetry constraint
even though it remained disabled in the cardiac experiment.

Node-update candidates are semi-implicit Euler, explicit Euler, an incorrect
node-damping update, and an MLP update that preserves the quiescent state. ADD
is the only aggregation operator tested.

## 4. Models

| Model | Meaning in D3 |
|---|---|
| Oracle | True force laws, parameters, bond graph, and integrator |
| Learned-True-Set | Three correct analytic messages and correct update; configurations learned |
| Learned-Single | Validation-selected single force kernel; unable to express the full mixture |
| MLP-only | Equivariant neural messages and neural update without analytic force kernels |
| Physics+Residual | Approximate three-force analytic model plus bounded residual; ideal reduced TaskKernel |
| TaskKernel-GNN | Correct forces, distractor force, and MLP candidates with learned temperature-softmax gates |

External EGNN/GNS baselines are added only after the controlled kernel question
is answered, so architecture differences do not obscure the mechanism test.

## 5. Splits

| Split | Particle count | Parameters/topology |
|---|---:|---|
| Train | 6 and 8 | nominal mass, force, damping, and sparse-bond ranges |
| Validation | 6 and 8 | same distribution, disjoint trajectories |
| ID test | 6 and 8 | same distribution, disjoint trajectories |
| Size-OOD | 12 and 20 | nominal parameters, matched bond-degree distribution |
| Parameter-OOD | 8 | unseen masses, `G`, spring stiffness, rest lengths, and damping |
| Topology-OOD | 8 | chain/ring/branched bond motifs excluded from training |
| SE(2)-OOD | 8 | global rotations, translations, and reflections of ID states |
| Initial-state OOD | 8 | wider position and velocity distributions |
| Long-rollout OOD | 8 | at least twice the training horizon |

The generator first creates a deterministic preflight set. The formal run uses
disjoint seeds, at least three training seeds, and reports every OOD condition
separately.

## 6. Metrics

- position and velocity rollout NRMSE;
- acceleration/force MAE;
- stable rollout horizon and explosion/collision rate;
- total momentum drift;
- mechanical-energy curve error and incorrect energy-increase rate;
- bonded and non-bonded pair-distance error;
- E(2) equivariance error;
- final gate weights, calibrated candidate contribution, and combined
  distractor-plus-MLP contribution.

## 7. Pre-registered interpretation

D3 supports cross-domain generality when all of the following hold:

1. `Learned-True-Set` and `Physics+Residual` are stable and clearly outperform
   `MLP-only`, showing that analytic force kernels replace an unstable or
   data-hungry neural simulator.
2. TaskKernel-GNN remains finite on ID, Size-OOD, Topology-OOD, and SE(2)-OOD,
   and outperforms both MLP-only and Learned-Single on ID plus at least three
   OOD dimensions.
3. The three correct message kernels receive the largest combined calibrated
   contribution and the correct second-order update is selected.
4. If TaskKernel-GNN trails Physics+Residual while retaining the same stability
   and OOD trend, the gap is attributed to candidate interference only when
   distractor/MLP contribution is nontrivial and reproducible across seeds.

Only after these frozen-method tests will D3 be used together with D2-D5a to
optimize how irrelevant message kernels are suppressed.

## 8. Frozen numerical setting after preflight

The generator preflight fixes `G=0.002`, spring stiffness centered at `0.8`,
pairwise dashpot strength centered at `0.18`, gravity softening `epsilon=0.10`,
and semi-implicit Euler step `dt=0.01`. Formal training rollouts cover physical
time `4.0`; long-rollout OOD covers time `8.0`. Initial positions have minimum
separation `0.34`, zero center of mass, and zero total momentum.

All 40 independent particle-count, topology, and long-horizon preflight runs
were finite and remained above the `0.075` close-approach threshold. Total
linear momentum and E(2) force equivariance held to numerical precision, and
mechanical energy decreased in every run. Detailed results are in
`results/Ablation/d3/particles/preflight/REPORT.md`.

## 9. Formal frozen-run status

The three-seed controlled comparison is complete. TaskKernel-GNN remained
finite and outperformed MLP-only and validation-selected Learned-Single on ID
and all nine separately reported OOD datasets. The correct three message
kernels received `0.6600 +/- 0.0039` combined gate weight, while the Gaussian
distractor plus MLP retained `0.3400 +/- 0.0039`. The correct semi-implicit
node update received `98.98%` weight. Physics+Residual remained better on every
split, localizing the current gap to message-candidate interference rather than
node-update selection. Detailed metrics and figures are in
`results/Ablation/d3/particles/formal/REPORT.md`.

