# Controlled operator benchmark

## Goal

Compare fLNO against positive and unconstrained rational pole-residue operators, a diagonal linear state-space model, FNO-1D and DeepONet on the same forcing-to-response task.

The benchmark is designed to distinguish four questions:

1. interpolation on the training-time window;
2. autonomous extrapolation to twice the training horizon;
3. recovery of the structural branching coordinate where that quantity is intrinsic to the architecture;
4. transfer of the same learned size-independent parametrization across graph discretizations.

## Primary task

Ground truth:
[
\lambda'(t)=-c_\lambda\lambda(t)+a(g_{\alpha,\theta}*\lambda)(t)+F(t).
]

The primary generator integrates the causal equation directly. Training forcings are smooth random low-frequency mixtures and pulses that vanish after the training horizon.

## Near-critical sweep

The structured fLNO and positive rational model are compared at several subcritical branching ratios. The experiment emphasizes that finite-horizon prediction error and recovery of the structural critical coordinate are different targets.

## Stability audit

Two diagnostics are intentionally separated:

- nominal repeated fits near criticality;
- a deliberately scarce/noisy signed-initialization stress test.

The latter is not interpreted as a population frequency of instability; it only demonstrates the absence of a by-construction guarantee for the unconstrained realization.

## Graph-size transfer

A size-independent modal law is fitted at n=48 and deployed without retraining at n=96 and n=192.
