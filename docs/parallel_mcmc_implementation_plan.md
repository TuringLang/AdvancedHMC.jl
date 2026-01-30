# Parallel MCMC Implementation Plan

**Target**: AdvancedHMC.jl
**Based on**: Zoltowski et al., "Parallelizing MCMC Across the Sequence Length" (NeurIPS 2025)
**Status**: 🚧 In Progress

---

## Overview

This document tracks the implementation of DEER (Doubly Efficient Estimation via Recursion) and its variants for parallelizing MCMC sampling across chain length in AdvancedHMC.jl.

**Goal**: Enable O(log T) time complexity per Newton iteration instead of O(T) sequential steps.

---

## Implementation Phases

### Phase 1: Core Infrastructure ✅
> Foundation components needed by all DEER variants

- [x] **1.1 Parallel Scan Implementation** ✅
  - [x] Associative operator for affine transforms: `(J₂, u₂) ⊕ (J₁, u₁) = (J₂*J₁, J₂*u₁ + u₂)`
  - [x] Full matrix version (for DEER) - `MatrixAffineTransform`
  - [x] Diagonal version (for Quasi-DEER) - `DiagonalAffineTransform`
  - [x] Block-diagonal version (for Block Quasi-DEER / leapfrog) - `Block2x2AffineTransform`
  - [x] CPU implementation using `Base.accumulate`
  - [ ] GPU implementation (CUDA.jl) - deferred to Phase 6

- [x] **1.2 Jacobian Computation Utilities** ✅
  - [x] Full Jacobian computation (finite differences, AD-agnostic interface)
  - [x] Diagonal extraction from Jacobian
  - [x] Stochastic diagonal estimation (Hutchinson's method with Rademacher vectors)
  - [x] JVP (Jacobian-vector product) wrapper
  - [x] VJP (Vector-Jacobian product) wrapper
  - [x] Batch versions for computing at multiple points
  - [x] Hessian diagonal utilities (for leapfrog)

- [x] **1.3 Core Types and Interfaces** ✅
  - [x] `AbstractParallelMethod` with subtypes: `FullDEER`, `QuasiDEER`, `StochasticQuasiDEER`, `BlockQuasiDEER`
  - [x] `ParallelMCMCState` to hold trajectory and convergence info
  - [x] `ParallelMCMCSettings` configuration struct

---

### Phase 2: DEER Algorithm Core ✅
> The main Newton iteration loop

- [x] **2.1 Newton Iteration Framework** ✅
  - [x] Main loop structure with convergence checking
  - [x] Parallel evaluation of f_t at all timesteps
  - [x] Parallel computation of Jacobians (method-dependent)
  - [x] Compute inputs u_t = f_t(s_{t-1}) - J_t * s_{t-1}
  - [x] Solve linear system via parallel scan
  - [x] Convergence criterion (max absolute difference)
  - [x] DEERResult type with trajectory, convergence info, residual history

- [x] **2.2 Full DEER Implementation** ✅
  - [x] Store and manipulate T × D × D Jacobian matrices
  - [x] Full matrix parallel scan

- [x] **2.3 Quasi-DEER Implementation** ✅
  - [x] Diagonal Jacobian storage (T × D)
  - [x] Elementwise parallel scan

- [x] **2.4 Stochastic Quasi-DEER Implementation** ✅
  - [x] Hutchinson estimator with configurable number of samples
  - [x] Rademacher vector generation (from Phase 1.2)
  - [x] JVP-based diagonal estimation

---

### Phase 3: MALA Integration ✅
> Parallelized Metropolis-Adjusted Langevin Algorithm

- [x] **3.1 MALA Transition Function** ✅
  - [x] Proposal: x̃ = x + ε∇log p(x) + √(2ε)ξ
  - [x] Acceptance ratio computation (forward and backward proposal densities)
  - [x] Stop-gradient trick for differentiable accept-reject
  - [x] Soft gating with sigmoid and straight-through estimator

- [x] **3.2 Parallel MALA Sampler** ✅
  - [x] MALARandomInputs type for pre-sampled (ξ, u) pairs
  - [x] sample_mala_inputs() for batch sampling
  - [x] parallel_mala() integrating with DEER framework
  - [x] sequential_mala() for reference/testing
  - [x] Convenience API with automatic input sampling

- [ ] **3.3 MALA-specific Optimizations** (deferred)
  - [ ] Preconditioning with Hessian eigendecomposition (optional)

---

### Phase 4: HMC Integration ✅
> Two approaches for parallelizing HMC

- [x] **4.1 Approach A: Parallelize Across HMC Steps** ✅
  - [x] Treat full HMC step as transition function f_t
  - [x] Sequential leapfrog within each step
  - [x] Parallel Newton across T HMC samples
  - [x] Good when T >> L (many samples, few leapfrog steps)

- [x] **4.2 Approach B: Parallelize Leapfrog Integration** ✅
  - [x] State is s = [x, v] (position + momentum)
  - [x] Apply DEER to L leapfrog steps within each HMC step
  - [x] Block Quasi-DEER with 2×2 block structure per dimension
  - [x] Good when L is large

- [x] **4.3 Block Quasi-DEER for Leapfrog** ✅
  - [x] Block Jacobian structure:
    ```
    J = [ I_D          ε*M⁻¹        ]
        [ ε*diag(H)    I_D + ε²*M⁻¹*diag(H) ]
    ```
  - [x] Efficient 2×2 block scan per dimension
  - [x] Hessian diagonal computation (hessian_diagonal_fd)

- [x] **4.4 Accept-Reject for HMC** ✅
  - [x] Soft gating with sigmoid (hmc_transition_soft)
  - [x] Momentum refresh handling (HMCRandomInputs)

---

### Phase 5: AdvancedHMC.jl Integration ✅
> Integrate with existing library architecture

- [x] **5.1 New Sampler Types** ✅
  - [x] `ParallelHMCSampler <: AbstractParallelSampler`
  - [x] `ParallelMALASampler <: AbstractParallelSampler`
  - [x] Convenience aliases: `ParallelHMC`, `ParallelMALA`

- [x] **5.2 AbstractMCMC Interface** ✅
  - [x] Implement `parallel_sample()` for batch sampling
  - [x] `ParallelSamplerState` with trajectory and convergence info
  - [x] Iterator interface for `for sample in state` patterns
  - [x] Handle RNG properly for reproducibility

- [x] **5.3 Integration with Existing Components** ✅
  - [x] Use existing `Metric` types (DiagEuclideanMetric, etc.)
  - [x] `SimpleLogDensity` wrapper implementing LogDensityProblems interface
  - [x] Compatible with existing gradient computation patterns
  - [x] Standalone testing mode (works without full AdvancedHMC)

---

### Phase 6: Advanced Features ⬜
> Optimizations and extensions

- [ ] **6.1 Sliding Window**
  - [ ] For memory-constrained high-dimensional problems
  - [ ] Window initialization and shifting logic
  - [ ] Convergence tracking per window

- [ ] **6.2 Early Stopping**
  - [ ] Return approximate samples before full convergence
  - [ ] Quality-vs-time tradeoff parameter

- [ ] **6.3 GPU Support**
  - [ ] CUDA.jl parallel scan kernel
  - [ ] GPU-compatible transition functions
  - [ ] Batch Jacobian computation on GPU

- [ ] **6.4 Diagnostics and Monitoring**
  - [ ] Iteration count tracking
  - [ ] Convergence history
  - [ ] Per-timestep residual monitoring

---

### Phase 7: Testing and Validation ⬜
> Ensure correctness and performance

- [ ] **7.1 Unit Tests**
  - [ ] Parallel scan correctness (compare to sequential)
  - [ ] Jacobian computation accuracy
  - [ ] Hutchinson estimator convergence

- [ ] **7.2 Integration Tests**
  - [ ] DEER converges to sequential MALA trace
  - [ ] DEER converges to sequential HMC trace
  - [ ] Quasi-DEER matches DEER (eventually)

- [ ] **7.3 Statistical Validation**
  - [ ] Samples match target distribution (known distributions)
  - [ ] Compare sample quality metrics (ESS, R-hat) to sequential

- [ ] **7.4 Performance Benchmarks**
  - [ ] Wall-clock time vs sequential
  - [ ] Scaling with chain length T
  - [ ] Memory usage profiling

---

## File Structure

```
src/
├── parallel/
│   ├── Parallel.jl          # Module definition and exports ✅
│   ├── types.jl             # Types and settings ✅
│   ├── scan.jl              # Parallel scan implementations ✅
│   ├── jacobian.jl          # Jacobian computation utilities ✅
│   ├── deer.jl              # Core DEER algorithm ✅
│   ├── mala.jl              # Parallel MALA ✅
│   ├── hmc.jl               # Parallel HMC ✅
│   └── abstractmcmc.jl      # AbstractMCMC integration ✅
test/
├── parallel/
│   ├── test_scan.jl         # ✅ 141 tests passing
│   ├── test_jacobian.jl     # ✅ 57 tests passing
│   ├── test_deer.jl         # ✅ 67 tests passing
│   ├── test_mala.jl         # ✅ 31 tests passing
│   ├── test_hmc.jl          # ✅ 49 tests passing
│   └── test_abstractmcmc.jl # ✅ 75 tests passing
```

---

## Dependencies

**Required:**
- ForwardDiff.jl - Jacobian computation
- LinearAlgebra - Matrix operations

**Optional:**
- CUDA.jl - GPU acceleration
- Zygote.jl / Enzyme.jl - Alternative AD for JVPs
- ChainRulesCore.jl - Custom rrules for stop-gradient

---

## Implementation Order (Recommended)

1. **Start with Phase 1.1** - Parallel scan is the foundation
2. **Then Phase 1.2-1.3** - Jacobian utilities and types
3. **Phase 2** - Core DEER loop (test with simple linear system first)
4. **Phase 3** - MALA (simpler than HMC, good first MCMC target)
5. **Phase 4** - HMC integration
6. **Phase 5** - AdvancedHMC.jl integration
7. **Phase 6-7** - Advanced features and thorough testing

---

## Progress Log

| Date | Phase | Item | Status | Notes |
|------|-------|------|--------|-------|
| 2026-01-20 | - | Initial plan created | ✅ | Based on implementation note |
| 2026-01-20 | 1.1 | Parallel scan implementation | ✅ | All 3 variants (matrix, diagonal, block 2x2) |
| 2026-01-20 | 1.3 | Core types and interfaces | ✅ | Types for methods, settings, state |
| 2026-01-20 | 7.1 | Parallel scan unit tests | ✅ | 141 tests passing |
| 2026-01-20 | 1.2 | Jacobian computation utilities | ✅ | FD-based, Hutchinson estimator, JVP/VJP |
| 2026-01-20 | 7.1 | Jacobian utility tests | ✅ | 57 tests passing |
| 2026-01-20 | 1 | **Phase 1 Complete** | ✅ | All core infrastructure done |
| 2026-01-20 | 2 | DEER algorithm core | ✅ | Newton iteration, Full/Quasi/Stochastic DEER |
| 2026-01-20 | 7.1 | DEER algorithm tests | ✅ | 67 tests passing |
| 2026-01-20 | 2 | **Phase 2 Complete** | ✅ | Core DEER algorithm working |
| 2026-01-20 | 3 | MALA transition function | ✅ | Proposal, acceptance, soft gating |
| 2026-01-20 | 3 | Parallel MALA sampler | ✅ | Integrated with DEER framework |
| 2026-01-20 | 7.1 | MALA tests | ✅ | 31 tests passing |
| 2026-01-20 | 3 | **Phase 3 Complete** | ✅ | Parallel MALA working |
| 2026-01-29 | 4.1 | Approach A: Parallelize HMC steps | ✅ | parallel_hmc(), soft MH gating |
| 2026-01-29 | 4.2 | Approach B: Parallelize leapfrog | ✅ | parallel_leapfrog(), leapfrog_transition() |
| 2026-01-29 | 4.3 | Block Quasi-DEER for leapfrog | ✅ | 2×2 block structure, hessian_diagonal_fd |
| 2026-01-29 | 7.1 | HMC tests | ✅ | 49 tests passing |
| 2026-01-29 | 4 | **Phase 4 Complete** | ✅ | Parallel HMC working (345 total tests) |
| 2026-01-30 | 5.1 | New sampler types | ✅ | ParallelHMCSampler, ParallelMALASampler |
| 2026-01-30 | 5.2 | AbstractMCMC interface | ✅ | parallel_sample(), ParallelSamplerState |
| 2026-01-30 | 5.3 | Integration with components | ✅ | Metric types, LogDensityProblems |
| 2026-01-30 | 7.1 | AbstractMCMC tests | ✅ | 75 tests passing |
| 2026-01-30 | 5 | **Phase 5 Complete** | ✅ | AdvancedHMC.jl integration (420 total tests) |

---

## Open Questions

1. **AD Backend Choice**: ForwardDiff for full Jacobians, but what for JVPs in stochastic quasi-DEER? Zygote vs Enzyme?
   - **Decision**: Use whichever works best; start with ForwardDiff

2. **GPU Priority**: Should GPU support be Phase 1 or deferred to Phase 6?
   - **Decision**: Deferred to Phase 6; focus on CPU correctness first

3. **Integration Approach**: Should we modify existing HMC/Leapfrog types or create entirely new parallel variants?
   - **Decision**: Be compatible with existing abstractions and API design as much as possible

4. **Testing Strategy**: What target distributions should we use for validation? Gaussian (analytical), banana, Neal's funnel?

5. **API Design**: Return full chain always, or support incremental parallel blocks?

---

## References

- Paper: Zoltowski et al., "Parallelizing MCMC Across the Sequence Length", NeurIPS 2025
- Reference implementation: https://github.com/lindermanlab/parallel-mcmc (JAX)
- DEER theory: Lim et al., Danieli et al.
- Hutchinson estimator: Hutchinson (1990), Bekas et al.
