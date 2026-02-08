# Implementation Note: Parallelizing MCMC Across Sequence Length

**Target Library**: [AdvancedHMC.jl](https://github.com/TuringLang/AdvancedHMC.jl)
**Reference Paper**: Zoltowski et al., "Parallelizing MCMC Across the Sequence Length" (NeurIPS 2025)
**Reference Implementation**: https://github.com/lindermanlab/parallel-mcmc (JAX)

* * *

## Executive Summary

This paper presents algorithms to evaluate MCMC samplers in **parallel across the chain length**, achieving sublinear time complexity. Instead of running chains sequentially (O(T) time for T samples), the approach reformulates the sampling process as solving a fixed-point problem via parallel Newton iterations, achieving O(log T) time per iteration on parallel hardware.

Key results from the paper:

  - Generate 100K HMC samples with only ~147 parallel iterations (vs 100K sequential steps)
  - Up to 30× speedup over sequential MALA in some configurations
  - Works for Gibbs sampling, MALA, and HMC

* * *

## 1. Core Algorithm: DEER (Parallel Newton's Method)

### 1.1 Problem Formulation

Any MCMC chain can be viewed as a nonlinear recursion:

```
s_t = f_t(s_{t-1})
```

where:

  - `s_t` is the state at step t (e.g., the MCMC sample)
  - `f_t` is the transition function (varies with t due to different random inputs ω_t at each step)
  - `s_0` is the initial state

The key insight is that the entire sequence `s_{1:T}` is the **fixed point** of the residual:

```
r(s_{1:T}) = [s_1 - f_1(s_0), s_2 - f_2(s_1), ..., s_T - f_T(s_{T-1})]
```

### 1.2 Newton Update (Equation 2)

At iteration i, compute the next state trajectory via:

```
s_t^{(i+1)} = f_t(s_{t-1}^{(i)}) + J_t * (s_{t-1}^{(i+1)} - s_{t-1}^{(i)})
```

where `J_t = ∂f_t/∂s (s_{t-1}^{(i)})` is the Jacobian of f_t evaluated at the current iterate.

### 1.3 Rearranged as Linear Recursion (Equation 3)

Rearranging:

```
s_t^{(i+1)} = J_t * s_{t-1}^{(i+1)} + u_t

where u_t = f_t(s_{t-1}^{(i)}) - J_t * s_{t-1}^{(i)}
```

This is a **linear dynamical system** with time-varying Jacobians `J_t` and inputs `u_t`.

**Critical**: Since `J_t` and `u_t` only depend on the previous Newton iteration's states, they can be computed **in parallel** across all t.

### 1.4 Parallel Scan

The linear system can be solved in O(log T) time using the **associative scan** algorithm:

```julia
# Pseudocode for parallel scan of linear recursion
# Given: J[1:T] (Jacobians), u[1:T] (inputs), s_0 (initial state)
# Output: s[1:T]

# Key insight: composition of affine transforms is affine
# If h_t(x) = J_t * x + u_t, then h_{t+1} ∘ h_t is also affine

function parallel_scan(J, u, s0)
    # Represent each transform as (J_t, u_t) pair
    # Associative binary operator: (J2, u2) ⊕ (J1, u1) = (J2*J1, J2*u1 + u2)

    # Use tree-reduction pattern:
    # Level 0: [(J1,u1), (J2,u2), (J3,u3), (J4,u4), ...]
    # Level 1: [(J2*J1, J2*u1+u2), (J4*J3, J4*u3+u4), ...]
    # Level 2: [(...), ...]
    # After log2(T) levels, have cumulative transforms

    # Then apply to s0 to get all states
end
```

Julia already has `accumulate` which can be adapted, but for GPU efficiency you'd want a proper parallel scan implementation (potentially using CUDA.jl or similar).

* * *

## 2. Algorithm Variants

### 2.1 Full DEER (Newton's Method)

  - Store full D×D Jacobian matrices at each timestep
  - **Memory**: O(T * D²)
  - **Work per iteration**: O(T * D³) for matrix multiplications
  - **Iterations**: Typically converges in tens of iterations
  - **Best for**: Low-dimensional problems

### 2.2 Quasi-DEER (Diagonal Approximation, Equation 4)

Replace full Jacobian with its diagonal:

```
s_t^{(i+1)} = diag(J_t) ⊙ s_{t-1}^{(i+1)} + u_t
```

  - **Memory**: O(T * D)
  - **Work per iteration**: O(T * D)
  - **Iterations**: More iterations than DEER, but each is cheaper
  - **Best for**: High-dimensional problems

### 2.3 Stochastic Quasi-DEER (Section 3.4)

Estimate diagonal via Hutchinson's method:

```
diag(J_t) ≈ E[z ⊙ (J_t * z)]  where z ~ Rademacher
```

**Key advantage**: Only requires a single forward pass through f_t plus one Jacobian-vector product (JVP), rather than D passes to compute the full diagonal.

```julia
# Pseudocode for stochastic diagonal estimation
function estimate_jacobian_diagonal(f, s, n_samples=1)
    D = length(s)
    diag_estimate = zeros(D)
    for _ in 1:n_samples
        z = rand([-1, 1], D)  # Rademacher
        Jz = jvp(f, s, z)      # Jacobian-vector product
        diag_estimate .+= z .* Jz
    end
    return diag_estimate ./ n_samples
end
```

In practice, **1-3 Monte Carlo samples often suffice**.

### 2.4 Block Quasi-DEER (for Leapfrog Integration)

For HMC's leapfrog integrator, the state is `s = [x, v]` (position and momentum). The Jacobian has block structure (Equation 9):

```
J = [ I_D          ε*I_D                              ]
    [ ε*H          I_D + ε²*H                         ]

where H = ∇²_x log p(x_{t-1} + ε*v_{t-1})  (Hessian)
```

Block quasi-DEER keeps the diagonal of each 2×2 block:

```
J_block_diag = [ I_D              ε*I_D         ]
               [ ε*diag(H)        I_D + ε²*diag(H) ]
```

This accounts for position-momentum interactions and converges ~2× faster than standard diagonal quasi-DEER for leapfrog.

* * *

## 3. Parallelizing Specific MCMC Algorithms

### 3.1 MALA (Metropolis-Adjusted Langevin Algorithm)

**Proposal**:

```
x̃_t = x_{t-1} + ε*∇_x log p(x_{t-1}) + √(2ε)*ω_t,  ω_t ~ N(0, I)
```

**Accept-reject** (non-differentiable):

```
α = min{1, p(x̃_t)*q(x_{t-1}|x̃_t) / (p(x_{t-1})*q(x̃_t|x_{t-1}))}
accept if u_t < α  where u_t ~ U(0,1)
```

**Key implementation detail**: The accept-reject step is non-differentiable. Use the **stop-gradient trick** for the Jacobian:

```julia
function mala_transition(x_prev, ξ, u, ε, ∇logp)
    # Proposal
    x̃ = x_prev + ε * ∇logp(x_prev) + sqrt(2ε) * ξ

    # Acceptance probability
    α = min(1, compute_acceptance_ratio(x_prev, x̃, ∇logp, ε))

    # Soft gating for gradient (stop-gradient trick)
    g̃ = log(α) - log(u)
    g = σ(g̃) + stop_gradient(float(g̃ > 0) - σ(g̃))

    # Accept-reject via gating
    x_t = g * x̃ + (1 - g) * x_prev

    return x_t
end
```

The forward pass computes the exact MALA update, but the backward pass (for Jacobian) uses the differentiable relaxation.

**Convergence guarantee**: Despite the approximate Jacobian, DEER/quasi-DEER still globally converge to the true sequential trace (Proposition 1 from Gonzalez et al. [8]).

### 3.2 HMC (Two Approaches)

#### Approach A: Parallelize Across HMC Steps (Sequential Leapfrog)

Treat each full HMC step (sample momentum → leapfrog → accept/reject) as one f_t:

```julia
function hmc_step(x_prev, ω_momentum, ω_leapfrog, u_accept, ε, L, ∇logp)
    # Sample momentum
    v = ω_momentum  # ~ N(0, I)

    # Leapfrog integration (sequential within this step)
    x, v = leapfrog(x_prev, v, ε, L, ∇logp)

    # Accept-reject (with stop-gradient trick as in MALA)
    # ...

    return x_new
end
```

  - Parallelize across T HMC samples
  - Each parallel iteration evaluates L leapfrog steps sequentially
  - Good when T >> L

#### Approach B: Parallelize Leapfrog Integration (Sequential HMC Steps)

Run HMC steps sequentially, but parallelize the inner leapfrog loop:

**Leapfrog step** (Equation 8):

```
x_t = x_{t-1} + ε*v_{t-1}
v_t = v_{t-1} + ε*∇_x log p(x_t)
```

State is `s_t = [x_t, v_t]`, apply DEER to the L leapfrog steps.

  - Good when L is large
  - Use **block quasi-DEER** for this (Section 2.4)
  - Convergence in ~L/2 iterations typically

### 3.3 Gibbs Sampling

For reparameterizable Gibbs samplers where each coordinate update is differentiable:

```
x_{t,d} = f_d(x_{t,1}, ..., x_{t,d-1}, x_{t-1,d+1}, ..., x_{t-1,D}, ξ_{t,d})
```

The full sweep `f = f_D ∘ ... ∘ f_2 ∘ f_1` is the transition function.

Note: The update for coordinate d does **not** depend on `x_{t-1,d}`, which affects the Jacobian structure.

* * *

## 4. Main Algorithm Pseudocode

```julia
function parallel_mcmc(f, s0, T; method=:quasi_deer, tol=1e-6, max_iters=1000)
    """
    f: transition function (s_{t-1}, ω_t) -> s_t
    s0: initial state  
    T: number of samples
    method: :deer, :quasi_deer, :stochastic_quasi_deer, :block_quasi_deer
    """

    D = length(s0)

    # Pre-sample all random inputs
    ω = sample_random_inputs(T)  # Shape: (T, ...)

    # Initialize state trajectory (e.g., all zeros, or repeat s0)
    s = zeros(T, D)
    s[1, :] = f(s0, ω[1])  # Or some initialization

    for iter in 1:max_iters
        # Step 1: Evaluate f and compute Jacobians (PARALLEL across t)
        f_vals = zeros(T, D)
        if method == :deer
            J = zeros(T, D, D)
        else
            J_diag = zeros(T, D)  # or block structure
        end

        @parallel for t in 1:T
            s_prev = (t == 1) ? s0 : s[t - 1, :]
            f_vals[t, :] = f(s_prev, ω[t])

            if method == :deer
                J[t, :, :] = jacobian(s -> f(s, ω[t]), s_prev)
            elseif method == :stochastic_quasi_deer
                J_diag[t, :] = estimate_jacobian_diagonal(s -> f(s, ω[t]), s_prev)
            else
                J_diag[t, :] = diag(jacobian(s -> f(s, ω[t]), s_prev))
            end
        end

        # Step 2: Compute inputs u_t = f_t(s_{t-1}^{(i)}) - J_t * s_{t-1}^{(i)}
        u = zeros(T, D)
        @parallel for t in 1:T
            s_prev = (t == 1) ? s0 : s[t - 1, :]
            if method == :deer
                u[t, :] = f_vals[t, :] - J[t, :, :] * s_prev
            else
                u[t, :] = f_vals[t, :] - J_diag[t, :] .* s_prev
            end
        end

        # Step 3: Solve linear system via parallel scan
        if method == :deer
            s_new = parallel_scan_full(J, u, s0)
        else
            s_new = parallel_scan_diagonal(J_diag, u, s0)
        end

        # Step 4: Check convergence
        if maximum(abs.(s_new - s)) < tol
            return s_new
        end

        s = s_new
    end

    @warn "Did not converge in $max_iters iterations"
    return s
end
```

* * *

## 5. Parallel Scan Implementation Details

### 5.1 For Full Jacobians (DEER)

The associative operator combines affine transforms:

```
(J₂, u₂) ⊕ (J₁, u₁) = (J₂ * J₁, J₂ * u₁ + u₂)
```

### 5.2 For Diagonal Jacobians (Quasi-DEER)

```
(d₂, u₂) ⊕ (d₁, u₁) = (d₂ .* d₁, d₂ .* u₁ + u₂)
```

This is element-wise and much cheaper.

### 5.3 For Block Quasi-DEER (Leapfrog)

With state `s = [x, v]` and block-diagonal Jacobian approximation, the scan can be done efficiently by tracking 2×2 blocks per dimension.

* * *

## 6. Practical Considerations

### 6.1 Preconditioning

For quasi-DEER, convergence depends on how well the diagonal approximates the true Jacobian. If the system has a known structure, apply an **orthogonal coordinate transformation**:

```
z_t = Q' * s_t
```

The transformed Jacobian is `Ĵ_t = Q' * J_t * Q`. If this is more diagonal, quasi-DEER converges faster.

**Especially useful for MALA**: The MALA Jacobian (ignoring accept-reject) is:

```
J_t = I + ε * ∇²_x log p(x_{t-1})
```

For symmetric Hessians, Q from the eigendecomposition diagonalizes this.

### 6.2 Sliding Window for High Dimensions

For high-dimensional problems where full-sequence parallel operations are memory-prohibitive:

 1. Initialize window at the start of the sequence
 2. Apply DEER update only within the window
 3. Shift window forward to the first non-converged timestep
 4. Repeat until the entire sequence converges

### 6.3 Early Stopping

Intermediate iterates before full convergence can produce **approximately valid** samples. This enables a time-quality tradeoff:

  - Run fewer Newton iterations for faster but approximate samples
  - The paper shows early-stopped parallel MALA can achieve comparable MMD to fully converged samples

### 6.4 Hyperparameter Considerations

**For MALA**:

  - Smaller step sizes ε → faster DEER convergence
  - Standard tuning (e.g., 80% acceptance rate) works well

**For HMC Leapfrog**:

  - Larger step sizes → more iterations to converge
  - Block quasi-DEER: ~2× fewer iterations than diagonal quasi-DEER
  - Parallelizing leapfrog is most beneficial when L (number of leapfrog steps) is large

### 6.5 Batch Size vs Chain Length Tradeoff

The paper reveals a new resource allocation question:

  - Traditional: Parallelize across independent chains (batch size B)
  - New: Also parallelize across chain length T

Findings suggest allocating **more resources to chain-length parallelization** is often more efficient than large batch sizes, because large batches can saturate GPU resources.

* * *

## 7. Julia-Specific Implementation Notes

### 7.1 Automatic Differentiation

For Jacobians and JVPs, use:

  - **ForwardDiff.jl** for full Jacobians (small D)
  - **Zygote.jl** or **Enzyme.jl** for JVPs in stochastic quasi-DEER

```julia
using ForwardDiff: jacobian

# Full Jacobian
J = jacobian(s -> f(s, ω), s_prev)

# JVP for stochastic estimation
using Zygote: pullback
_, back = pullback(s -> f(s, ω), s_prev)
Jᵀz = back(z)[1]  # This is J' * z
# For J * z, use forward-mode or dual numbers
```

### 7.2 GPU Parallelism

For the parallel evaluations of f and Jacobians:

  - Use `CUDA.jl` with `@cuda` kernels
  - Or `KernelAbstractions.jl` for portable GPU code
  - Ensure f is GPU-compatible (no scalar indexing, etc.)

For parallel scan:

  - Implement custom CUDA kernel for efficiency
  - Or use `Transducers.jl` with GPU support

### 7.3 Integration with AdvancedHMC.jl

Suggested integration points:

 1. **New sampler type**: `ParallelHMC` or `DEER_HMC`
 2. **Modify `Leapfrog` integrator**: Add parallel mode
 3. **New `sample` method**: Returns full chain in one call

```julia
# Possible API sketch
struct ParallelHMCSettings
    method::Symbol  # :deer, :quasi_deer, :stochastic_quasi_deer
    tol::Float64
    max_iters::Int
    window_size::Union{Int,Nothing}  # for sliding window
end

function sample(
    rng::AbstractRNG, model, sampler::ParallelHMC, n_samples::Int; initial_θ=nothing
)
    # Returns the full chain computed via parallel Newton iterations
end
```

* * *

## 8. Summary of Algorithm Choices

| Scenario                          | Recommended Method              |
|:--------------------------------- |:------------------------------- |
| Low-D, need fastest convergence   | DEER (full Jacobian)            |
| High-D, memory-constrained        | Stochastic quasi-DEER           |
| HMC with many leapfrog steps      | Block quasi-DEER on leapfrog    |
| Very long chains                  | Add sliding window              |
| Approximate samples OK            | Early stopping                  |
| MALA with known Hessian structure | Quasi-DEER with preconditioning |

* * *

## 9. References

  - **Paper**: Zoltowski et al., "Parallelizing MCMC Across the Sequence Length", NeurIPS 2025
  - **Code**: https://github.com/lindermanlab/parallel-mcmc
  - **DEER theory**: Lim et al. [7], Danieli et al. [6]
  - **Quasi-DEER**: Gonzalez et al. [8]
  - **Stochastic diagonal estimation**: Hutchinson [22], Bekas et al. [20]

* * *

## Appendix: Key Equations Reference

**Newton update (Eq. 2)**:
$$s_t^{(i+1)} = f_t(s_{t-1}^{(i)}) + J_t \left( s_{t-1}^{(i+1)} - s_{t-1}^{(i)} \right)$$

**Linear system form (Eq. 3)**:
$$s_t^{(i+1)} = J_t s_{t-1}^{(i+1)} + u_t$$
where $u_t = f_t(s_{t-1}^{(i)}) - J_t s_{t-1}^{(i)}$

**Quasi-DEER (Eq. 4)**:
$$s_t^{(i+1)} = \text{diag}(J_t) \odot s_{t-1}^{(i+1)} + u_t$$

**Leapfrog (Eq. 8)**:
$$x_t = x_{t-1} + \epsilon v_{t-1}; \quad v_t = v_{t-1} + \epsilon \nabla_x \log p(x_t)$$

**Stochastic diagonal (Eq. 10)**:
$$\text{diag}(J_t) = \mathbb{E}_{z \sim \text{Rad}}[z \odot (J_t z)]$$
