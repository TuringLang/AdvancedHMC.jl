# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

AdvancedHMC.jl is a Julia implementation of advanced Hamiltonian Monte Carlo (HMC) algorithms. It serves as a backend for Turing.jl and can be used directly for flexible MCMC sampling.

## Common Commands

### Testing
```bash
# Run all core tests
julia --project -e 'using Pkg; Pkg.test()'

# Run specific test group
AHMC_TEST_GROUP=AdvancedHMC julia --project test/runtests.jl

# Run experimental/research tests
AHMC_TEST_GROUP=Experimental julia --project test/runtests.jl

# Run downstream Turing.jl integration tests
AHMC_TEST_GROUP=Downstream julia --project test/runtests.jl
```

Tests use ReTest.jl. Test files are in `test/` and cover: metric, hamiltonian, integrator, trajectory, adaptation, sampler, abstractmcmc, mcmcchains, constructors.

### Code Formatting
```bash
julia -e 'using JuliaFormatter; format(".")'
```
Uses Blue style (see `.JuliaFormatter.toml`).

## Architecture

The library uses a modular, composable design where sampling is built from independent components:

```
Sampler (NUTS/HMC/HMCDA)
    ↓
Hamiltonian (metric + kinetic energy + log density)
    ↓
Integrator (Leapfrog variants)
    ↓
Trajectory (termination criteria + sampling method)
    ↓
Adaptation (step size + mass matrix)
```

### Key Source Files

- `src/metric.jl` - Metric types: `UnitEuclideanMetric`, `DiagEuclideanMetric`, `DenseEuclideanMetric`
- `src/hamiltonian.jl` - `Hamiltonian`, `PhasePoint`, `DualValue` (caches value+gradient)
- `src/integrator.jl` - `Leapfrog`, `JitteredLeapfrog`, `TemperedLeapfrog`
- `src/trajectory.jl` - Termination criteria (`ClassicNoUTurn`, `GeneralisedNoUTurn`, `FixedNSteps`) and trajectory samplers (`EndPointTS`, `SliceTS`, `MultinomialTS`)
- `src/adaptation/` - Step size (`NesterovDualAveraging`), mass matrix (`WelfordVar`, `WelfordCov`, `NutpieVar`), composite (`StanHMCAdaptor`, `NaiveHMCAdaptor`)
- `src/constructors.jl` - Convenience constructors: `NUTS`, `HMC`, `HMCDA`
- `src/abstractmcmc.jl` - AbstractMCMC.jl interface implementation
- `src/riemannian/` - Riemannian manifold support with `GeneralizedLeapfrog`

### Extensions (ext/)

Optional features loaded when dependencies are available:
- `AdvancedHMCOrdinaryDiffEqExt` - DiffEqIntegrator support
- `AdvancedHMCMCMCChainsExt` - MCMCChains.jl integration
- `AdvancedHMCCUDAExt` - CUDA array support
- `AdvancedHMCComponentArraysExt` - ComponentArrays support
- `AdvancedHMCADTypesExt` - ADTypes.jl integration

## Key Interfaces

Works with:
- `LogDensityProblems.jl` - Define target distributions
- `LogDensityProblemsAD.jl` - Automatic differentiation
- `AbstractMCMC.jl` - MCMC sampling interface

## Code Conventions

Mathematical notation in comments:
- `ℓπ` - log density of target distribution
- `θ` - position/parameters
- `r` - momentum
- `z` - phase point (θ, r)
- `∂ℓπ∂θ` - gradient of log density
