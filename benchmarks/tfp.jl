# Based on https://rlouf.github.io/post/jax-random-walk-metropolis/

include(joinpath(@__DIR__, "argparse.jl"))

using PyCall

function main()
    parsed_args = parse_commandline()
    @assert parsed_args[:target] in ("mog", "gaussian") "Only mixture of Gaussians and Gaussian targets are implemented in TFP."
    
    py"""
    import os, time
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    from functools import partial
    import numpy as np
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfd = tfp.distributions

    dtype = np.float32

    def trace_everything(states, previous_kernel_results):
        return previous_kernel_results

    def hmc_sampler(n_dims, n_samples, n_chains, target):
        samples, _ = tfp.mcmc.sample_chain(
            num_results=n_samples,
            current_state=np.zeros((n_dims, n_chains), dtype=dtype),
            kernel=tfp.mcmc.HamiltonianMonteCarlo(
                target_log_prob_fn=target.log_prob,
                num_leapfrog_steps=4,
                step_size=0.01
            ),
            trace_fn=trace_everything
        )
        return samples

    # Define Gaussian mixtures
    target_dict = {}
    target_mog = tfd.Mixture(
        cat=tfd.Categorical(probs=[0.2, 0.3, 0.1, 0.4]),
        components=[
            tfd.Normal(loc=dtype(-2.0), scale=dtype(1.2)),
            tfd.Normal(loc=dtype(+0.0), scale=dtype(1.0)),
            tfd.Normal(loc=dtype(+3.2), scale=dtype(5.0)),
            tfd.Normal(loc=dtype(+2.5), scale=dtype(2.8)),
        ],
    )
    target_dict["mog"] = target_mog

    # Define Gaussian
    target_gauss = tfd.Normal(loc=dtype(-0.0), scale=dtype(1.0))
    target_dict["gaussian"] = target_gauss

    # Choose target
    target_name = $(parsed_args[:target])
    target = target_dict[target_name]

    # Define parameters
    n_dims = $(parsed_args[:n_dims])
    n_samples = $(parsed_args[:n_samples])
    n_chains = $(parsed_args[:n_chains])

    # Run sampling
    print(f"Running {n_dims} dimensional {target_name} for {n_samples} samples with {n_chains} chains")
    run_mcm = partial(hmc_sampler, n_dims, n_samples, n_chains, target)
    start = time.time()
    tf.xla.experimental.compile(run_mcm) if $(parsed_args[:use_xla]) else run_mcm()
    t = time.time() - start
    print(f"...Done with {t} seconds")
    """
end

main()
