# Based on https://rlouf.github.io/post/jax-random-walk-metropolis/

using ArgParse, PyCall

function parse_commandline()
    s = ArgParseSettings()

    @add_arg_table s begin
        "--n_dims"
            arg_type = Int
            required = true
        "--n_samples"
            arg_type = Int
            required = true
        "--n_chains"
            arg_type = Int
            required = true
        "--use_xla"
            arg_type = Bool
            required = true
        "--target"
            arg_type = String
            required = true
            range_tester = (x -> x in ("mog", "gauss"))
    end

    return parse_args(s)
end

function main()
    parsed_args = parse_commandline()
    n_dims = parsed_args["n_dims"]
    n_samples = parsed_args["n_samples"]
    n_chains = parsed_args["n_chains"]
    use_xla = parsed_args["use_xla"]
    target = parsed_args["target"]
    
    py"""
    import os
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
    target_dict["gauss"] = target_gauss

    target = target_dict[$target]

    import time

    n_dims = $n_dims
    n_samples = $n_samples
    n_chains = $n_chains

    ts = []

    for c in [1, 10, 100, 1000, 10000]:
        print(f"Running {c} chains")
        run_mcm = partial(hmc_sampler, n_dims, n_samples, c, target)
        start = time.time()
        tf.xla.experimental.compile(run_mcm) if $use_xla else run_mcm()
        t = time.time() - start
        print(f"...Done with {t} seconds")
        ts.append(t)

    print(ts)
    """
end

main()
