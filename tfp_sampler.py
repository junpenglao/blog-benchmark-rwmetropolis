import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


def gen_mixture_log_prob_tfd(weights, loc, scale):
    def mixture_log_prob(x):
        return tf.math.reduce_logsumexp(
            tf.math.log(weights) + tfd.Normal(loc, scale).log_prob(x),
            axis=-1)
    return mixture_log_prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", default=1000, required=False, type=int, help="Number of samples to take"
    )
    parser.add_argument("--chains", default=4, type=int,
                        help="Number of chains to run")
    parser.add_argument("--runonce", default=True, type=bool,
                        help="Whether to run once to compile the function")
    parser.add_argument("--keep_all_samples", default=True, type=bool,
                        help="Keep all MCMC samples, otherwise keep the last sample only")
    args = parser.parse_args()

    n_dims = 4
    n_samples = args.samples
    n_chains = args.chains
    run_once = args.runonce
    use_scan = args.keep_all_samples

    loc = tf.constant([-2, 0, 3.2, 2.5])
    scale = tf.constant([1.2, 1, 5, 2.8])
    weights = tf.constant([0.2, 0.3, 0.1, 0.4])

    gm_log_prob_tfp = gen_mixture_log_prob(weights, loc, scale)
    rw_kernel_tfp = tfp.mcmc.RandomWalkMetropolis(
        gm_log_prob_tfp,
        new_state_fn=tfp.mcmc.random_walk_normal_fn(scale=0.1)
    )
    position = tf.random.normal([n_chains, 4])

    if use_scan:
        @tf.function(
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)],
            experimental_compile=True)
        def run_mcmc(nsample):
            state, log_probs = tfp.mcmc.sample_chain(
                num_results=nsample, 
                current_state=position,
                kernel=rw_kernel_tfp,
                trace_fn=lambda _, pkr: pkr.accepted_results.log_acceptance_correction)
            return state, log_probs
    else:
        @tf.function(
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)],
            experimental_compile=True)
        def run_mcmc(nsample):
            state, log_probs = tfp.mcmc.sample_chain(
                num_results=1, 
                num_burnin_steps=nsample-1,
                current_state=position,
                kernel=rw_kernel_tfp,
                trace_fn=lambda _, pkr: pkr.accepted_results.log_acceptance_correction)
            return state, log_probs

    if run_once:
        _ = run_mcmc(1)

    _ = run_mcmc(n_samples)
