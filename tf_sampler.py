import argparse

import numpy as np
import tensorflow as tf


def gen_rw_metropolis_kernel(target_log_prob):
    def rw_metropolis_kernel(position, log_prob):
        move_proposal = tf.random.normal(
            position.shape, 0., .1, position.dtype)
        proposal = position + move_proposal
        proposal_log_prob = target_log_prob(proposal)
        log_uniform = tf.math.log(
            tf.random.uniform(
                shape=log_prob.shape,
                dtype=log_prob.dtype))
        is_accept = log_uniform < (proposal_log_prob - log_prob)
        is_accept_expanded = tf.reshape(
            is_accept,
            tf.pad(
                tf.shape(is_accept),
                [[0, tf.rank(position)-tf.rank(is_accept)]],
                constant_values=1))
        new_position = tf.where(is_accept_expanded, proposal, position)
        new_log_prob = tf.where(is_accept, proposal_log_prob, log_prob)
        return new_position, new_log_prob
    return rw_metropolis_kernel


def gen_mixture_log_prob(weights, loc, scale):
    def mixture_log_prob(x):
        log_unnormalized = -0.5 * tf.math.squared_difference(
            x / scale, loc / scale)
        log_normalization = tf.constant(
            0.5 * np.log(2. * np.pi), dtype=x.dtype) + tf.math.log(scale)
        log_probs = log_unnormalized - log_normalization
        return tf.math.reduce_logsumexp(
            tf.math.log(weights) + log_probs,
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
    gm_log_prob = gen_mixture_log_prob(weights, loc, scale)

    rw_metropolis_kernel = gen_rw_metropolis_kernel(gm_log_prob)

    position = tf.random.normal([n_chains, 4])
    log_prob = gm_log_prob(position)

    if use_scan:
        @tf.function(
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)],
            experimental_compile=True)
        def run_mcmc(nsample):
            final_state, final_log_prob = tf.scan(
                fn=lambda states, _: rw_metropolis_kernel(*states),
                elems=tf.ones(nsample),
                initializer=(
                    position,
                    log_prob
                )
            )
            return final_state, final_log_prob
    else:
        @tf.function(
            input_signature=[tf.TensorSpec(shape=None, dtype=tf.int32)],
            experimental_compile=True)
        def run_mcmc(nsample):
            i, final_state, final_log_prob = tf.while_loop(
                cond=lambda i, *kargs: i < nsample,
                body=lambda i, state, log_prob: (
                    i+1, *rw_metropolis_kernel(state, log_prob)),
                loop_vars=(
                    tf.constant(0),
                    position,
                    log_prob
                )
            )
            return final_state, final_log_prob

    if run_once:
        _ = run_mcmc(1)

    _ = run_mcmc(n_samples)
