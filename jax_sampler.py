import argparse
from functools import partial
import time

import jax
import jax.numpy as np
from jax.scipy.stats import norm
from jax.scipy.special import logsumexp


@partial(jax.jit, static_argnums=(1,))
def rw_metropolis_kernel(rng_key, logpdf, position, log_prob):
    """Moves the chains by one step using the Random Walk Metropolis algorithm.

    Attributes
    ----------
    rng_key: jax.random.PRNGKey
        Key for the pseudo random number generator.
    logpdf: function
      Returns the log-probability of the model given a position.
    position: np.ndarray, shape (n_dims, n_chains)
      The starting position.
    log_prob: np.ndarray (, n_chains)
      The log probability at the starting position.

    Returns
    -------
    Tuple
        The next positions of the chains along with their log probability.
    """
    key1, key2 = jax.random.split(rng_key)
    move_proposals = jax.random.normal(key1, shape=position.shape) * 0.1
    proposal = position + move_proposals
    proposal_log_prob = logpdf(proposal)

    log_uniform = np.log(jax.random.uniform(key2, shape=position.shape))
    do_accept = log_uniform < proposal_log_prob - log_prob

    position = np.where(do_accept, proposal, position)
    log_prob = np.where(do_accept, proposal_log_prob, log_prob)
    return position, log_prob


def rw_metropolis_sampler(rng_key, logpdf, initial_position):
    """Generate samples using the Random Walk Metropolis Algorithm.
    Attributes
    ----------
    rng_key: jax.random.PRNGKey
        Key for the pseudo random number generator.
    logpdf: function
      Returns the log-probability of the model given a position.
    inital_position: np.ndarray (n_dims, n_chains)
      The starting position.

    Yields
    ------
    np.ndarray (,n_chains)
      The next sample generated by the random walk metropolis algorithm.
    """
    position = initial_position
    log_prob = logpdf(initial_position)
    yield position

    while True:
        rng_key, sub_key = jax.random.split(rng_key)
        position, log_prob = rw_metropolis_kernel(sub_key, logpdf, position, log_prob)
        yield position


def mixture_logpdf(x):
    """Log probability distribution function of a gaussian mixture model.

    Attribute
    ---------
    x: np.ndarray (4, n_chains)
        Position at which to evaluate the probability density function.

    Returns
    -------
    np.ndarray (, n_chains)
        The value of the log probability density function at x.
    """
    dist_1 = jax.partial(norm.logpdf, loc=-2.0, scale=1.2)
    dist_2 = jax.partial(norm.logpdf, loc=0, scale=1)
    dist_3 = jax.partial(norm.logpdf, loc=3.2, scale=5)
    dist_4 = jax.partial(norm.logpdf, loc=2.5, scale=2.8)
    log_probs = np.array([dist_1(x[0]), dist_2(x[1]), dist_3(x[2]), dist_4(x[3])])
    weights = np.repeat(np.array([[0.2, 0.3, 0.1, 0.4]]).T, x.shape[1], axis=1)
    return -logsumexp(np.log(weights) - log_probs, axis=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--samples", default=1000, required=False, type=int, help="Number of samples to take"
    )
    parser.add_argument("--chains", default=4, type=int, help="Number of chains to run")
    parser.add_argument(
        "--precompiled", type=bool, default=False, help="Whether to time with a precompiled model (faster)"
    )
    args = parser.parse_args()

    n_dim = 4
    n_samples = args.samples
    n_chains = args.chains
    rng_key = jax.random.PRNGKey(42)

    # If we want to measure the sampling speed excluding
    # the compilation, we first need to make sure that
    # Jax JITs the functions by running the algorithm once
    # and then time the subsequent loops.
    logpdf = jax.jit(mixture_logpdf)
    initial_position = np.zeros((n_dim, n_chains))
    samples = rw_metropolis_sampler(rng_key, logpdf, initial_position)
    for i, sample in enumerate(samples):
        if i == n_samples - 1:
            break
    sample.block_until_ready()

    if args.precompiled:
        times = []
        for _ in range(10):
            start = time.time()
            for i, sample in enumerate(samples):
                if i == n_samples - 1:
                    break
            sample.block_until_ready()
            stop = time.time()
            times.append(stop - start)
        print("JAX minus compile time --samples {} --chains {}: ".format(n_samples, n_chains), np.mean(np.array(times)))
