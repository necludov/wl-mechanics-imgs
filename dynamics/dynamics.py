import jax
import jax.random as random
import jax.numpy as jnp

from . import utils

@utils.register_dynamics(name='linear')
def generation(key, data, t):
  x_0 = random.normal(key, shape=data.shape)
  x_1 = data
  x_t = (1-t)*x_0 + t*x_1
  return x_0, x_1, x_t
