import jax
import jax.random as random
import jax.numpy as jnp

from . import utils

@utils.register_dynamics(name='generation')
def generation(key, data, t):
  x_0 = random.normal(key, shape=data.shape)
  x_1 = data
  def add_noise(img):
    img_key = random.fold_in(random.PRNGKey(1), ((img*0.5 + 0.5)*255).astype(int).sum())
    return img + 5e-2*random.normal(img_key, img.shape)
  add_noise = jax.vmap(add_noise, in_axes=0)
  x_1 = add_noise(x_1)
  x_t = (1-t)*x_0 + t*x_1
  return x_0, x_1, x_t
