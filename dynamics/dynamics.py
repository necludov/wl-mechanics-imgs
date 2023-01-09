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
    return img + 1e-1*random.normal(img_key, img.shape)
  add_noise = jax.vmap(add_noise, in_axes=0)
  x_1 = add_noise(x_1)
  x_t = (1-t)*x_0 + t*x_1
  return x_0, x_1, x_t

@utils.register_dynamics(name='superres')
def superres(key, data, t):
  x_0 = random.normal(key, shape=data.shape)
  x_1 = data
  downscaled_shape = (data.shape[0], data.shape[1]//2, data.shape[2]//2, data.shape[3])
  downscaled_x = jax.image.resize(x_1, downscaled_shape, method='nearest')
  downscaled_x = jax.image.resize(downscaled_x, x_1.shape, method='nearest')
  downscaled_x = downscaled_x.at[:,1::2,:,:].set(x_0[:,1::2,:,:])
  downscaled_x = downscaled_x.at[:,:,1::2,:].set(x_0[:,:,1::2,:])
  x_t = (1-t)*downscaled_x + t*x_1
  return downscaled_x, x_1, x_t

@utils.register_dynamics(name='color')
def color(key, data, t):
  def add_noise(img):
    img_key = random.fold_in(random.PRNGKey(1), ((img*0.5 + 0.5)*255).astype(int).sum())
    return img + 3e-2*random.normal(img_key, img.shape)
  add_noise = jax.vmap(add_noise, in_axes=0)
  x_1 = data
  grayscale = data.mean(-1, keepdims=True)
  x_0 = jnp.tile(grayscale, (1,1,1,3))
  x_0 = add_noise(x_0)
  x_0 = jax.lax.concatenate([x_0, grayscale], 3)
  x_1 = jax.lax.concatenate([x_1, grayscale], 3)
  x_t = (1-t)*x_0 + t*x_1
  return x_0, x_1, x_t

@utils.register_dynamics(name='inpaint')
def inpaint(key, data, t):
  key, x_0_key = random.split(key)
  x_0 = random.normal(x_0_key, shape=data.shape)
  x_1 = data
  key, mask_key = random.split(key)
  H = data.shape[2]
  u = random.randint(mask_key, (x_0.shape[0],1,1,1), 0, 2)
  mask = jnp.zeros_like(x_0)
  mask = mask.at[:,:H//2,:,:].set(u)
  mask = mask.at[:,H//2:,:,:].set(1-u)
  x_t = mask*x_1 + (1-mask)*((1-t)*x_0 + t*x_1)
  return x_0, x_1, x_t

