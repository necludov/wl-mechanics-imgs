import jax
import flax.linen as nn
import jax.numpy as jnp
import ml_collections
import functools

from . import utils, layers, normalization

RefineBlock = layers.RefineBlock
ResidualBlock = layers.ResidualBlock
ResnetBlockDDPM = layers.ResnetBlockDDPM
Upsample = layers.Upsample
Downsample = layers.Downsample
conv3x3 = layers.ddpm_conv3x3
get_act = layers.get_act
get_normalization = normalization.get_normalization
default_initializer = layers.default_init

@utils.register_model(name='mlp')
class MLP(nn.Module):
  config: ml_collections.ConfigDict

  @nn.compact
  def __call__(self, t: jnp.ndarray, x: jnp.ndarray, train: bool):
    config = self.config
    act = get_act(config)

    nf = config.nf

    temb = layers.get_timestep_embedding(t.ravel(), nf)
    temb = nn.Dense(nf)(temb)
    temb = nn.Dense(nf)(act(temb))

    h = nn.Dense(nf)(x.reshape(x.shape[0],config.image_size**2*config.num_channels)) + temb
    h = act(h)
    h = nn.Dense(nf)(h) + temb
    h = act(h)
    h = nn.Dense(nf)(h) + temb
    h = act(h)
    h = nn.Dense(nf)(h) + temb
    h = act(h)
    h = nn.Dense(config.image_size**2*config.num_channels)(h)
    h = h.reshape((-1, config.image_size, config.image_size, config.num_channels))
    return x + h*t*(1-t)
