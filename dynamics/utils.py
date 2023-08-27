import math
from typing import Any

import flax
import optax
import functools
import jax.numpy as jnp
import jax
import numpy as np

@flax.struct.dataclass
class SamplerState:
  u0: jnp.ndarray

@flax.struct.dataclass
class TimeSampler:
  sample_t: Any

_DYNAMICS = {}


def register_dynamics(name):
  """A decorator for registering dynamics functions."""

  def _register(f):
    if name in _DYNAMICS:
      raise ValueError(f'Already registered dynamics with name: {name}')
    _DYNAMICS[name] = f
    return f

  return _register


def get_dynamics(name):
  return _DYNAMICS[name]

def get_time_sampler(config):

  def sample_uniformly(bs, state, t_0=0.0, t_1=1.0):
    u = (state.u0 + math.sqrt(2)*jnp.arange(bs*jax.device_count())) % 1
    new_state = state.replace(u0=u[-1:])
    t = (t_1-t_0)*u[jax.process_index()*bs:(jax.process_index()+1)*bs] + t_0
    return t, new_state

  sampler = TimeSampler(sample_t=sample_uniformly)
  init_state = SamplerState(u0=jnp.array([0.5]))

  return sampler, init_state
