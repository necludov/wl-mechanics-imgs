import math

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from dynamics import dynamics
import dynamics.utils as dutils


def get_loss(config, model, q_t, time_sampler, train):
  if 'am' == config.model.loss:
    loss_fn = get_am_loss(config, model, q_t, time_sampler, train)
  elif 'sam' == config.model.loss:
    loss_fn = get_stoch_am_loss(config, model, q_t, time_sampler, train)
  elif 'amot' == config.model.loss:
    loss_fn = get_amot_loss(config, model, q_t, time_sampler, train)
  else:
    raise NotImplementedError(f'loss {config.model.loss} is not implemented')
  return loss_fn


def get_am_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def am_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=5)
    s = mutils.get_model_fn(model, params, train=train)
    dsdtdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=[0,1])
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    x_0, x_1, x_t = q_t(keys[0], data, t)

    # boundaries loss
    s_0 = s(t_0, x_0, rng=keys[1])
    s_1 = s(t_1, x_1, rng=keys[2])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')

    # time loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[3])
    p_t = time_sampler.invdensity(t)
    s_t = s(t, x_t, keys[4])
    print(p_t.shape, dsdt.shape, dsdx.shape, 'p_t.shape, dsdt.shape, dsdx.shape')
    loss += w_t_fn(t)*p_t*(dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True))
    loss += s_t.reshape((-1,1,1,1))*dwdt_fn(t)*p_t
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return am_loss


def get_stoch_am_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  sigma = lambda t: config.model.sigma*jnp.ones_like(t)

  def sam_loss(key, params, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model, params, train=train)
    dsdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=1)
    dsdt_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=0)
    
    data = batch['image']
    bs = data.shape[0]

    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    x_0, x_1, x_t = q_t(keys[0], data, t)

    # boundaries loss
    s_0 = s(t_0, x_0, rng=keys[1])
    s_1 = s(t_1, x_1, rng=keys[2])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')

    # time loss
    eps = random.randint(keys[3], x_t.shape, 0, 2).astype(float)*2 - 1.0
    dsdx_val, jvp_val = jax.jvp(lambda _x: dsdx_fn(t, _x, keys[4]), (x_t,), (eps,))
    dsdt_val = dsdt_fn(t, x_t, keys[5])
    s_t = s(t, x_t, keys[6])
    p_t = time_sampler.invdensity(t)
    print(p_t.shape, dsdt_val.shape, dsdx_val.shape, 'p_t.shape, dsdt.shape, dsdx.shape')
    time_loss = dsdt_val + 0.5*(dsdx_val**2).sum((1,2,3), keepdims=True)
    time_loss += 0.5*sigma(t)**2*(jvp_val*eps).sum((1,2,3), keepdims=True)
    time_loss *= w_t_fn(t)
    time_loss += s_t.reshape((-1,1,1,1))*dwdt_fn(t)
    loss += p_t*time_loss
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return sam_loss


def get_amot_loss(config, model, q_t, time_sampler, train):

  w_t_fn = lambda t: (1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def loss(key, params, sampler_state, batch):
    keys = random.split(key, num=9)
    s = mutils.get_model_fn(model, params, train=train)
    s_detached = mutils.get_model_fn(model, jax.lax.stop_gradient(params), train=train)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    dsdtdx_fn_detached = jax.grad(lambda _t,_x,_key: s_detached(_t,_x,_key).sum(), argnums=[0,1])
    dsdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=1)
    
    data = batch['image']
    bs = data.shape[0]
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    x_0, x_1, _ = q_t(keys[0], data, t)

    mask = random.randint(keys[1], [bs,1,1,1], 0, 2)
    x_init = mask*x_1 + (1-mask)*x_0
    x_t = x_init + (-mask*(1-t) + (1-mask)*t)*dsdx_fn(mask.astype(float), x_init, keys[2])
    s_0 = s(t_0, x_0, rng=keys[3])
    s_1 = s(t_1, x_1, rng=keys[4])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')
    dsdt_min, dsdx_min = dsdtdx_fn(t, jax.lax.stop_gradient(x_t), keys[5])
    min_loss = w_t_fn(t)*(dsdt_min + 0.5*(dsdx_min**2).sum((1,2,3), keepdims=True))
    min_loss += s(t, x_t, keys[6]).reshape((-1,1,1,1))*dwdt_fn(t)
    print(loss.shape, 'detached_x.shape')
    dsdt_max, dsdx_max = dsdtdx_fn_detached(t, x_t, keys[7])
    max_loss = -w_t_fn(t)*(dsdt_max + 0.5*(dsdx_max**2).sum((1,2,3), keepdims=True))
    max_loss += -s_detached(t, x_t, keys[8]).reshape((-1,1,1,1))*dwdt_fn(t)
    print(loss.shape, 'detached_params.shape')
    loss += min_loss + max_loss
    print(loss.shape, 'final.shape')
    return loss.mean(), next_sampler_state

  return loss
