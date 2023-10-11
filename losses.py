import math

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from dynamics import dynamics
import dynamics.utils as dutils


# def get_loss(config, model_s, model_q, time_sampler, train):
#   if 'am' == config.loss:
#     losses = get_am_loss(config, model_s, model_q, time_sampler, train)
#   elif 'rf' == config.loss:
#     if 'unet' == config.model_s.name:
#       losses = get_rf_vf_loss(config, model_s, model_q, time_sampler, train)
#     else:
#       losses = get_rf_loss(config, model_s, model_q, time_sampler, train)
#   else:
#     raise NotImplementedError(f'loss {config.loss} is not implemented')
#   return losses

def get_interpolant(config):

  def q_linear(model, params, train):
    nn = mutils.get_model_fn(model, params, train=train)
    def f(t, x_0, x_1, key):
      # x_t = (1-t)*x_0 + t*x_1
      key, u_key = random.split(key)
      input_x = jax.lax.concatenate([x_0, x_1, random.normal(u_key, shape=x_1.shape)], 3)
      intermediate = nn(t, input_x, key)
      out = (1-t)*x_0 + t*x_1 + (2*(1-t)*t)*nn(t, input_x, key)
      print(out.shape, 'out.shape', flush=True)
      return out
    return f

  def q_vp(model, params, train):
    nn = mutils.get_model_fn(model, params, train=train)
    def f(t, x_0, x_1, key):
      x_t = (1-t)*x_0 + t*x_1
      # key, u_key = random.split(key)
      input_x = jax.lax.concatenate([x_0, x_1], 3)
      out = x_t + jnp.sqrt(2*(1-t)*t)*nn(t, input_x, key)
      print(out.shape, 'out.shape', flush=True)
      return out
    return f

  def q_batch(model, params, train):
    nn = mutils.get_model_fn(model, params, train=train)
    def batch2channel(batch):
      bs = batch.shape[0]
      ids = (jnp.tile(jnp.arange(bs)[None,:], [bs,1]) + jnp.arange(bs)[:,None]) % bs
      return batch[ids].transpose(0,2,3,1,4).reshape(batch.shape[:3] + (batch.shape[0]*batch.shape[3],))
    def f(t, x_0, x_1, key):
      x_t = (1-t)*x_0 + t*x_1
      input_x = jax.lax.concatenate([batch2channel(x_0), batch2channel(x_1)], 3)
      return x_t + jnp.sqrt(2*(1-t)*t)*nn(t, input_x, key)
    return f

  def q_trig(model, params, train):
    nn = mutils.get_model_fn(model, params, train=train)
    def f(t, x_0, x_1, key):
      x_t = jnp.cos(0.5*jnp.pi*t)*x_0 + jnp.sin(0.5*jnp.pi*t)*x_1
      input_x = jax.lax.concatenate([x_0, x_1, x_t], 3)
      return x_t + jnp.cos(0.5*jnp.pi*t)*jnp.sin(0.5*jnp.pi*t)*nn(t, input_x, key)
    return f

  def q_sub(model, params, train):
    nn = mutils.get_model_fn(model, params, train=train)
    def f(t, x_0, x_1, key):
      x_t = (1-t)*x_0 + t*x_1
      input_x = jax.lax.concatenate([x_0, x_1, x_t], 3)
      output = nn(t, input_x, key)
      output += t*(x_1 - nn(jnp.ones_like(t), input_x, key))
      output += (1-t)*(x_0 - nn(jnp.zeros_like(t), input_x, key))
      return output
    return f

  def q_sub_trig(model, params, train):
    nn = mutils.get_model_fn(model, params, train=train)
    def f(t, x_0, x_1, key):
      x_t = jnp.cos(0.5*jnp.pi*t)*x_0 + jnp.sin(0.5*jnp.pi*t)*x_1
      input_x = jax.lax.concatenate([x_0, x_1, x_t], 3)
      output = nn(t, input_x, key)
      output += jnp.sin(0.5*jnp.pi*t)*(x_1 - nn(jnp.ones_like(t), input_x, key))
      output += jnp.cos(0.5*jnp.pi*t)*(x_0 - nn(jnp.zeros_like(t), input_x, key))
      return output
    return f

  if config.interpolant == 'linear':
    return q_linear
  elif config.interpolant == 'vp':
    return q_vp
  elif config.interpolant == 'batch':
    return q_batch
  elif config.interpolant == 'trig':
    return q_trig
  elif config.interpolant == 'sub':
    return q_sub
  elif config.interpolant == 'sub_trig':
    return q_sub_trig
  else:
    raise NotImplementedError(f'interpolant {config.interpolant} is not implemented')


def get_loss(config, model_s, model_q, time_sampler, train):

  interpolant = get_interpolant(config)

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    keys = random.split(key, num=7)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = interpolant(model_q, params_q, train=train)
    
    ################################################# loss s #################################################
    dsdtdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=[0,1])
    def potential(t, x, _key):
      dsdt, dsdx = dsdtdx_fn(t, x, _key)
      return dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    acceleration_fn = jax.grad(lambda t, x, _key: potential(t, x, _key).sum(), argnums=1)
    
    bs = batch[0].shape[0]
    # sample time
    t_0, t_1 = jnp.zeros((bs,1,1,1)), jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))

    # sample data
    x_0, x_1 = batch[0], batch[1]
    samples_q = q(t, x_0, x_1, keys[0])
    x_t = jax.lax.stop_gradient(samples_q)
    t_mult = 1e-1*(2*(1-t)*t)
    for i in range(config.train.n_grad_steps):
      dx = jax.lax.stop_gradient(acceleration_fn(t, x_t, jax.random.fold_in(keys[1], i)))
      x_t = x_t + t_mult*jnp.clip(dx, -1, 1)
    
    # boundaries loss
    s_0 = s(t_0, x_0, keys[2])
    s_1 = s(t_1, x_1, keys[3])
    loss_s = s_0.reshape((-1,1,1,1)) - s_1.reshape((-1,1,1,1))
    print(loss_s.shape, 'boundaries.shape', flush=True)

    # time loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[4])
    loss_s += dsdt
    metrics = {}
    metrics['cross_var'] = ((loss_s.squeeze()-loss_s.mean())**2).mean()
    loss_s += 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    print(loss_s.shape, 'final.shape', flush=True)
    metrics['loss_s'] = loss_s.mean()
    total_loss = loss_s.mean()
    
    ################################################# loss q #################################################
    
    s_detached = mutils.get_model_fn(model_s, jax.lax.stop_gradient(params_s), train=train)
    dsdtdx_fn_detached = jax.grad(lambda _t, _x, _key: s_detached(_t, _x, _key).sum(), argnums=[0,1])
    dsdt_detached, dsdx_detached = dsdtdx_fn_detached(t, samples_q, keys[5])
    loss_q = -(dsdt_detached + 0.5*(dsdx_detached**2).sum((1,2,3), keepdims=True))
    print(loss_q.shape, 'final.shape', flush=True)
    metrics['loss_q'] = loss_q.mean()
    total_loss += loss_q.mean()
    
    metrics['acceleration'] = jnp.sqrt((acceleration_fn(t, x_t, keys[6])**2).sum((1,2,3))).mean()
    potential = jax.lax.stop_gradient((dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)).squeeze())
    metrics['potential_var'] = ((potential.mean() - potential)**2).mean()
    return total_loss, (next_sampler_state, metrics)

  return loss_fn

################################################# RF losses #################################################

def get_rf_loss(config, model_s, model_q, time_sampler, train):

  interpolant = get_interpolant(config)

  def loss_s(key, params_s, params_q, sampler_state, batch):
    keys = random.split(key, num=8)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = interpolant(model_q, params_q, train=False)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    def potential(t, x, _key): 
      dsdt, dsdx = dsdtdx_fn(t, x, _key)
      return dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    acceleration_fn = jax.grad(lambda t, x, _key: potential(t, x, _key).sum(), argnums=1)
    
    bs = batch['image'].shape[0]
    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state, config.data.t_0, config.data.t_1)
    t = jnp.expand_dims(t, (1,2,3))

    # sample data
    x_1 = batch['image']
    x_0 = random.normal(keys[0], shape=x_1.shape)
    x_t = q(t, x_0, x_1, keys[1])

    # loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[2])
    loss = jax.grad(lambda _t: -(dsdx*q(_t, x_0, x_1, keys[1])).sum())(t)
    metrics = {}
    metrics['cross_var'] = ((loss.squeeze()-loss.mean())**2).mean()
    loss += 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    print(loss.shape, 'final.shape', flush=True)
    metrics['acceleration'] = jnp.sqrt((acceleration_fn(t, x_t, keys[5])**2).sum((1,2,3))).mean()
    potential = jax.lax.stop_gradient((dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)).squeeze())
    metrics['potential_var'] = ((potential.mean() - potential)**2).mean()
    return loss.mean(), (next_sampler_state, metrics)

  # def loss_q(key, params_q, params_s, sampler_state, batch):
  #   keys = random.split(key, num=8)
  #   s = mutils.get_model_fn(model_s, params_s, train=False)
  #   q = interpolant(model_q, params_q, train=train)
  #   dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    
  #   bs = batch['image'].shape[0]
  #   # sample time
  #   t_0, t_1 = jnp.zeros((bs,1,1,1)), jnp.ones((bs,1,1,1))
  #   t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
  #   t = jnp.expand_dims(t, (1,2,3))

  #   # sample data
  #   x_1 = batch['image']
  #   x_0 = random.normal(keys[0], shape=x_1.shape)
  #   x_t = q(t, x_0, x_1, keys[1])

  #   # loss
  #   dsdt, dsdx = dsdtdx_fn(t, x_t, keys[2])
  #   loss = jax.grad(lambda _t: (dsdx*q(_t, x_0, x_1, keys[1])).sum())(t)
  #   loss += -0.5*(dsdx**2).sum((1,2,3), keepdims=True)
  #   print(loss.shape, 'final.shape', flush=True)
  #   return loss.mean(), (next_sampler_state, {})

  def loss_q(key, params_q, params_s, sampler_state, batch):
    keys = random.split(key, num=8)
    s = mutils.get_model_fn(model_s, params_s, train=False)
    q = interpolant(model_q, params_q, train=train)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    
    bs = batch['image'].shape[0]
    # sample time
    t_0, t_1 = jnp.zeros((bs,1,1,1)), jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))

    # sample data
    x_1 = batch['image']
    x_0 = random.normal(keys[0], shape=x_1.shape)
    x_t = q(t, x_0, x_1, keys[1])

    # loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[2])
    loss = -(dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True))
    print(loss.shape, 'final.shape')
    return loss.mean(), (next_sampler_state, {})

  return loss_s, loss_q

def get_rf_vf_loss(config, model_s, model_q, time_sampler, train):

  interpolant = get_interpolant(config)

  def loss_s(key, params_s, params_q, sampler_state, batch):
    keys = random.split(key, num=8)
    v_fn = mutils.get_model_fn(model_s, params_s, train=train)
    q = interpolant(model_q, params_q, train=False)
    
    bs = batch['image'].shape[0]
    # sample time
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state, config.data.t_0, config.data.t_1)
    t = jnp.expand_dims(t, (1,2,3))

    # sample data
    x_1 = batch['image']
    x_0 = random.normal(keys[0], shape=x_1.shape)
    x_t = q(t, x_0, x_1, keys[1])

    # loss
    v = v_fn(t, x_t, keys[2])
    loss = jax.grad(lambda _t: -(v*q(_t, x_0, x_1, keys[3])).sum())(t)
    metrics = {}
    metrics['cross_var'] = ((loss.squeeze()-loss.mean())**2).mean()
    loss += 0.5*(v**2).sum((1,2,3), keepdims=True)
    print(loss.shape, 'final.shape', flush=True)

    dvdt = jax.jacfwd(lambda _t: v_fn(_t, x_t, keys[4]).sum(0))(t)
    dvdt = jnp.squeeze(dvdt, (4,5,6)).transpose((3,0,1,2))
    print(dvdt.shape, 'dvdt.shape', flush=True)
    dvdxv = jax.jvp(lambda _x: v_fn(t, _x, keys[4]), (x_t,), (v,))[0]
    print(dvdxv.shape, 'dvdxv.shape', flush=True)
    acceleration = dvdt + dvdxv

    metrics['acceleration'] = jnp.sqrt((acceleration**2).sum((1,2,3))).mean()
    metrics['potential_var'] = jnp.zeros_like(metrics['acceleration'])
    return loss.mean(), (next_sampler_state, metrics)

  def loss_q(key, params_q, params_s, sampler_state, batch):
    keys = random.split(key, num=8)
    v_fn = mutils.get_model_fn(model_s, params_s, train=train)
    q = interpolant(model_q, params_q, train=train)
    
    bs = batch['image'].shape[0]
    # sample time
    t_0, t_1 = jnp.zeros((bs,1,1,1)), jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))

    # sample data
    x_1 = batch['image']
    x_0 = random.normal(keys[0], shape=x_1.shape)
    x_t = q(t, x_0, x_1, keys[1])

    # loss
    v = v_fn(t, x_t, keys[2])
    loss = jax.grad(lambda _t: (v*q(_t, x_0, x_1, keys[1])).sum())(t)
    loss += -0.5*(v**2).sum((1,2,3), keepdims=True)
    print(loss.shape, 'final.shape', flush=True)
    return loss.mean(), (next_sampler_state, {})

  return loss_s, loss_q
