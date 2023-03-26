import math

import flax
import jax
import jax.numpy as jnp
import jax.random as random
from models import utils as mutils
from dynamics import dynamics
import dynamics.utils as dutils


def get_loss(config, model_s, model_q, dynamics, time_sampler, train):
  if 'am' == config.loss:
    loss_fn = get_am_loss(config, model_s, model_q, dynamics, time_sampler, train)
  elif 'sam' == config.loss:
    loss_fn = get_stoch_am_loss(config, model, q_t, time_sampler, train)
  elif 'q_ot' == config.loss:
    loss_fn = get_q_ot_loss(config, model_s, model_q, dynamics, time_sampler, train)
  elif 'q_sb' == config.loss:
    loss_fn = get_sb_loss(config, model, q_t, time_sampler, train)
  else:
    raise NotImplementedError(f'loss {config.model.loss} is not implemented')
  return loss_fn


def get_am_loss(config, model_s, model_q, dynamics, time_sampler, train):

  def loss_fn(key, params_s, params_q, sampler_state, batch):
    keys = random.split(key, num=8)
    s = mutils.get_model_fn(model_s, params_s, train=train)
    q = mutils.get_model_fn(model_q, params_q, train=False)
    dsdtdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=[0,1])
    dsdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=1)
    def potential(t, x, _key): 
      dsdt, dsdx = dsdtdx_fn(t, x, _key)
      return dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    acceleration_fn = jax.value_and_grad(lambda t, x, _key: potential(t, x, _key).sum(), argnums=1)
    
    bs = batch['image'].shape[0]
    # sample time
    t_0, t_1 = jnp.zeros((bs,1,1,1)), jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))

    # sample data
    x_1 = batch['image']
    x_0 = random.normal(keys[0], shape=x_1.shape)
    z = dsdx_fn(t, (1-t)*x_0 + t*x_1, keys[1])
    x_t = jax.lax.concatenate([x_0, x_1, z], 3)
    x_t = jax.lax.stop_gradient(q(t, x_t, keys[2]))
    _, acceleration = acceleration_fn(t, x_t, keys[3])
    x_t = jax.lax.stop_gradient(x_t + 1e-3*acceleration)

    # boundaries loss
    s_0 = s(t_0, x_0, keys[5])
    s_1 = s(t_1, x_1, keys[6])
    loss = (s_0.reshape((-1,1,1,1)) - s_1.reshape((-1,1,1,1))).mean()
    print(loss.shape, 'boundaries.shape')

    # time loss
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[7])
    loss += (dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)).mean()
    print(loss.shape, 'final.shape')
    return loss.mean(), (next_sampler_state, jnp.sqrt((acceleration**2).sum((1,2,3))).mean())

  return loss_fn


def get_q_ot_loss(config, model_s, model_q, dynamics, time_sampler, train):

  def loss_fn(key, params_q, params_s, sampler_state, batch):
    keys = random.split(key, num=5)
    s = mutils.get_model_fn(model_s, params_s, train=False)
    q = mutils.get_model_fn(model_q, params_q, train=train)
    dsdtdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=[0,1])
    dsdx_fn = jax.grad(lambda t,x,_key: s(t,x,_key).sum(), argnums=1)
    def potential(t, x, _key): 
      dsdt, dsdx = dsdtdx_fn(t, x, _key)
      return dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    acceleration_fn = jax.grad(lambda t, x, _key: potential(t, x, _key).sum(), argnums=1)

    bs = batch['image'].shape[0]

    # sample time
    t_0, t_1 = jnp.zeros((bs,1,1,1)), jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    # sample data
    x_1 = batch['image']
    x_0 = random.normal(keys[0], shape=x_1.shape)
    z = dsdx_fn(t, (1-t)*x_0 + t*x_1, keys[1])
    x_t = jax.lax.concatenate([x_0, x_1, z], 3)
    x_t = q(t, x_t, rng=keys[2])
    loss = -potential(t, x_t, keys[3])
    print(loss.shape, 'final.shape')
    return loss.mean(), (next_sampler_state, 0.0)

  return loss_fn


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


def get_ot_loss_(config, model, q_t, time_sampler, train):

  def schedule(step):
    return jnp.min(jnp.array([(step-10000)/20000, 1.0]))

  w_t_fn = lambda t: jnp.ones_like(1-t)
  dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def loss(key, params, step, sampler_state, batch):
    keys = random.split(key, num=10)
    # define functions
    s = mutils.get_model_fn(model, params, train=train)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    dsdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=1)
    def a(_t, _x, _key):
      dsdt, dsdx = dsdtdx_fn(_t, _x, _key)
      return dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True)
    dadx_fn = jax.grad(lambda _t,_x,_key: a(_t,_x,_key).sum(), argnums=1)
    
    # sample intermediate
    data = batch['image']
    bs = data.shape[0]
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    # num_timesteps = x_t.shape[0]
    # t_ids = (t.ravel()*num_timesteps).astype(int)
    # x_t = x_t[t_ids,jnp.arange(bs)]
    # t = (t_ids/num_timesteps).reshape((bs,1,1,1))
    t = t.reshape((bs,1,1,1))
    x_0, x_1, x_t = q_t(keys[0], batch, t)

    # mask = random.randint(keys[1], [bs,1,1,1], 0, 2).astype(float)
    # x_init = mask*x_1 + (1-mask)*x_0
    # t_init = mask
    # num_steps = 10
    # shoot_dt = -mask*(1-t)/num_steps + (1-mask)*t/num_steps
    # def ode_step(carry_state, key):
    #   x, t = carry_state
    #   next_x = jax.lax.stop_gradient(x + shoot_dt*dsdx_fn(t, x, key))
    #   next_t = t + shoot_dt
    #   return (next_x, next_t), None
    # shoot_x_t = jax.lax.scan(ode_step, (x_init, t_init), random.split(keys[2], num_steps))[0][0]
    # shoot_x_t = jax.lax.stop_gradient(shoot_x_t)
    # x_t = shoot_x_t

    # mask = jax.random.uniform(keys[8], (bs,1,1,1)) < schedule(step)
    # mask = mask.astype(float)
    # x_t = mask*shoot_x_t + (1-mask)*x_t
    
    # dt = config.model.w2_step_size
    # num_steps = config.model.w2_steps
    # def max_step(carry_state, key):
    #   x, t = carry_state
    #   next_x = jax.lax.stop_gradient(x + dt*dadx_fn(t, x, key))
    #   return (next_x, t), None
    # x_t = jax.lax.scan(max_step, (x_t, t), random.split(keys[3], num_steps))[0][0]
    # x_t = jax.lax.stop_gradient(x_t)
    # print(x_t.shape, 'x_t.shape')

    # eval loss
    s_0 = s(t_0, x_0, rng=keys[4])
    s_1 = s(t_1, x_1, rng=keys[5])
    loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')
    dsdt, dsdx = dsdtdx_fn(t, x_t, keys[6])
    loss += w_t_fn(t)*(dsdt + 0.5*(dsdx**2).sum((1,2,3), keepdims=True))
    print(loss.shape, 'final.shape')
    loss += dwdt_fn(t)*s(t, x_t, rng=keys[7]).reshape((-1,1,1,1))
    print(loss.shape, 'final.shape')
    return loss.mean(), (next_sampler_state,)

  return loss


def get_ot_loss(config, model, q_t, time_sampler, train):

  # w_t_fn = lambda t: (1-t)
  # dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def loss(key, params, step, sampler_state, batch):
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
    t_init = mask.astype(float)
    num_steps = config.model.grad_steps
    dt = -mask*(1-t)/num_steps + (1-mask)*t/num_steps

    def ode_step(carry_state, key):
      x, t = carry_state
      next_x = x + dt*dsdx_fn(t, x, key)
      next_t = t + dt
      return (next_x, next_t), None

    x_t, _ = jax.lax.scan(ode_step, (x_init, t_init), random.split(keys[2], num_steps))[0]
    s_0 = s(t_0, x_0, rng=keys[3])
    s_1 = s(t_1, x_1, rng=keys[4])
    # loss = w_t_fn(t_0)*s_0.reshape((-1,1,1,1)) - w_t_fn(t_1)*s_1.reshape((-1,1,1,1))
    loss = s_0.reshape((-1,1,1,1)) - s_1.reshape((-1,1,1,1))
    print(loss.shape, 'boundaries.shape')
    dsdt_min, dsdx_min = dsdtdx_fn(t, jax.lax.stop_gradient(x_t), keys[5])
    min_loss = dsdt_min + 0.5*(dsdx_min**2).sum((1,2,3), keepdims=True)
    # min_loss = w_t_fn(t)*(dsdt_min + 0.5*(dsdx_min**2).sum((1,2,3), keepdims=True))
    # min_loss += s(t, x_t, keys[6]).reshape((-1,1,1,1))*dwdt_fn(t)
    print(loss.shape, 'detached_x.shape')
    dsdt_max, dsdx_max = dsdtdx_fn_detached(t, x_t, keys[7])
    max_loss = -(dsdt_max + 0.5*(dsdx_max**2).sum((1,2,3), keepdims=True))
    # max_loss = -w_t_fn(t)*(dsdt_max + 0.5*(dsdx_max**2).sum((1,2,3), keepdims=True))
    # max_loss += -s_detached(t, x_t, keys[8]).reshape((-1,1,1,1))*dwdt_fn(t)
    print(loss.shape, 'detached_params.shape')
    loss += min_loss + max_loss
    print(loss.shape, 'final.shape')
    return loss.mean(), (next_sampler_state,)

  return loss


def get_ot_loss__(config, model, q_t, time_sampler, train):

  # w_t_fn = lambda t: (1-t)
  # dwdt_fn = jax.grad(lambda t: w_t_fn(t).sum(), argnums=0)

  def loss(key, params, step, sampler_state, batch):
    keys = random.split(key, num=9)
    s = mutils.get_model_fn(model, params, train=train)
    s_detached = mutils.get_model_fn(model, jax.lax.stop_gradient(params), train=train)
    dsdtdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=[0,1])
    dsdtdx_fn_detached = jax.grad(lambda _t,_x,_key: s_detached(_t,_x,_key).sum(), argnums=[0,1])
    dsdx_fn = jax.grad(lambda _t,_x,_key: s(_t,_x,_key).sum(), argnums=1)
    dsdx_fn_detached = jax.grad(lambda _t,_x,_key: s_detached(_t,_x,_key).sum(), argnums=1)
    
    data = batch['image']
    bs = data.shape[0]
    t_0, t_1 = config.data.t_0*jnp.ones((bs,1,1,1)), config.data.t_1*jnp.ones((bs,1,1,1))
    t, next_sampler_state = time_sampler.sample_t(bs, sampler_state)
    t = jnp.expand_dims(t, (1,2,3))
    x_0, x_1, _ = q_t(keys[0], data, t)

    dsdx = dsdx_fn(t_1, x_1, keys[1])
    q_0 = x_1 - dsdx
    x_t = x_1 - (1-t)*dsdx

    loss = s(t_0, x_0, rng=keys[3]).reshape((-1,1,1,1)) - s(t_0, jax.lax.stop_gradient(q_0), rng=keys[2]).reshape((-1,1,1,1))
    loss += -0.5*(dsdx_fn(t, jax.lax.stop_gradient(x_t), keys[4])**2).sum((1,2,3), keepdims=True)
    print(loss.shape, 'boundaries.shape')
    loss += s_detached(t_0, q_0, rng=keys[2]).reshape((-1,1,1,1))
    loss += 0.5*(dsdx_fn_detached(t, x_t, keys[4])**2).sum((1,2,3), keepdims=True)
    print(loss.shape, 'final.shape')
    return loss.mean(), (next_sampler_state,)

  return loss
