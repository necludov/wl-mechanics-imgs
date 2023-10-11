from functools import partial
import math

import jax
import jax.numpy as jnp
import flax
import optax
import diffrax
import numpy as np

from models import utils as mutils

def get_optimizer(config):
  if config.decay:
    schedule = optax.warmup_exponential_decay_schedule(0.0, config.lr, config.warmup, 
      transition_steps=200_000, decay_rate=0.5, transition_begin=50_000)
  else:
    schedule = optax.join_schedules([optax.linear_schedule(0.0, config.lr, config.warmup), 
                                     optax.constant_schedule(config.lr)], 
                                     boundaries=[config.warmup])
  optimizer = optax.adamw(learning_rate=schedule, b1=config.beta1, eps=config.eps)
  optimizer = optax.chain(
    optax.clip(config.grad_clip),
    optimizer
  )
  return optimizer


def get_step_fn(config, optimizer_s, optimizer_q, loss_fn):

  def step_fn(carry_state, batch):
    (key, state_s, state_q) = carry_state
    key, step_key = jax.random.split(key)
    grad_fn = jax.value_and_grad(loss_fn, argnums=[1,2], has_aux=True)
    (loss, (new_sampler_state, metrics)), grads = grad_fn(step_key, 
      state_s.model_params,
      state_q.model_params,
      state_s.sampler_state, 
      batch)
    
    def update(optimizer, grad, state):
      updates, opt_state = optimizer.update(grad, state.opt_state, state.model_params)
      new_params = optax.apply_updates(state.model_params, updates)
      new_params_ema = jax.tree_map(
        lambda p_ema, p: p_ema * state.ema_rate + p * (1. - state.ema_rate),
        state.params_ema, new_params
      )
      new_state = state.replace(
        step=state.step+1,
        opt_state=opt_state,
        sampler_state=new_sampler_state, 
        model_params=new_params,
        params_ema=new_params_ema
      )
      return new_state
      
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    metrics = jax.tree_map(lambda _metric: jax.lax.pmean(_metric, axis_name='batch'), metrics)
    
    new_state_s = update(optimizer_s, grads[0], state_s)
    new_state_q = update(optimizer_q, grads[1], state_q)
    new_carry_state = (key, new_state_s, new_state_q)
    return new_carry_state, (loss, metrics)

  return step_fn

def get_artifact_generator(model, config, artifact_shape):
  if 'am' == config.loss:
    generator = get_ot_generator(model, config, artifact_shape)
  elif 'rf' == config.loss:  
    generator = get_ot_generator(model, config, artifact_shape)
  else:
    raise NotImplementedError(f'generator for {config.model.loss} is not implemented')
  return generator


def get_ode_generator(model, config, dynamics, artifact_shape):

  def artifact_generator(key, state, batch):
    x_0, _, _ = dynamics(key, batch, t=jnp.zeros((1)))
    
    def vector_field(t,y,state):
      s = mutils.get_model_fn(model, 
                              state.params_ema if config.eval.use_ema else state.model_params, 
                              train=False)
      dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
      return dsdx(t,y)
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Euler(), 
                    t0=0.0, t1=1.0, dt0=1e-2, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())
  
    solution = solve(y0=x_0, args=state)
    return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']
    
  return artifact_generator


def get_sde_generator(model, config, dynamics, artifact_shape):

  def artifact_generator(key, state, batch):
    x_0, _, _ = dynamics(key, batch, t=jnp.zeros((1)))

    def vector_field(t,y,state):
      s = mutils.get_model_fn(model, 
                              state.params_ema if config.eval.use_ema else state.model_params, 
                              train=False)
      dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
      return dsdx(t,y)
    
    diffusion = lambda t, y, args: config.model.sigma * jnp.ones(x_0.shape)
    brownian_motion = diffrax.UnsafeBrownianPath(shape=x_0.shape, key=key)
    terms = diffrax.MultiTerm(diffrax.ODETerm(vector_field), 
                              diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion))
    solve = partial(diffrax.diffeqsolve, 
                    terms=terms, 
                    solver=diffrax.Euler(), 
                    t0=0.0, t1=1.0, dt0=1e-2, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())

    solution = solve(y0=x_0, args=state)
    return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']

  return artifact_generator


def get_ot_generator(model, config, artifact_shape):

  def artifact_generator(key, state, x_0):
    x_0 = x_0[:x_0.shape[0]//2]
    s = mutils.get_model_fn(model, 
                            state.params_ema if config.eval.use_ema else state.model_params, 
                            train=False)
    if 'unet' == config.model_s.name:
      dsdx = s
    else:
      dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
    vector_field = lambda _t,_x,_args: dsdx(_t,_x)
    solve = partial(diffrax.diffeqsolve, 
                    terms=diffrax.ODETerm(vector_field), 
                    solver=diffrax.Euler(), 
                    t0=0.0, t1=1.0, dt0=1e-2, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())
    solution = solve(y0=x_0, args=state)
    one_step_artifacts = x_0 + dsdx(jnp.zeros((x_0.shape[0], 1, 1, 1)), x_0)
    artifacts = jnp.stack([solution.ys[-1][:,:,:,:artifact_shape[3]], one_step_artifacts], 0)
    return artifacts, solution.stats['num_steps']
    
  return artifact_generator


def stack_imgs(x, n=8, m=8):
    im_size = x.shape[2]
    big_img = np.zeros((n*im_size,m*im_size,x.shape[-1]),dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            p = x[i*m+j] * 255
            p = p.clip(0, 255).astype(np.uint8)
            big_img[i*im_size:(i+1)*im_size, j*im_size:(j+1)*im_size, :] = p
    return big_img
