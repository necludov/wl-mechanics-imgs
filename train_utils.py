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
  schedule = optax.join_schedules([optax.linear_schedule(0.0, config.train.lr, config.train.warmup), 
                                   optax.constant_schedule(config.train.lr)], 
                                   boundaries=[config.train.warmup])
  optimizer = optax.adam(learning_rate=schedule, b1=config.train.beta1, eps=config.train.eps)
  optimizer = optax.chain(
    optax.clip(config.train.grad_clip),
    optimizer
  )
  return optimizer


def get_step_fn(config, optimizer, loss_fn):

  def step_fn(carry_state, batch):
    (rng, state) = carry_state
    rng, step_rng = jax.random.split(rng)
    grad_fn = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)
    alpha = jnp.min(jnp.array([state.step/config.train.pretrain_steps, 1.0]))
    (loss, new_sampler_state), grad = grad_fn(step_rng, state.model_params, state.sampler_state, batch, alpha=alpha)
    grad = jax.lax.pmean(grad, axis_name='batch')
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

    loss = jax.lax.pmean(loss, axis_name='batch')
    new_carry_state = (rng, new_state)
    return new_carry_state, loss

  return step_fn


def get_artifact_generator(model, config, dynamics, artifact_shape):
  if 'am' == config.model.loss:
    generator = get_ode_generator(model, config, dynamics, artifact_shape)
  elif 'amot' == config.model.loss:  
    generator = get_ode_generator(model, config, dynamics, artifact_shape)
  elif 'sam' == config.model.loss:  
    generator = get_sde_generator(model, config, dynamics, artifact_shape)
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


def stack_imgs(x, n=8, m=8):
    im_size = x.shape[2]
    big_img = np.zeros((n*im_size,m*im_size,x.shape[-1]),dtype=np.uint8)
    for i in range(n):
        for j in range(m):
            p = x[i*m+j] * 255
            p = p.clip(0, 255).astype(np.uint8)
            big_img[i*im_size:(i+1)*im_size, j*im_size:(j+1)*im_size, :] = p
    return big_img
