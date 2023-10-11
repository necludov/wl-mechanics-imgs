from typing import Any
from functools import partial

import math

import jax
import jax.numpy as jnp
import flax
import diffrax

from models import utils as mutils


@flax.struct.dataclass
class EvalState:
  bpd_batch_id: int
  sample_batch_id: int
  key: Any


def get_bpd_estimator(model, config):

  def vector_field(t,data,args):
    state, eps = args
    x, log_p = data
    s = mutils.get_model_fn(model, state.params_ema, train=False)
    dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
    dsdx_val, jvp_val = jax.jvp(lambda _x: dsdx(t, _x), (x,), (eps,))
    return (dsdx_val, (jvp_val*eps).sum((1,2,3))) # mind that dt is negative in the solver

  solve = partial(diffrax.diffeqsolve, 
                  terms=diffrax.ODETerm(vector_field), 
                  solver=diffrax.Dopri5(), 
                  t0=1.0, t1=0.0, dt0=-1e-2, 
                  saveat=diffrax.SaveAt(ts=[0.0]),
                  stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5), 
                  adjoint=diffrax.NoAdjoint())

  def get_bpd(key, state, batch):
    x_1 = batch['image']
    key, eps_key = jax.random.split(key)
    eps = jax.random.randint(eps_key, x_1.shape, 0, 2).astype(float)*2 - 1.0
    solution = solve(y0=(x_1, jnp.zeros(x_1.shape[0])), args=(state, eps))
    x_0, delta_log_p = solution.ys[0][-1], solution.ys[1][-1]
    D = jnp.array(x_0.shape[1:]).prod()
    log_p_0 = -0.5*(x_0**2).sum((1,2,3)) - 0.5*D*math.log(2*math.pi)
    log_p_1 = log_p_0 + delta_log_p
    bpd = -log_p_1 / math.log(2) / D + 7.0
    return jax.lax.pmean(bpd.mean(), axis_name='batch'), solution.stats['num_steps']

  return get_bpd


def get_artifact_generator(model, config, dynamics, artifact_shape):
  if 'am' == config.model.loss:
    generator = get_ode_generator(model, config, dynamics, artifact_shape)
  elif 'sam' == config.model.loss:  
    generator = get_sde_generator(model, config, dynamics, artifact_shape)
  else:
    raise NotImplementedError(f'generator for f{config.model.loss} is not implemented')
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
                    solver=diffrax.Dopri5(), 
                    t0=0.0, t1=1.0, dt0=1e-3, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.PIDController(rtol=1e-5, atol=1e-5), 
                    adjoint=diffrax.NoAdjoint())
  
    solution = solve(y0=x_0, args=state)
    return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']
    
  return artifact_generator


def get_sde_generator(model, config, dynamics, artifact_shape):

  def artifact_generator(key, state, batch):
    key, dynamics_key = jax.random.split(key)
    x_0, _, _ = dynamics(dynamics_key, batch, t=jnp.zeros((1)))

    def vector_field(t,y,state):
      s = mutils.get_model_fn(model, 
                              state.params_ema if config.eval.use_ema else state.model_params, 
                              train=False)
      dsdx = jax.grad(lambda _t, _x: s(_t, _x).sum(), argnums=1)
      return dsdx(t,y)

    diffusion = lambda t, y, args: config.model.sigma * jnp.ones(x_0.shape)
    brownian_motion = diffrax.VirtualBrownianTree(t0=0.0, t1=1.0, tol=1e-3, shape=x_0.shape, key=key)
    terms = diffrax.MultiTerm(diffrax.ODETerm(vector_field), 
                              diffrax.WeaklyDiagonalControlTerm(diffusion, brownian_motion))
    solve = partial(diffrax.diffeqsolve, 
                    terms=terms, 
                    solver=diffrax.Euler(), 
                    t0=0.0, t1=1.0, dt0=1e-3, 
                    saveat=diffrax.SaveAt(ts=[1.0]),
                    stepsize_controller=diffrax.ConstantStepSize(True), 
                    adjoint=diffrax.NoAdjoint())

    solution = solve(y0=x_0, args=state)
    return solution.ys[-1][:,:,:,:artifact_shape[3]], solution.stats['num_steps']

  return artifact_generator
