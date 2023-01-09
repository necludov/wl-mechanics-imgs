import math
import jax
import flax
import numpy as np
from jax import random
from jax import numpy as jnp
# from scipy import integrate
from functools import partial

import diffrax
from diffrax import diffeqsolve, Euler, ODETerm, SaveAt, ConstantStepSize


from models import utils as mutils


# def euler_scheme(ode_func, t0, t1, x, dt):
#   timesteps = np.arange(t0, t1, dt)
#   solution = integrate._ivp.ivp.OdeResult(y=np.zeros([len(x), len(timesteps)+1]), 
#                                           t=np.zeros(len(timesteps)+1), 
#                                           nfev=0)
#   solution.y[:,0] = x
#   for i, t in enumerate(timesteps):
#     dx = ode_func(t, solution.y[:,i])
#     solution.t[i] = t
#     solution.y[:,i+1] = solution.y[:,i] + dt*dx
#     solution.nfev += 1
#   solution.t[-1] = t1
#   return solution


# def get_ode_solver(model, shape, t0=0.0, t1=1.0, atol=1e-5, rtol=1e-5, method='RK45', dt=1e-2):

#   def s_sum(state,t,x,rng):
#     s = mutils.get_model_fn(model, state.model_params, train=False)
#     return s(t,x,rng).sum()
#   dsdx = jax.pmap(jax.grad(s_sum, argnums=2), axis_name='batch')
#   pshape = (jax.local_device_count(), shape[0]//jax.local_device_count()) + shape[1:]
  
#   def ode_solver(prng, pstate, init_x):
#     rng = flax.jax_utils.unreplicate(prng)
#     rng, *next_rng = random.split(rng, num=jax.local_device_count() + 1)
#     next_rng = jnp.asarray(next_rng)
#     x = mutils.to_flattened_numpy(init_x)

#     def ode_func(t, x):
#       x = mutils.from_flattened_numpy(x, pshape)
#       vec_t = jnp.ones((x.shape[0], x.shape[1])) * t
#       grad = dsdx(pstate, vec_t, x, next_rng)
#       return mutils.to_flattened_numpy(grad)

#     if 'euler' != method:
#       solution = integrate.solve_ivp(ode_func, (t0, t1), x, rtol=rtol, atol=atol, method=method)
#     else:
#       solution = euler_scheme(ode_func, t0, t1, x, dt)

#     nfe = solution.nfev
#     x = jnp.asarray(solution.y[:, -1]).reshape(shape)
#     return x, nfe
#   return ode_solver
