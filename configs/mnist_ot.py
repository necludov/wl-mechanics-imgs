import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 0

  # data
  config.data = data = ml_collections.ConfigDict()
  data.dataset = 'MNIST'
  data.ndims = 3
  data.image_size = 32
  data.num_channels = 1
  data.uniform_dequantization = True
  data.norm_mean = (0.5)
  data.norm_std = (0.5)
  data.random_flip = False
  data.task = 'OT'
  data.dynamics = 'linear'
  data.t_0, data.t_1 = 0.0, 1.0

  # models
  config.model_s = model_s = ml_collections.ConfigDict()
  model_s.num_channels = 1
  model_s.image_size = 32
  model_s.name = 'anet'
  model_s.loss = 'am'
  model_s.ema_rate = 0.99
  model_s.normalization = 'GroupNorm'
  model_s.nonlinearity = 'swish'
  model_s.nf = 32
  model_s.ch_mult = (1, 2, 2)
  model_s.num_res_blocks = 1
  model_s.attn_resolutions = (16,)
  model_s.resamp_with_conv = True
  model_s.dropout = 0.1

  config.model_q = model_q = ml_collections.ConfigDict()
  model_q.num_channels = 3
  model_q.image_size = 32
  model_q.name = 'unet'
  model_q.loss = 'q_ot'
  model_q.ema_rate = 0.99
  model_q.normalization = 'GroupNorm'
  model_q.nonlinearity = 'swish'
  model_q.nf = 32
  model_q.ch_mult = (1, 2, 2)
  model_q.num_res_blocks = 1
  model_q.attn_resolutions = (16,)
  model_q.resamp_with_conv = True
  model_q.dropout = 0.1

  # opts
  config.optimizer_s = optimizer_s = ml_collections.ConfigDict()
  optimizer_s.lr = 2e-4
  optimizer_s.beta1 = 0.9
  optimizer_s.eps = 1e-8
  optimizer_s.warmup = 5_000
  optimizer_s.grad_clip = 1.

  config.optimizer_q = optimizer_q = ml_collections.ConfigDict()
  optimizer_q.lr = 2e-4
  optimizer_q.beta1 = 0.9
  optimizer_q.eps = 1e-8
  optimizer_q.warmup = 5_000
  optimizer_q.grad_clip = 1.
  
  # training
  config.train = train = ml_collections.ConfigDict()
  train.batch_size = 128
  train.n_jitted_steps = 1
  train.n_iters = 100_000
  train.save_every = 5_000
  train.eval_every = 5_000
  train.log_every = 50

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.batch_size = 128
  eval.artifact_size = 64
  eval.num_samples = 500
  eval.use_ema = False
  eval.estimate_bpd = False

  return config
