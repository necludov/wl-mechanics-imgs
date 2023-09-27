import ml_collections


def get_config():
  config = ml_collections.ConfigDict()

  config.seed = 0
  config.loss = 'am'
  config.interpolant = 'linear'

  # data
  config.data = data = ml_collections.ConfigDict()
  data.source = 'normal'
  data.target = 'CIFAR10'
  data.ndims = 3
  data.image_size = 32
  data.num_channels = 3
  data.uniform_dequantization = True
  data.norm_mean = (0.5)
  data.norm_std = (0.5)
  data.random_flip = True
  data.task = 'OT'
  data.dynamics = 'linear'
  data.t_0, data.t_1 = 0.0, 1.0

  # models
  config.model_s = model_s = ml_collections.ConfigDict()
  model_s.num_channels = 3
  model_s.image_size = 32
  model_s.name = 'anet'
  model_s.ema_rate = 0.999
  model_s.normalization = 'GroupNorm'
  model_s.nonlinearity = 'swish'
  model_s.nf = 64
  model_s.ch_mult = (1, 2, 2, 2)
  model_s.num_res_blocks = 2
  model_s.attn_resolutions = (16,8)
  model_s.resamp_with_conv = True
  model_s.dropout = 0.1

  config.model_q = model_q = ml_collections.ConfigDict()
  model_q.num_channels = 3*3
  model_q.num_out_channels = 1*3
  model_q.image_size = 32
  model_q.name = 'unet'
  model_q.ema_rate = 0.999
  model_q.normalization = 'GroupNorm'
  model_q.nonlinearity = 'swish'
  model_q.nf = 64
  model_q.ch_mult = (1, 2, 2, 2)
  model_q.num_res_blocks = 2
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
  optimizer_s.decay = False

  config.optimizer_q = optimizer_q = ml_collections.ConfigDict()
  optimizer_q.lr = 2e-4
  optimizer_q.beta1 = 0.9
  optimizer_q.eps = 1e-8
  optimizer_q.warmup = 5_000
  optimizer_q.grad_clip = 1.
  optimizer_q.decay = False
  
  # training
  config.train = train = ml_collections.ConfigDict()
  train.batch_size = 128
  train.n_jitted_steps = 1
  train.n_grad_steps = 5
  train.n_iters = 300_000
  train.save_every = 20_000
  train.eval_every = 20_000
  train.log_every = 100

  # evaluation
  config.eval = eval = ml_collections.ConfigDict()
  eval.batch_size = 128
  eval.artifact_size = 64
  eval.num_samples = 500
  eval.use_ema = True
  eval.estimate_bpd = False

  return config
