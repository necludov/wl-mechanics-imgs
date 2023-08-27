# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
"""Return training and evaluation/test datasets from config files."""
import jax

from jax import numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def get_batch_iterator(config, evaluation=False):
  batch_size = config.eval.batch_size if evaluation else config.train.batch_size

  train_ds, _, _ = get_dataset(config.data.target, 
    batch_size, 
    config.data.image_size, 
    config.data.random_flip,
    evaluation,
    additional_dim=config.train.n_jitted_steps,
    uniform_dequantization=config.data.uniform_dequantization)
  train_iter_target = iter(train_ds)

  scaler = get_image_scaler(config)
  inverse_scaler = get_image_inverse_scaler(config)

  if config.data.source == 'normal':
    def batch_iterator(key):
      batch_target = scaler(next(train_iter_target)['image']._numpy())
      batch_source = jax.random.normal(key, shape=batch_target.shape)
      return (batch_source, batch_target)
  else:
    train_ds, _, _ = get_dataset(config.data.source, 
      batch_size, 
      config.data.image_size, 
      config.data.random_flip,
      evaluation,
      additional_dim=config.train.n_jitted_steps,
      uniform_dequantization=config.data.uniform_dequantization)
    train_iter_source = iter(train_ds)

    def batch_iterator(key):
      batch_target = scaler(next(train_iter_target)['image']._numpy())
      batch_source = scaler(next(train_iter_source)['image']._numpy())
      return (batch_source, batch_target)

  return batch_iterator, scaler, inverse_scaler

def get_image_scaler(config):
  def scaler(x):
    return (x - 0.5)/0.5
  return scaler


def get_image_inverse_scaler(config):
  def inv_scaler(x):
    return x*0.5 + 0.5
  return inv_scaler


def crop_resize(image, resolution):
  """Crop and resize an image to the given resolution."""
  crop = tf.minimum(tf.shape(image)[0], tf.shape(image)[1])
  h, w = tf.shape(image)[0], tf.shape(image)[1]
  image = image[(h - crop) // 2:(h + crop) // 2,
          (w - crop) // 2:(w + crop) // 2]
  image = tf.image.resize(
    image,
    size=(resolution, resolution),
    antialias=True,
    method=tf.image.ResizeMethod.BICUBIC)
  return tf.cast(image, tf.uint8)


def resize_small(image, resolution):
  """Shrink an image to the given resolution."""
  h, w = image.shape[0], image.shape[1]
  ratio = resolution / min(h, w)
  h = tf.round(h * ratio, tf.int32)
  w = tf.round(w * ratio, tf.int32)
  return tf.image.resize(image, [h, w], antialias=True)


def central_crop(image, size):
  """Crop the center of an image to the given size."""
  top = (image.shape[0] - size) // 2
  left = (image.shape[1] - size) // 2
  return tf.image.crop_to_bounding_box(image, top, left, size, size)


def get_dataset(dataset, batch_size, image_size, random_flip, evaluation, additional_dim=None, uniform_dequantization=False):
  """Create data loaders for training and evaluation.

  Args:
    additional_dim: An integer or `None`. If present, add one additional dimension to the output data,
      which equals the number of steps jitted together.
    uniform_dequantization: If `True`, add uniform dequantization to images.
    evaluation: If `True`, fix number of epochs to 1.

  Returns:
    train_ds, eval_ds, dataset_builder.
  """
  # Compute batch size for this worker.
  if batch_size % jax.device_count() != 0:
    raise ValueError(f'Batch sizes ({batch_size} must be divided by'
                     f'the number of devices ({jax.device_count()})')

  per_device_batch_size = batch_size // jax.device_count()
  # Reduce this when image resolution is too large and data pointer is stored
  shuffle_buffer_size = 10000
  prefetch_size = tf.data.experimental.AUTOTUNE
  num_epochs = None if not evaluation else 1
  # Create additional data dimension when jitting multiple steps together
  if additional_dim is None:
    batch_dims = [jax.local_device_count(), per_device_batch_size]
  else:
    batch_dims = [jax.local_device_count(), additional_dim, per_device_batch_size]

  # Create dataset builders for each dataset.
  if dataset == 'MNIST':
    dataset_builder = tfds.builder('mnist')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [image_size, image_size], antialias=True)

  elif dataset == 'EMNIST':
    dataset_builder = tfds.builder('emnist/letters')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [image_size, image_size], antialias=True)

  elif dataset == 'CIFAR10':
    dataset_builder = tfds.builder('cifar10')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [image_size, image_size], antialias=True)

  elif dataset == 'SVHN':
    dataset_builder = tfds.builder('svhn_cropped')
    train_split_name = 'train'
    eval_split_name = 'test'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      return tf.image.resize(img, [image_size, image_size], antialias=True)


  elif dataset == 'CELEBA':
    dataset_builder = tfds.builder('celeb_a')
    train_split_name = 'train'
    eval_split_name = 'validation'

    def resize_op(img):
      img = tf.image.convert_image_dtype(img, tf.float32)
      img = central_crop(img, 140)
      img = resize_small(img, image_size)
      return img
  else:
    raise NotImplementedError(
      f'Dataset {dataset} not yet supported.')

  def preprocess_fn(d):
    """Basic preprocessing function scales data to [0, 1) and randomly flips."""
    img = resize_op(d['image'])
    if random_flip and not evaluation:
      img = tf.image.random_flip_left_right(img)
    if uniform_dequantization:
      img = (tf.random.uniform(img.shape, dtype=tf.float32) + img * 255.) / 256.
    img += 1e-2*tf.random.normal(img.shape, dtype=tf.float32)
    return dict(image=img, label=d.get('label', None))

  def create_dataset(dataset_builder, split):
    dataset_options = tf.data.Options()
    dataset_options.experimental_optimization.map_parallelization = True
    dataset_options.threading.private_threadpool_size = 48
    dataset_options.threading.max_intra_op_parallelism = 1
    read_config = tfds.ReadConfig(options=dataset_options)
    if isinstance(dataset_builder, tfds.core.DatasetBuilder):
      dataset_builder.download_and_prepare()
      ds = dataset_builder.as_dataset(
        split=split, shuffle_files=True, read_config=read_config)
    else:
      ds = dataset_builder.with_options(dataset_options)
    ds = ds.repeat(count=num_epochs)
    ds = ds.shuffle(shuffle_buffer_size)
    ds = ds.map(preprocess_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    for batch_size in reversed(batch_dims):
      ds = ds.batch(batch_size, drop_remainder=True)
    return ds.prefetch(prefetch_size)

  train_ds = create_dataset(dataset_builder, train_split_name)
  eval_ds = create_dataset(dataset_builder, eval_split_name)
  return train_ds, eval_ds, dataset_builder
