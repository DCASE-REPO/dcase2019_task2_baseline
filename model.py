"""Model definitions for DCASE 2019 Task 2 Baseline models."""

import csv

import numpy as np
import tensorflow as tf

slim = tf.contrib.slim

def parse_hparams(flag_hparams):
  # Default values for all hyperparameters.
  hparams = tf.contrib.training.HParams(
      # Window and hop length for Short-Time Fourier Transform applied to audio
      # waveform to make the spectrogram.
      stft_window_seconds=0.025,
      stft_hop_seconds=0.010,
      # Parameters controlling conversion of spectrogram into mel spectrogram.
      mel_bands=96,
      mel_min_hz=20,
      mel_max_hz=20000,
      # log mel spectrogram = log(mel-spectrogram + mel_log_offset)
      mel_log_offset=0.001,
      # Window and hop length used to frame the log mel spectrogram into
      # examples.
      example_window_seconds=1.0,
      example_hop_seconds=0.5,
      # Number of examples in each batch fed to the model.
      batch_size=64,
      # For all CNN classifiers, whether to use global mean or max pooling.
      global_pool='mean',
      # Dropout keep probability. Set to zero to skip dropout layer.
      dropout=0.0,
      # Label smoothing. A setting of alpha will make the ground truth
      # label (1 - alpha) * 1.0 + alpha * 0.5 (smoothing towards the
      # uniform 0.5 rather than a hard 1.0). Set to zero to disable.
      lsmooth=0.0,
      # Standard deviation of the normal distribution with mean 0 used to
      # initialize the weights of the model. A standard deviation of zero
      # selects Xavier initialization. Biases are always initialized to 0.
      weights_init_stddev=0.0,
      # Whether to use batch norm, and corresponding decay and epsilon
      # if batch norm is enabled.
      bn=1,
      bndecay=0.9997,
      bneps=0.001,
      # Whether to warm-start from an existing checkpoint. Use --warmstart_*
      # flags to specify the checkpoint and include/exclude scopes.
      warmstart=0,
      # Type of optimizer (sgd, adam)
      opt='adam',
      # Learning rate.
      lr=1e-4,
      # Epsilon passed to the Adam optimizer.
      adam_eps=1e-8,
      # Learning rate decay. Set to zero to disable. If non-zero, then
      # learning rate gets multiplied by 'lrdecay' every 'decay_epochs'
      # epochs.
      lrdecay=0.0,
      # How many epochs to wait between each decay of learning rate.
      decay_epochs=2.5)

  # Let flags override default hparam values.
  hparams.parse(flag_hparams)

  return hparams

def get_global_pool_op(hparams):
  if hparams.global_pool == 'mean':
    return tf.reduce_mean
  elif hparams.global_pool == 'max':
    return tf.reduce_max
  else:
    raise NotImplementedError('Unknown global pooling function: %r', hparams.global_pool)

# Adapted from MobileNet v1's official TF-Slim implementation
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
def define_mobilenet_v1(features=None, hparams=None):
  """Defines MobileNet v1 CNN, without the classifier layer."""
  net = tf.expand_dims(features, axis=3)

  with slim.arg_scope([slim.conv2d], padding='SAME'):
    net = slim.conv2d(net, 16, kernel_size=3, stride=1, padding='SAME')
    stride_depths = [
        (1, 64),
        (2, 128),
        (1, 128),
        (2, 256),
        (1, 256),
        (2, 512),
        (1, 512),
        (1, 512),
        (1, 512),
        (1, 512),
        (1, 512),
        (2, 1024),
        (1, 1024),
    ]
    for (stride, depth) in stride_depths:
      # Create a separable convolution layer using a sequence of depthwise
      # and pointwise convolutions. We don't do it using a single call to
      # slim.separable_conv2d() because that adds a single batch norm layer
      # after both convolutions while we want a batch norm after each of
      # thed depthwise and pointwise convolutions.

      # Use num_outputs=None to force creating only depthwise convolution per
      # channel
      net = slim.separable_conv2d(net, num_outputs=None, kernel_size=[3, 3],
                                  depth_multiplier=1, stride=stride, padding='SAME')
      # Now create a separate pointwise (1x1) convolution.
      net = slim.conv2d(net, depth, kernel_size=[1, 1], stride=1, padding='SAME')

  net = get_global_pool_op(hparams)(net, axis=[1, 2], keepdims=True)
  net = slim.flatten(net)
  return net

def define_model(model_name=None, features=None, labels=None, num_classes=None,
                 hparams=None, epoch_batches=None, training=False):
  """Defines a classifier model.

  Args:
    model_name: one of ['mlp', 'cnn'], determines the model architecture.
    features: tensor containing a batch of input features.
    labels: tensor containing a batch of corresponding labels.
    num_classes: number of classes.
    hparams: model hyperparameters.
    epoch_batches: Number of batches in an epoch (used for controlling lr decay).
    training: True iff the model is being trained.

  Returns:
    global_step: tensor containing the global step.
    prediction: tensor containing the predictions from the classifier layer.
    loss: tensor containing the training loss for each batch.
    train_op: op that runs training (forward and backward passes) for each batch.
  """
  if hparams.weights_init_stddev == 0.0:
    weights_initializer = slim.initializers.xavier_initializer()
  else:
    weights_initializer = tf.truncated_normal_initializer(stddev=hparams.weights_init_stddev)

  if hparams.bn:
    normalizer = slim.batch_norm
    normalizer_params = {
        'center': True,
        'scale': True,
        'decay': hparams.bndecay,
        'epsilon': hparams.bneps
    }
  else:
    normalizer = normalizer_params = None

  with slim.arg_scope([slim.conv2d, slim.separable_conv2d],
                      weights_initializer=weights_initializer,
                      biases_initializer=tf.zeros_initializer(),
                      normalizer_fn=normalizer,
                      normalizer_params=normalizer_params,
                      activation_fn=tf.nn.relu,
                      trainable=training), \
       slim.arg_scope([slim.fully_connected],
                      weights_initializer=weights_initializer,
                      biases_initializer=tf.zeros_initializer(),
                      trainable=training), \
       slim.arg_scope([slim.batch_norm, slim.dropout],
                      is_training=training):

    global_step = tf.train.create_global_step()

    # Define the model without the classifier layer.
    if model_name == 'mobilenet-v1':
      embedding = define_mobilenet_v1(features=features, hparams=hparams)
    else:
      raise NotImplementedError('Unknown model %r' % model_name)

    if hparams.dropout > 0.0:
      embedding = slim.dropout(embedding, keep_prob=hparams.dropout)

    # Add the logits and the classifier layer.
    logits = slim.fully_connected(embedding, num_classes, activation_fn=None)
    prediction = tf.nn.sigmoid(logits)

  if training:
    # In training mode, also create loss and train op.
    xent = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits, label_smoothing=hparams.lsmooth,
        reduction=tf.losses.Reduction.NONE)
    loss = tf.reduce_mean(xent)
    tf.summary.scalar('loss', loss)

    if hparams.lrdecay > 0.0:
      decay_batches = int(epoch_batches * hparams.decay_epochs)
      lr = tf.train.exponential_decay(
          learning_rate=hparams.lr,
          global_step=global_step,
          decay_steps=decay_batches,
          decay_rate=hparams.lrdecay,
          staircase=True)
    else:
      lr = hparams.lr

    if hparams.opt == 'sgd':
      optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    elif hparams.opt == 'adam':
      optimizer = tf.train.AdamOptimizer(
          learning_rate=lr, epsilon=hparams.adam_eps)
    else:
      raise NotImplementedError('Unknown optimizer: %r' % hparams.opt)

    train_op = slim.learning.create_train_op(loss, optimizer)
  else:
    loss = None
    train_op = None

  return global_step, prediction, loss, train_op
