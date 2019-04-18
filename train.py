"""Trainer for DCASE 2019 Task 2 Baseline models."""

from __future__ import print_function

import os
import sys

import tensorflow as tf

import inputs
import model

def train(model_name=None, hparams=None, class_map_path=None, train_csv_path=None, train_clip_dir=None,
          train_dir=None, epoch_batches=None, warmstart_checkpoint=None,
          warmstart_include_scopes=None, warmstart_exclude_scopes=None):
  """Runs the training loop."""
  print('\nTraining model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Training data: clip dir {} and labels {}'.format(train_clip_dir, train_csv_path))
  print('Training dir {}\n'.format(train_dir))

  with tf.Graph().as_default():
    # Create the input pipeline.
    features, labels, num_classes, input_init = inputs.train_input(
        train_csv_path=train_csv_path, train_clip_dir=train_clip_dir, class_map_path=class_map_path, hparams=hparams)
    # Create the model in training mode.
    global_step, prediction, loss_tensor, train_op = model.define_model(
        model_name=model_name, features=features, labels=labels, num_classes=num_classes,
        hparams=hparams, epoch_batches=epoch_batches, training=True)

    # Define our own checkpoint saving hook, instead of using the built-in one,
    # so that we can specify additional checkpoint retention settings.
    saver = tf.train.Saver(
        max_to_keep=10000, keep_checkpoint_every_n_hours=0.25)
    saver_hook = tf.train.CheckpointSaverHook(
        save_steps=100, checkpoint_dir=train_dir, saver=saver)

    summary_op = tf.summary.merge_all()
    summary_hook = tf.train.SummarySaverHook(
        save_steps=10, output_dir=train_dir, summary_op=summary_op)

    if hparams.warmstart:
      var_include_scopes = warmstart_include_scopes
      if not var_include_scopes: var_include_scopes = None
      var_exclude_scopes = warmstart_exclude_scopes
      if not var_exclude_scopes: var_exclude_scopes = None
      restore_vars = tf.contrib.framework.get_variables_to_restore(
          include=var_include_scopes, exclude=var_exclude_scopes)
      # Only restore trainable variables, we don't want to restore
      # batch-norm or optimizer-specific local variables.
      trainable_vars = set(tf.contrib.framework.get_trainable_variables())
      restore_vars = [var for var in restore_vars if var in trainable_vars]

      print('Warm-start: restoring variables:\n%s\n' % '\n'.join([x.name for x in restore_vars]))
      print('Warm-start: restoring from ', warmstart_checkpoint)
      assert restore_vars, 'No warm-start variables to restore!'
      restore_op, feed_dict = tf.contrib.framework.assign_from_checkpoint(
          model_path=warmstart_checkpoint, var_list=restore_vars, ignore_missing_vars=True)

      scaffold = tf.train.Scaffold(
          init_fn=lambda scaffold, session: session.run(restore_op, feed_dict),
          summary_op=summary_op, saver=saver)
    else:
      scaffold = None

    with tf.train.SingularMonitoredSession(hooks=[saver_hook, summary_hook],
                                           checkpoint_dir=train_dir,
                                           scaffold=scaffold) as sess:
      sess.raw_session().run(input_init)
      while not sess.should_stop():
        step, _, pred, loss = sess.run([global_step, train_op, prediction, loss_tensor])
        print(step, loss)
        sys.stdout.flush()
