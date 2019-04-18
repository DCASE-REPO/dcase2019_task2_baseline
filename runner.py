#!/usr/bin/env python
"""Driver for DCASE 2019 Task 2 Baseline.

See README.md in this directory for a more detailed description.

Usage:

- Download Kaggle data: train_{curated,noisy}.{csv,zip}, test.zip. Unzip
  zip files into directories train_curated, train_noisy, test.

- We start by demonstrating how to train on the curated dataset and
  run inference on the test set. The procedure is similar if you wanted
  to train on the noisy dataset, or some combination of curated and noisy.
  We also support training on noisy data and then warmstarting curated
  training with a noisily trained checkpoint using the --warmstart* flags.
  See the README.md in this repo for more details.

- Shuffle and randomly split the training CSV into a train set and held-out
  validation set: train.csv, validation.csv.

- Prepare class map:
  $ make_class_map.py < /path/to/train_curated.csv > /path/to/class_map.csv
  This should match the class_map.csv provided in this repository.

- Train a model with checkpoints produced in a new train_dir:
  $ runner.py \
      --mode train \
      --model mobilenet-v1 \
      --class_map_path /path/to/class_map.csv \
      --train_clip_dir /path/to/train_curated \
      --train_csv_path /path/to/train.csv \
      --train_dir /path/to/train_dir
  To override default hyperparameters, also pass in the --hparams flag:
      --hparams name=value,name=value,..
  See model.parse_hparams() for default values of all hyperparameters.

- Evaluate the trained model on the validation set on all checkpoints
  in train_dir:
  $ runner.py \
      --mode eval \
      --model mobilenet-v1 \
      --class_map_path /path/to/class_map.csv \
      --eval_clip_dir /path/to/train_curated \
      --eval_csv_path /path/to/validation.csv \
      --eval_dir /path/to/eval_dir \
      --train_dir /path/to/train_dir
  (make sure to use the same hparams overrides as used in training)
  Evaluation iterates over all available checkpoints and writes marker
  files (containing per-class and overall lwlrap) in eval_dir for each
  checkpoint, so that it can be stopped and resumed safely without having
  to repeat any work.

- Training and evaluation will produce TensorFlow summaries in event log
  files in train_dir and eval_dir which you can view by running a TensorBoard
  server pointed at these directories. Typically, you would have several
  train/eval jobs running in parallel (one for each combination of
  hyperparameters in a grid search), and a single TensorBoard visualizer job
  that lets you look at the results from all the runs in real time.

- Run inference on a trained model to produce predictions in the Kaggle
  submission format in file submission.csv. You will do this inside a kernel
  to make your submission.
  $ runner.py \
      --mode inference \
      --model mobilenet-v1 \
      --class_map_path /path/to/class_map.csv \
      --inference_clip_dir /path/to/test \
      --inference_checkpoint /path/to/train_dir/model.ckpt-<N> \
      --predictions_csv_path /path/to/submission.csv
  (make sure to use the same hparams overrides as used in training)
"""

from __future__ import print_function

import argparse
import sys
import tensorflow as tf

import evaluation
import inference
import model
import train

def parse_flags(argv):
  parser = argparse.ArgumentParser(description='DCASE 2019 Task 2 Baseline')

  # Flags common to all modes.
  all_modes_group = parser.add_argument_group('Flags common to all modes')
  all_modes_group.add_argument(
      '--mode', type=str, choices=['train', 'eval', 'inference'], required=True,
      help='Run one of training, evaluation, or inference.')
  all_modes_group.add_argument(
      '--model', type=str, choices=['mobilenet-v1'],
      default='mobilenet-v1', required=True,
      help='Name of a model architecture. Current options: mobilenet-v1.')
  all_modes_group.add_argument(
      '--hparams', type=str, default='',
      help='Model hyperparameters in comma-separated name=value format.')
  all_modes_group.add_argument(
      '--class_map_path', type=str, default='', required=True,
      help='Path to CSV file containing map between class index and name.')

  # Flags for training only.
  training_group = parser.add_argument_group('Flags for training only')
  training_group.add_argument(
      '--train_clip_dir', type=str, default='',
      help='Path to directory containing training clips.')
  training_group.add_argument(
      '--train_csv_path', type=str, default='',
      help='Path to CSV file containing training clip filenames and labels.')
  training_group.add_argument(
      '--epoch_num_batches', type=int, default=0,
      help='Number of batches in an epoch.')
  training_group.add_argument(
      '--warmstart_checkpoint', type=str, default='',
      help='Path to a model checkpoint to use for warm-started training.')
  training_group.add_argument(
      '--warmstart_include_scopes', type=str, default='',
      help='Comma-separated list of variable scopes to include when loading '
      'the warm-start checkpoint.')
  training_group.add_argument(
      '--warmstart_exclude_scopes', type=str, default='',
      help='Comma-separated list of variable scopes to exclude when loading '
      'the warm-start checkpoint.')

  # Flags for training and evaluation.
  train_eval_group = parser.add_argument_group('Flags for training and eval')
  train_eval_group.add_argument(
      '--train_dir', type=str, default='',
      help='Path to a directory which will hold model checkpoints and other outputs.')

  # Flags for evaluation only.
  eval_group = parser.add_argument_group('Flags for evaluation only')
  eval_group.add_argument(
      '--eval_clip_dir', type=str, default='',
      help='Path to directory containing evaluation clips.')
  eval_group.add_argument(
      '--eval_csv_path', type=str, default='',
      help='Path to CSV file containing evaluation clip filenames and labels.')
  eval_group.add_argument(
      '--eval_dir', type=str, default='',
      help='Path to a directory holding eval results.')

  # Flags for inference only.
  inference_group = parser.add_argument_group('Flags for inference only')
  inference_group.add_argument(
      '--inference_checkpoint', type=str, default='',
      help='Path to a model checkpoint to use for inference.')
  inference_group.add_argument(
      '--inference_clip_dir', type=str, default='',
      help='Path to directory containing test clips.')
  inference_group.add_argument(
      '--predictions_csv_path', type=str, default='',
      help='Path to a CSV file in which to store predictions.')

  flags, rest_argv = parser.parse_known_args(argv)

  # Additional per-mode validation.
  try:
    if flags.mode == 'train':
      assert flags.train_clip_dir, 'Must specify --train_clip_dir'
      assert flags.train_csv_path, 'Must specify --train_csv_path'
      assert flags.train_dir, 'Must specify --train_dir'
      if 'lrdecay' in flags.hparams:
        assert flags.epoch_num_batches > 0, (
            'When using hparams.lrdecay, must specify --epoch_num_batches')
      if 'warmstart' in flags.hparams:
        assert flags.warmstart_checkpoint, (
            'When using hparams.warmstart, must specify --warmstart_checkpoint')

    elif flags.mode == 'eval':
      assert flags.eval_clip_dir, 'Must specify --eval_clip_dir'
      assert flags.eval_csv_path, 'Must specify --eval_csv_path'
      assert flags.eval_dir, 'Must specify --eval_dir'
      assert flags.train_dir, 'Must specify --train_dir'

    else:
      assert flags.mode == 'inference'
      assert flags.inference_checkpoint, 'Must specify --inference_checkpoint'
      assert flags.inference_clip_dir, 'Must specify --inference_clip_dir'
      assert flags.predictions_csv_path, 'Must specify --predictions_csv_path'
  except AssertionError as e:
    print('\nError: ', e, '\n', file=sys.stderr)
    parser.print_help(file=sys.stderr)
    sys.exit(1)

  return flags, rest_argv

flags = None

def main(argv):
  hparams = model.parse_hparams(flags.hparams)

  if flags.mode == 'train':
    def split_csv(scopes):
      return scopes.split(',') if scopes else None
    train.train(model_name=flags.model, hparams=hparams,
                class_map_path=flags.class_map_path,
                train_csv_path=flags.train_csv_path,
                train_clip_dir=flags.train_clip_dir,
                train_dir=flags.train_dir,
                epoch_batches=flags.epoch_num_batches,
                warmstart_checkpoint=flags.warmstart_checkpoint,
                warmstart_include_scopes=split_csv(flags.warmstart_include_scopes),
                warmstart_exclude_scopes=split_csv(flags.warmstart_exclude_scopes))

  elif flags.mode == 'eval':
    evaluation.evaluate(model_name=flags.model, hparams=hparams,
                        class_map_path=flags.class_map_path,
                        eval_csv_path=flags.eval_csv_path,
                        eval_clip_dir=flags.eval_clip_dir,
                        eval_dir=flags.eval_dir,
                        train_dir=flags.train_dir)

  else:
    assert flags.mode == 'inference'
    inference.predict(model_name=flags.model, hparams=hparams,
                      class_map_path=flags.class_map_path,
                      inference_clip_dir=flags.inference_clip_dir,
                      inference_checkpoint=flags.inference_checkpoint,
                      predictions_csv_path=flags.predictions_csv_path)

if __name__ == '__main__':
  flags, sys.argv = parse_flags(sys.argv)
  tf.app.run(main)
