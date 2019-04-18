"""Inference for DCASE 2019 Task2 Baseline models."""

from __future__ import print_function

import csv
from collections import defaultdict
import os
import sys

import numpy as np
import tensorflow as tf

import inputs
import model

def predict(model_name=None, hparams=None, inference_clip_dir=None,
            class_map_path=None, inference_checkpoint=None, predictions_csv_path=None):
  """Runs the prediction loop, producting prediction output in Kaggle submission format."""
  print('\nPrediction for model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Prediction data: clip dir {} and checkpoint {}'.format(inference_clip_dir, inference_checkpoint))
  print('Predictions in CSV {}\n'.format(predictions_csv_path))

  # Read in class map CSV into a class index -> class name map.
  class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}
  class_names = [class_map[i] for i in range(len(class_map))]
  num_classes = len(class_names)

  with tf.Graph().as_default():
    clip_placeholder = tf.placeholder(tf.string, [])  # Fed during prediction loop.

    # Use a simpler in-order input pipeline without labels for prediction
    # compared to the one used in training. The features consist of a batch of
    # all possible framed log mel spectrum examples from the same clip.
    features = inputs.clip_to_log_mel_examples(
        clip_placeholder, clip_dir=inference_clip_dir, hparams=hparams)

    # Creates the model in prediction mode.
    _, prediction, _, _ = model.define_model(
        model_name=model_name, features=features, num_classes=num_classes,
        hparams=hparams, training=False)

    with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=inference_checkpoint) as sess:

      inference_clips = sorted(os.listdir(inference_clip_dir))
      pred_writer = csv.DictWriter(open(predictions_csv_path, 'w'), fieldnames=['fname'] + class_names)
      pred_writer.writeheader()

      for (i, clip) in enumerate(inference_clips):
        print(i+1, clip)
        sys.stdout.flush()

        scores = sess.run(prediction, {clip_placeholder: clip})
        # Average per-example scores to get overall clip scores.
        scores = np.average(scores, axis=0)

        row_dict = {class_map[i]: scores[i] for i in range(len(scores))}
        row_dict['fname'] = clip
        pred_writer.writerow(row_dict)
        sys.stdout.flush()
