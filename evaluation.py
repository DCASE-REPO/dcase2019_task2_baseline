"""Evaluation for DCASE 2019 Task 2 Baseline models."""

import csv
from collections import defaultdict
import os
import random
import sys

import numpy as np
import sklearn.metrics
import tensorflow as tf

import inputs
import model


class Lwlrap(object):
  """Computes label-weighted label-ranked average precision (lwlrap)."""

  def __init__(self, class_map):
    self.num_classes = 0
    self.total_num_samples = 0
    self._class_map = class_map

  def accumulate(self, batch_truth, batch_scores):
    """Accumulate a new batch of samples into the metric.

    Args:
      truth: np.array of (num_samples, num_classes) giving boolean
        ground-truth of presence of that class in that sample for this batch.
      scores: np.array of (num_samples, num_classes) giving the 
        classifier-under-test's real-valued score for each class for each
        sample.
    """
    assert batch_scores.shape == batch_truth.shape
    num_samples, num_classes = batch_truth.shape
    if not self.num_classes:
      self.num_classes = num_classes
      self._per_class_cumulative_precision = np.zeros(self.num_classes)
      self._per_class_cumulative_count = np.zeros(self.num_classes, 
                                                  dtype=np.int)
    assert num_classes == self.num_classes
    for truth, scores in zip(batch_truth, batch_scores):
      pos_class_indices, precision_at_hits = (
        self._one_sample_positive_class_precisions(scores, truth))
      self._per_class_cumulative_precision[pos_class_indices] += (
        precision_at_hits)
      self._per_class_cumulative_count[pos_class_indices] += 1
    self.total_num_samples += num_samples

  def _one_sample_positive_class_precisions(self, scores, truth):
    """Calculate precisions for each true class for a single sample.

    Args:
      scores: np.array of (num_classes,) giving the individual classifier scores.
      truth: np.array of (num_classes,) bools indicating which classes are true.

    Returns:
      pos_class_indices: np.array of indices of the true classes for this sample.
      pos_class_precisions: np.array of precisions corresponding to each of those
        classes.
    """
    num_classes = scores.shape[0]
    pos_class_indices = np.flatnonzero(truth > 0)
    # Only calculate precisions if there are some true classes.
    if not len(pos_class_indices):
      return pos_class_indices, np.zeros(0)
    # Retrieval list of classes for this sample. 
    retrieved_classes = np.argsort(scores)[::-1]
    # class_rankings[top_scoring_class_index] == 0 etc.
    class_rankings = np.zeros(num_classes, dtype=np.int)
    class_rankings[retrieved_classes] = range(num_classes)
    # Which of these is a true label?
    retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
    retrieved_class_true[class_rankings[pos_class_indices]] = True
    # Num hits for every truncated retrieval list.
    retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
    # Precision of retrieval list truncated at each hit, in order of pos_labels.
    precision_at_hits = (
        retrieved_cumulative_hits[class_rankings[pos_class_indices]] / 
        (1 + class_rankings[pos_class_indices].astype(np.float)))
    return pos_class_indices, precision_at_hits

  def per_class_lwlrap(self):
    """Return a vector of the per-class lwlraps for the accumulated samples."""
    return (self._per_class_cumulative_precision /
            np.maximum(1, self._per_class_cumulative_count))

  def per_class_weight(self):
    """Return a normalized weight vector for the contributions of each class."""
    return (self._per_class_cumulative_count / 
            float(np.sum(self._per_class_cumulative_count)))

  def overall_lwlrap(self):
    """Return the scalar overall lwlrap for cumulated samples."""
    return np.sum(self.per_class_lwlrap() * self.per_class_weight())

  def __str__(self):
    per_class_lwlrap = self.per_class_lwlrap()
    # List classes in descending order of lwlrap.
    s = (['Lwlrap(%s) = %.6f' % (name, lwlrap) for (lwlrap, name) in
             sorted([(per_class_lwlrap[i], self._class_map[i]) for i in range(self.num_classes)],
                    reverse=True)])
    s.append('Overall lwlrap = %.6f' % (self.overall_lwlrap()))
    return '\n'.join(s)


# Alternate implementation of lwlrap computation that uses sklearn.metrics. Useful for
# debugging metric computation. Does not provide per-class metrics.
class LwlrapSklearn(object):
  def __init__(self, class_map):
    self._class_map = class_map
    self.num_classes = len(class_map)
    self._truth = np.array([], dtype=np.float32).reshape((0, self.num_classes))
    self._scores = np.array([], dtype=np.float32).reshape((0, self.num_classes))

  def accumulate(self, batch_truth, batch_scores):
    self._truth = np.concatenate((self._truth, batch_truth))
    self._scores = np.concatenate((self._scores, batch_scores))

  def overall_lwlrap(self):
    # sklearn doesn't correctly apply weighting to samples with no labels, so just skip them.
    sample_weight = np.sum(self._truth > 0, axis=1)
    nonzero_weight_sample_indices = np.flatnonzero(sample_weight > 0)
    overall_lwlrap = sklearn.metrics.label_ranking_average_precision_score(
        self._truth[nonzero_weight_sample_indices, :] > 0,
        self._scores[nonzero_weight_sample_indices, :],
        sample_weight=sample_weight[nonzero_weight_sample_indices])
    return overall_lwlrap

  def __str__(self):
    return 'Overall lwlrap: %.6f' % self.overall_lwlrap()


def evaluate(model_name=None, hparams=None, eval_csv_path=None, eval_clip_dir=None,
             class_map_path=None, checkpoint_path=None):
  """Runs the evaluation loop."""
  print('\nEvaluation for model:{} with hparams:{} and class map:{}'.format(model_name, hparams, class_map_path))
  print('Evaluation data: clip dir {} and labels {}'.format(eval_clip_dir, eval_csv_path))
  print('Checkpoint: {}\n'.format(checkpoint_path))

  with tf.Graph().as_default():
    label_class_index_table, num_classes = inputs.get_class_map(class_map_path)
    csv_record = tf.placeholder(tf.string, [])  # fed during evaluation loop.

    # Use a simpler in-order input pipeline for eval than the one used in
    # training, since we don't want to shuffle examples across clips.
    # The features consist of a batch of all possible framed log mel spectrum
    # examples from the same clip. The labels in this case will contain a batch
    # of identical 1-hot vectors.
    features, labels = inputs.record_to_labeled_log_mel_examples(
        csv_record, clip_dir=eval_clip_dir, hparams=hparams,
        label_class_index_table=label_class_index_table, num_classes=num_classes)

    # Create the model in prediction mode.
    global_step, prediction, _, _ = model.define_model(
        model_name=model_name, features=features, num_classes=num_classes,
        hparams=hparams, training=False)

    # Write evaluation graph to checkpoint directory.
    tf.train.write_graph(tf.get_default_graph().as_graph_def(add_shapes=True),
                         os.path.dirname(checkpoint_path), 'eval.pbtxt')

    with tf.train.SingularMonitoredSession(checkpoint_filename_with_path=checkpoint_path) as sess:
      # Read in class map CSV into a class index -> class name map.
      class_map = {int(row[0]): row[1] for row in csv.reader(open(class_map_path))}

      # Read in the validation CSV, skipping the header.
      eval_records = open(eval_csv_path).readlines()[1:]
      # Shuffle the lines so that as we print incremental stats, we get good
      # coverage across classes and get a quick initial impression of how well
      # the model is doing across classes well before evaluation is completed.
      random.shuffle(eval_records)

      lwlrap = Lwlrap(class_map)

      for (i, record) in enumerate(eval_records):
        record = record.strip()
        print record
        sys.stdout.flush()

        try:
          actual, predicted = sess.run([labels, prediction], {csv_record: record})
        except:
          print lwlrap
          raise

        # By construction, actual consists of identical rows, where each row is
        # the same 1-hot label (because we are looking at features from the same
        # clip). So we can just use the first row as the ground truth.
        actual_labels = actual[0]

        # We make a clip prediction by averaging the prediction scores across
        # all examples for the clip.
        predicted_labels = np.average(predicted, axis=0)

        # Update eval metric, and periodically print.
        lwlrap.accumulate(actual_labels[np.newaxis, :], predicted_labels[np.newaxis, :])
        if i % 50 == 0:
          print
          print lwlrap
          print
          sys.stdout.flush()

      # Print final eval metric values.
      print
      print lwlrap
      print
