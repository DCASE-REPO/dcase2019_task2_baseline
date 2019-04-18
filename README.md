
# Baseline system for Task 2 of DCASE 2019

This is the baseline system for [Task 2](http://dcase.community/challenge2019/task-audio-tagging) of the
[DCASE 2019](http://dcase.community/challenge2019) challenge. The system implements an audio classifier using
an efficient MobileNet v1 convolutional neural network, which takes log mel spectrogram features as input and
produces predicted scores for the 80 classes in the dataset.

Task 2 is hosted as a [Kaggle challenge](https://kaggle.com/c/freesound-audio-tagging-2019)
and the baseline system was used to produce the baseline submission named _Challenge Baseline_ in the
[Kaggle challenge leaderboard](https://kaggle.com/c/freesound-audio-tagging-2019/leaderboard).

## Installation

* Clone [this GitHub repository](https://github.com/DCASE-REPO/dcase2019_task2_baseline).
* Requirements: python, numpy, sklearn, tensorflow. The baseline was tested with
  on a Debian-like Linux OS with Python v2.7.16/v3.6.5, NumPy v1.16.2, Scikit-learn v0.20.3,
  TensorFlow v1.13.1.
* Download the dataset [from Kaggle](https://kaggle.com/c/freesound-audio-tagging-2019/data):
  `audio_curated.zip`, `audio_noisy.zip`, `test.zip`, `train_curated.csv`, `train_noisy.csv`.
  Unzip the zip files to produce `train_curated`, `train_noisy`, and `test` directories.

## Code Layout

* `runner.py`: Main driver. Run runner.py --help to see all available flags.
* `train.py`: Training loop. Called by `runner.py` when passed `--mode train`
* `evaluation.py`: Computes evaluation metrics. Called by `runner.py` when passed `--mode eval`
* `inference.py`: Generates model predictions. Called by `runner.py` when passed `--mode inference`
* `inputs.py`: TensorFlow input pipeline for decoding CSV input and WAV files, and constructing
   framed and labeled log mel spectrogtram examples.
* `model.py`: Tensorflow model and hyperparameter definitions.
* `make_class_map.py`: Utility to create a class map from the training dataset.

## Usage

* Prepare a class map, which is a CSV file that maps between class indices and class names, and is
  used by various parts of the system:
```shell
$  make_class_map.py < /path/to/train_curated.csv > /path/to/class_map.csv
```
  Note that we have provided a canonical `class_map.csv` (in this repo) where the order of classes
  matches the order of columns required in Kaggle submission files.

* If you want to use a validation set to compare models, prepare a hold-out validation set by moving
  some random fraction (say, 10%) of the rows from the training CSV files into a validation CSV
  file, while keeping the same header line. This is a multi-label task so, to avoid any bias in the
  split, make sure that the training and validation sets have roughly the same number of labels per
  class. The rows of the original training CSV files are not necessarily in random order so make sure
  to shuffle rows when making splits.

* Train a model on the curated data with checkpoints created in `train_dir`:
```shell
$ main.py \
    --mode train \
    --model mobilenet-v1 \
    --class_map_path /path/to/class_map.csv \
    --train_clip_dir /path/to/train_curated \
    --train_csv_path /path/to/train.csv \
    --train_dir /path/to/train_dir
```
  This will produce checkpoint files in `train_dir` having the name prefix `model.ckpt-N` with
  increasing N, where N represents the number of batches of examples seen by the model.  By default,
  checkpoints are written every 500 batches (edit the saver settings in `train.py`o to change this).

  This will also print the loss at each step on standard output, as well as add summary entries to a
  TensorFlow event log in `train_dir` which can be viewed by running a TensorBoard server pointed at
  that directory.

  By default, this will use the default hyperparameters defined inside `model.py`. These can be
  overridden using the `--hparams` flag to pass in comma-separated `name=value` pairs. For example,
  `--hparams batch_size=32,lr=0.01` will use a batch size of 32 and a learning rate of 0.01.  For
  more information about the hyperparameters, see below in the Model description section. Note that
  if you use non-default hyperparameters during training, you must use the same hyperparameters when
  running the evaluation and inference steps described below.

* Evaluate the model checkpoints in the training directory on the (curated) validation set:
```shell
$ main.py \
    --mode eval \
    --model mobilenet-v1 \
    --class_map_path /path/to/class_map.csv \
    --eval_clip_dir /path/to/train_curated \
    --eval_csv_path /path/to/validation.csv \
    --train_dir /path/to/train_dir \
    --eval_dir /path/to/eval_dir
```
  This will loop through all checkpoints in `train_dir` and run evaluation on each checkpoint. A
  running Lwlrap (per-class and overall) will be periodically printed on stdout. The final Lwlrap
  will be printed on stdout and logged into a text file named `eval-<N>.txt` in `eval_dir` (these
  files are checked by the evaluator so that if it is interrupted and re-started on the same data,
  then it will skip re-evaluating any checkpoints that have already been evaluated).  Lwlrap summary
  values will also be written in TensorFlow event logs in `eval_dir` (both the full Lwlrap as well
  as a partial Lwlrap from 5% of the data) which can be viewed in TensorBoard. Evaluation can be
  sped up by modifying the top-level loop in `evaluation.py` to look at every K-th checkpoint
  instead of every single one, or by spawning multiple copies of eval where each one is looking at
  a different subset of checkpoints.

* Generate predictions in `submission.csv` from a particular trained model checkpoint for submission
  to Kaggle:
```shell
$ main.py \
    --mode inference \
    --model mobilenet-v1 \
    --class_map_path /path/to/class_map.csv \
    --inference_clip_dir /path/to/test \
    --inference_checkpoint /path/to/train_dir/model.ckpt-<N> \
    --predictions_csv_path /path/to/submission.csv
```

* We also support warm-starting training of a model using weights from the checkpoint of a previous
  training run. This allows, for example, training a model on the noisy dataset and then
  warm-starting a curated training run using a noisily trained checkpoint.
```shell
$ main.py \
    --mode train \
    --model mobilenet-v1 \
    --class_map_path /path/to/class_map.csv \
    --train_clip_dir /path/to/train_curated \
    --train_csv_path /path/to/train.csv \
    --train_dir /path/to/train_dir \
    --hparams warmstart=1,<other hparams ...> \
    --warmstart_checkpoint=/path/to/model.ckpt-<N> \
    --warmstart_include_scopes=<excludescope>,... \\
    --warmstart_exclude_scopes=<includescope>,...
```
  This will initialize training with weights taken from `model.ckpt-<N>`, which assumes that the
  model being trained and the model that generated the checkpoint have compatible architectures and
  layer names. If the `--warmstart_{exclude,include}_scopes` flags are not specified, then all
  weights are used.  The scope flags specify comma-separated lists of TensorFlow scope names
  matching variables that are to be included and excluded. The include scope defaults to match all
  variables, and the exclude scope defaults to match no variables. Inclusions are applied before
  exclusions. For example, if you had a trained model which had a stack of convolution layers
  followed by a single fully connected layer with a scope named`fully_connected`, and you wanted to
  use the convolution weights only, then you could specify
  `--warmstart_exclude_scopes=fully_connected` to exclude the last layer.

## Model Description and Performance

We use the MobileNet v1 convolutional neural network architecture [1], which gives us a light-weight
and efficient model with reasonable performance.

### Input features

We use frames of log mel spectrogram as input features, which has been demonstrated to work well for
audio CNN classifiers by Hershey et. al. [2].

We compute log mel spectrogram examples as follows:

* The incoming audio is always at 44.1 kHz mono.

* The [spectrogram](https://en.wikipedia.org/wiki/Spectrogram) is computed using the magnitude of
  the [Short-Time Fourier Transform](https://en.wikipedia.org/wiki/Short-time_Fourier_transform)
  (STFT) with a window size of 25ms, a window hop size of 10ms, and a periodic Hann window.

* The mel spectrogram is computed by mapping the spectrogram to 96 mel bins covering the range 20 Hz - 20 kHz.
  The [mel scale](https://en.wikipedia.org/wiki/Mel_scale) is intended to better represent
  human audio perception by using more bins in the lower frequencies and fewer bins in the higher
  frequencies.

* The stabilized log mel spectrogram is computed by applying `log`(mel spectrogram + 0.001) where
  the offset of 0.001 is used to avoid taking a logarithm of 0. The compressive non-linearity of the
  logarithm is used to reduce the dynamic range of the feature values.

* The log mel spectrogram is then framed into overlapping examples with a window size of 1s and a
  hop size of 0.5s.  The overlap allows generating more examples from the same data than with no
  overlap, which helps to increase the effective size of the dataset, and also gives the model a
  little more context to learn from because it now sees the same slice of audio in different
  examples with different context windows.

The input pipeline parses CSV records, decodes WAV files, creates examples containing log mel
spectrum examples with multi-1-hot-encoded labels, shuffles them across clips, and does all of this
on-the-fly and purely in TensorFlow, without requiring any Python preprocessing or separate feature
generation or storage step. This gives you the flexibility of including feature generation
hyperparameters in your grid search without having to generate features offline, and also improves
performance if you are running a version of TensorFlow that includes GPU-accelerated versions of the
FFT and other signal processing operations used by this input pipeline.

### Architecture

We use a variant of the MobileNet v1 CNN architecture [1] which consists of a stack of separable
convolution layers, each of which are composed of a pair of a depthwise convolution (which acts on
each depth channel independently) followed by a 1x1 pointwise convolution (which acts across all
channels).  This factoring greatly reduces the number of layer weights (up to 9 times less than
standard convolutions if using 3x3 filters) with only a small reduction in model accuracy.

The model layers are listed in the table below using notation `C(kernel size, stride, depth)` and
`SC(kernel size, stride, depth)` to denote 2-D convolution and separable convolution layers,
respectively.  ReduceMax or Mean applies a maximum-value or mean-value reduction, respectively,
across the first two dimensions. Logistic applies a fully-connected linear layer followed by a
sigmoid classifier. Activation shapes do not include the batch dimension.

 Layer              | Activation shape | # Weights | # Multiplies
--------------------|------------------|----------:|---------------:
Input               | (100, 96, 1)     | -         | -
C(3x3, 16, 1)       | (100, 96, 16)    | 144       | 1.4M
SC(3x3, 64, 1)      | (100, 96, 64)    | 1.2K      | 11.2M
SC(3x3, 128, 2)     | (50, 48, 128)    | 8.8K      | 21M
SC(3x3, 128, 1)     | (50, 48, 128)    | 17.5K     | 42.1M
SC(3x3, 256, 2)     | (25, 24, 256)    | 33.9K     | 20.4M
SC(3x3, 256, 1)     | (25, 24, 256)    | 67.8K     | 40.7M
SC(3x3, 512, 2)     | (13, 12, 512)    | 133.4K    | 20.8M
SC(3x3, 512, 1)     | (13, 12, 512)    | 266.8K    | 41.6M
SC(3x3, 512, 1)     | (13, 12, 512)    | 266.8K    | 41.6M
SC(3x3, 512, 1)     | (13, 12, 512)    | 266.8K    | 41.6M
SC(3x3, 512, 1)     | (13, 12, 512)    | 266.8K    | 41.6M
SC(3x3, 512, 1)     | (13, 12, 512)    | 266.8K    | 41.6M
SC(3x3, 1024, 2)    | (7, 6, 1024)     | 528.9K    | 22.2M
SC(3x3, 1024, 1)    | (7, 6, 1024)     | 1.1M      | 44.4M
ReduceMax/Mean      | (1, 1, 1024)     | 0         | 0
Logistic(80)        | (80,)            | 81.9K     | 81.9K
**Total**           |                  | **3.3M**| **432.4M**

Our MobileNet baseline is ~8x smaller than a ResNet-50 and uses ~4x less compute.

Our implementation follows the version released as part of the [TF-Slim model
library](https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py) with
the main difference being that we tried a stride of 1 instead of 2 in the first convolution layer,
which gives us a little more time-frequency resolution in the layers before the final reduce. Note
that MobileNet naturally allows scaling the model size and compute by changing the number of
filters in each layer by the same factor.

The classifier predicts 80 scores for individual 0.5s-wide examples, which we average
across time for all framed examples generated from a clip to produce a clip-wide
list of 80 scores.

### Hyperparameters

The following hyperparameters, defined with their default values in `runner.py`,
are used in the input pipeline and model definition.

```python
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
    # Note that this is the keep probability, not the the dropout rate.
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
    # Learning rate exponential decay. Set to zero to disable. If non-zero,
    # learning rate gets multiplied by 'lrdecay' every 'decay_epochs'
    # epochs.
    lrdecay=0.0,
    # How many epochs to wait between each decay of learning rate.
    decay_epochs=2.5)
```

In order to override the defaults, pass the `--hparams` flag a comma-separated list of `name=value`
pairs.  For example, `--hparams example_window_seconds=0.5,batch_size=32,lr=0.01` will use examples
of size 0.5s, a batch size of 32, and a learning rate of 0.01.

A few notes about some of the hyperparameters:

* Label Smoothing: This was introduced in Inception v3 [3] and converts each ground truth label into a blend of the original label and 0.5 (representing a uniform probability distribution). The higher the `lsmooth` hyperparameter (in the range [0, 1]), the more the labels are blended towards 0.5. This is useful when training with noisy labels that we don't trust.

* Warm Start: As mentioned in the Usage section earlier, specifying `warmstart=1` requires also specifying a `--warmstart_checkpoint` flag as well as optionally the `--warmstart_{include,exclude}_scopes` flags.

* Exponential Decay: Setting `lrdecay` greater than 0 will enable exponential decay of learning rate, as described in the TensorFlow documentation to [tf.train.exponential_decay](https://www.tensorflow.org/api_docs/python/tf/train/exponential_decay). You will also need to specify the `--epoch_num_batches` flag to specify the number of batches in an epoch for the training dataset that you will be using, as well as the `decay_epochs` hyperparameter if you want to change the default number of epochs before the learning rate changes.

An aside on computing epoch sizes: We can use a simple back-of-the-envelope calculation of epoch
sizes from dataset size because we use uncompressed WAVs with a fixed sample rate (44.1 kHz) and a
fixed sample size (16-bit signed PCM). For example, the total size of the `train_curated` directory
containing all the clips is 3.2 GB. Each sample is 2 bytes, and each second needs 44100 samples, so
the total number of seconds in the training set is (3.2 * 2 ^ 30) / (2 * 44100) = ~38956. We frame
examples with a hop of 0.5s seconds and if we have a batch size of 64, then the number of batches in
an epoch would be 38956 / 0.5 / 64 = ~1217. Similarly, for a batch size of 64, the number of batches
in an epoch of the noisy dataset is ~9148 and the number of batches in an epoch of the combined
datasets is ~10272. Note that you will need to adjust these numbers because you are probably using a
different, smaller training dataset after splitting out a validation dataset.

Note that the input pipeline defined in `inputs.py` contains some fixed parameters that affect training speed
(parallelism of example extraction, various buffer sizes, prefetch parameters, etc) and which should be changed
if the available machine resources don't match what we used to build the baseline.

### Performance

The baseline system submission was produced using the following recipe:

* We first train the model on the noisy dataset which lets us learn an audio representation from a lot of data. We use dropout and label smoothing to deal with noisy labels and to avoid overfitting.

* We then warmstart training on the curated dataset using a noisily trained checkpoint. This transfer learning approach lets us use all the data without having to deal with the domain mismatch if we tried to train on both noisy and curated in the same training run. We continue to use dropout and label smoothing because even the curated labels are not 100% trustworthy and we do not want to overfit to the smaller dataset.

* We used the following hyperparameters:

  * Noisy training: batch size 64, learning rate 1e-4, no learning rate decay, dropout keep probability 0.8, label smoothing factor 0.3, global max pooling. We trained on the entire noisy training set for ~10 epochs (~3 hrs on a Tesla V-100).

  * Curated training: batch size 64, learning rate 3e-3, learning rate decay 0.94, dropout keep probability 0.6, label smoothing factor 0.1, global max pooling, warmstarted using all weights from all layers. We trained on the entire curated training set for ~100 epochs (~5 hrs on a Tesla V-100).

The baseline system achieves Lwlraps of ~0.546 on the entire test set and ~0.537 on the public leaderboard. Per-class Lwlraps on the entire test set, in decreasing order:
```
Lwlrap(Bicycle_bell) = 0.893735
Lwlrap(Purr) = 0.873156
Lwlrap(Screaming) = 0.865613
Lwlrap(Chewing_and_mastication) = 0.849182
Lwlrap(Bass_drum) = 0.841581
Lwlrap(Keys_jangling) = 0.833121
Lwlrap(Applause) = 0.826680
Lwlrap(Burping_and_eructation) = 0.823632
Lwlrap(Toilet_flush) = 0.810323
Lwlrap(Computer_keyboard) = 0.800228
Lwlrap(Shatter) = 0.782168
Lwlrap(Cheering) = 0.773989
Lwlrap(Writing) = 0.765854
Lwlrap(Harmonica) = 0.763851
Lwlrap(Zipper_(clothing)) = 0.713306
Lwlrap(Scissors) = 0.713218
Lwlrap(Electric_guitar) = 0.707559
Lwlrap(Church_bell) = 0.699981
Lwlrap(Microwave_oven) = 0.689582
Lwlrap(Marimba_and_xylophone) = 0.687920
Lwlrap(Motorcycle) = 0.676952
Lwlrap(Sink_(filling_or_washing)) = 0.674356
Lwlrap(Bass_guitar) = 0.673926
Lwlrap(Accordion) = 0.646651
Lwlrap(Accelerating_and_revving_and_vroom) = 0.644568
Lwlrap(Sneeze) = 0.639724
Lwlrap(Bathtub_(filling_or_washing)) = 0.636410
Lwlrap(Car_passing_by) = 0.626718
Lwlrap(Knock) = 0.624838
Lwlrap(Female_singing) = 0.616608
Lwlrap(Sigh) = 0.611468
Lwlrap(Drawer_open_or_close) = 0.611059
Lwlrap(Crowd) = 0.609081
Lwlrap(Hi-hat) = 0.601695
Lwlrap(Meow) = 0.600351
Lwlrap(Fill_(with_liquid)) = 0.592782
Lwlrap(Bark) = 0.581057
Lwlrap(Clapping) = 0.578710
Lwlrap(Water_tap_and_faucet) = 0.574853
Lwlrap(Stream) = 0.557559
Lwlrap(Race_car_and_auto_racing) = 0.555504
Lwlrap(Yell) = 0.538160
Lwlrap(Hiss) = 0.531418
Lwlrap(Traffic_noise_and_roadway_noise) = 0.531006
Lwlrap(Cutlery_and_silverware) = 0.528571
Lwlrap(Acoustic_guitar) = 0.520993
Lwlrap(Frying_(food)) = 0.514982
Lwlrap(Crackle) = 0.505139
Lwlrap(Strum) = 0.504977
Lwlrap(Slam) = 0.504614
Lwlrap(Cricket) = 0.503219
Lwlrap(Skateboard) = 0.488623
Lwlrap(Gong) = 0.471829
Lwlrap(Gurgling) = 0.457993
Lwlrap(Gasp) = 0.457561
Lwlrap(Whispering) = 0.454770
Lwlrap(Waves_and_surf) = 0.444573
Lwlrap(Glockenspiel) = 0.439053
Lwlrap(Walk_and_footsteps) = 0.432062
Lwlrap(Chink_and_clink) = 0.418145
Lwlrap(Buzz) = 0.417702
Lwlrap(Male_singing) = 0.401102
Lwlrap(Trickle_and_dribble) = 0.395132
Lwlrap(Tick-tock) = 0.389962
Lwlrap(Female_speech_and_woman_speaking) = 0.381376
Lwlrap(Printer) = 0.373431
Lwlrap(Fart) = 0.369772
Lwlrap(Finger_snapping) = 0.369079
Lwlrap(Child_speech_and_kid_speaking) = 0.356280
Lwlrap(Squeak) = 0.324860
Lwlrap(Raindrop) = 0.307449
Lwlrap(Run) = 0.305023
Lwlrap(Drip) = 0.273305
Lwlrap(Male_speech_and_man_speaking) = 0.262371
Lwlrap(Mechanical_fan) = 0.261341
Lwlrap(Tap) = 0.255278
Lwlrap(Bus) = 0.232600
Lwlrap(Dishes_and_pots_and_pans) = 0.221663
Lwlrap(Cupboard_open_or_close) = 0.218870
Lwlrap(Chirp_and_tweet) = 0.127003
```

## Ideas for improvement

* Minimize domain mismatch better by using audio features that are less sensitive to loudness and noise (e.g., PCEN [4]), or by using the right kind of data augmentation.

* More sophisticated transfer learning: perhaps train a deeper model on the noisy data, and use only the lower layers to warmstart the curated training.

* Explore other kinds of regularization and loss functions that can handle varying amounts of label noise.

* Explore how performance varies by class. The evaluator generates per-class Lwlraps for each checkpoint and Lwlrap has been designed to allow aggregation of per-class Lwlraps into an overall Lwlrap. Use this to figure out how the noisy and curated data sets differ in terms of per-class Lwlraps, which could then let you use the best parts of each dataset to boost the overall Lwlrap.

## Contact

For general discussion of this task, please use the [Kaggle Discussion board](https://www.kaggle.com/c/freesound-audio-tagging-2019/discussion).

For specific issues with the code for this baseline system, please create an issue or a pull request on GitHub for the
[DCASE 2019 Baseline repo](https://github.com/DCASE-REPO/dcase2019_task2_baseline) and make sure to @-mention `plakal`.

## References

1. Howard, A. et. al., MobileNets: [Efficient Convolutional Neural Networks for Mobile Vision Applications](https://ai.googleblog.com/2017/06/mobilenets-open-source-models-for.html), https://arxiv.org/abs/1704.04861.

2. Hershey, S. et. al., [CNN Architectures for Large-Scale Audio Classification](https://ai.google/research/pubs/pub45611), ICASSP 2017.

3. Szegedy, C. et. al., [Rethinking the Inception Architecture for Computer Vision](https://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Szegedy_Rethinking_the_Inception_CVPR_2016_paper.pdf), CVPR 2016.

4. Wang, Y. et. al., [Trainable Frontend For Robust and Far-Field Keyword Spotting](https://ai.google/research/pubs/pub45911), ICASSP 2017.
