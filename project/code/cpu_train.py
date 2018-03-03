# Copyright 2017 AxleHire. All Rights Reserved.
#
# ==============================================================================

""" A binary to train the CNN model using a single GPU.

    Accuracy:
        cpu_train.py achieves ... accuracy after ... steps (... epochs of
        data) as judged by eval.py.

    Speed: With batch_size 128...

    Usage:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import input
import recognizer

TMP_BIN_TRAIN_DIR = "tmp/train/bin_train"
CKPT_DIR = "./ckpt"                             # Directory where to write event logs and checkpoint
MAX_STEPS = 1000                                # Number of batches to run
LOG_DEVICE_PLACEMENT = False                    # Whether to log device placement
LOG_FREQUENCY = 10                              # How often to log results to the console


def train():
  """Train CNN model for a number of steps."""
  with tf.Graph().as_default():
    global_step = tf.contrib.framework.get_or_create_global_step()

    # Get images and labels for the CNN model.
    # Force input pipeline to CPU:0 to avoid operations sometimes ending up on
    # GPU and resulting in a slow down.
    with tf.device('/cpu:0'):
      images, labels = input.distorted_inputs(TMP_BIN_TRAIN_DIR)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = recognizer.inference("train", images)

    # Calculate loss.
    loss = recognizer.loss(logits, labels)

    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = recognizer.train(loss, global_step)

    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % LOG_FREQUENCY == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = LOG_FREQUENCY * input.TRAIN_EVAL_BATCH_SIZE / duration
          sec_per_batch = float(duration / LOG_FREQUENCY)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), self._step, loss_value,
                               examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=CKPT_DIR,
        hooks=[tf.train.StopAtStepHook(last_step=MAX_STEPS),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=LOG_DEVICE_PLACEMENT)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)


def main(argv=None):  # pylint: disable=unused-argument

  if tf.gfile.Exists(CKPT_DIR):
    tf.gfile.DeleteRecursively(CKPT_DIR)
  tf.gfile.MakeDirs(CKPT_DIR)
  train()


if __name__ == '__main__':
  tf.app.run()
