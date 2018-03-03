# Copyright 2017 AxleHire. All Rights Reserved.
#
# ==============================================================================

""" Evaluate the CNN model.

    Accuracy:

    Speed:

    Usage:
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time

import numpy as np
import tensorflow as tf

import input
import recognizer

TMP_BIN_EVAL_DIR = "./tmp/eval/bin_eval"    # Directory where to find binary evaluation data set
TMP_OUT_EVAL_DIR = "./tmp/eval/out_eval"    # Directory where to write event logs.
MODE = "eval"                               # Mode to run eval or test
CHECKPOINT_DIR = "./ckpt"                   # Directory where to read model checkpoints
EVAL_INTERVAL_SECS = 60 * 5                 # How often to run the eval
NUM_EXAMPLES = 484                          # Number of examples to run
RUN_ONCE = True                             # Whether to run eval only once


def eval_once(num_examples, mode, saver, labels, logits, summary_writer, summary_op):
  """Run Eval once.

  Args:
    saver: Saver.
    summary_writer: Summary writer.
    summary_op: Summary op.
  """
  with tf.Session() as sess:
    ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(sess, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/ckpt/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
      return

    # Start the queue runners.
    coord = tf.train.Coordinator()
    try:
      threads = []
      for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
        threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                         start=True))
      true_count, step = 0, 0
      if mode == MODE:

        num_iter = int(math.ceil(NUM_EXAMPLES / input.TRAIN_EVAL_BATCH_SIZE))
        total_sample_count = num_iter * input.TRAIN_EVAL_BATCH_SIZE

        while step < num_iter and not coord.should_stop():
          top_k_op = tf.nn.in_top_k(logits, labels, 1)
          predictions = sess.run([top_k_op])
          true_count += np.sum(predictions)
          step += 1

        # Compute precision @ 1.
        precision = true_count / total_sample_count
        print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        summary = tf.Summary()
        summary.ParseFromString(sess.run(summary_op))
        summary.value.add(tag='Precision @ 1', simple_value=precision)
        summary_writer.add_summary(summary, global_step)

      else:

        num_iter = int(math.ceil(num_examples / input.TEST_BATCH_SIZE))
        total_sample_count = num_iter * input.TEST_BATCH_SIZE
        fid = open("./tmp/test/bin_test/filenames.txt", "r")

        while step < num_iter and not coord.should_stop():
          predictions = sess.run(logits)
          print(fid.readline().strip() + ": " + str(predictions[0]))
          for e in predictions:
              if e[0] > e[1]:
                  print("  No package")
              else:
                  print("  Package")
                  true_count += 1
          step += 1

        print("Total box: %d/%d" % (true_count, total_sample_count))
        fid.close()

    except Exception as e:  # pylint: disable=broad-except
      coord.request_stop(e)

    coord.request_stop()
    coord.join(threads, stop_grace_period_secs=10)


def evaluate(num_examples=NUM_EXAMPLES, mode=MODE, data_dir=TMP_BIN_EVAL_DIR):
  """Eval the CNN model for a number of steps."""
  with tf.Graph().as_default() as g:
    # Get images and labels for the CNN model.
    images, labels = input.inputs(mode, data_dir)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = recognizer.inference(mode, images)

    # Restore the moving average version of the learned variables for eval.
    variable_averages = tf.train.ExponentialMovingAverage(
        recognizer.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(TMP_OUT_EVAL_DIR, g)

    while True:
      eval_once(num_examples, mode, saver, labels, logits, summary_writer, summary_op)
      if RUN_ONCE:
        break
      time.sleep(EVAL_INTERVAL_SECS)


def main(argv=None):  # pylint: disable=unused-argument

  if tf.gfile.Exists(TMP_OUT_EVAL_DIR):
    tf.gfile.DeleteRecursively(TMP_OUT_EVAL_DIR)
  tf.gfile.MakeDirs(TMP_OUT_EVAL_DIR)
  evaluate()


if __name__ == '__main__':
  tf.app.run()
