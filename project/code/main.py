# Copyright 2017 AxleHire. All Rights Reserved.
#
# ==============================================================================

""" Main script to run ImageRecognizer
    Usage: either
        $ python ./main.py full_path_to_image_file
        $ python ./main.py full_path_to_image_folder
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from argparse import ArgumentParser
import tensorflow as tf
import utils
import eval

TMP_RESIZED_TEST_DIR = "./tmp/test/resized_test"
TMP_BIN_TEST_DIR = "./tmp/test/bin_test"


def parse_arguments():

    parser = ArgumentParser()
    parser.add_argument("filename")
    args = parser.parse_args()
    return args.filename


def main(arg=None):

    print("Starting ...")

    try:
        filename = parse_arguments()
    except:
        print("Need a full path to image file or folder")
        exit(1)

    if os.path.isfile(filename):
        image_dir = os.path.dirname(filename)
    else:
        image_dir = filename

    num_examples = utils.resize_raw_img(image_dir, TMP_RESIZED_TEST_DIR)
    utils.convert_resized_img_into_binary(TMP_RESIZED_TEST_DIR, TMP_BIN_TEST_DIR)
    eval.evaluate(num_examples, "test", TMP_BIN_TEST_DIR)


if __name__ == '__main__':
    tf.app.run()
