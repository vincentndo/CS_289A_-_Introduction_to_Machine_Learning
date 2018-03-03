# Copyright 2017 AxleHire. All Rights Reserved.
#
# ==============================================================================

import numpy as np
import random
import os
import utils

""" temporary Script to convert 32x32 box images into a single binary data file
    Usage:
        $ python ./tmp_make_train_data
        output: none

        side effect: in the binary data directory exists data_batch.bin
"""


if __name__ == "__main__":

    data_dir = "/home/taflab/data/"
    raw_box_dir = os.path.join(data_dir, "raw_box")
    raw_nobox_dir = os.path.join(data_dir, "raw_nobox")
    temp_dir = os.path.join(data_dir, "tmp")
    resized_box_dir = os.path.join(temp_dir, "resized_box")
    resized_nobox_dir = os.path.join(temp_dir, "resized_nobox")

    # print("Resizing box images ...")
    # assert 20394 == utils.resize_raw_img(raw_box_dir, resized_box_dir), "Wrong count in raw_box"
    # print("Resizing nobox images ...")
    # assert 6100 == utils.resize_raw_img(raw_nobox_dir, resized_nobox_dir), "Wrong count in raw_nobox"

    print("Converting 32x32 box images into a single binary data file ...")

    bin_train_dir = "./tmp/train/bin_train"
    bin_eval_dir = "./tmp/eval/bin_eval"

    box_list = []
    for filename in os.listdir(resized_box_dir):
        file_path = os.path.join(resized_box_dir, filename)
        box_list.append(utils.image_to_binary(file_path, [1]))
    assert len(box_list) == 22394, "Wrong appending box filenames"
    random.shuffle(box_list)

    print("Converting 32x32 nobox images into a single binary data file ...")

    nobox_list = []
    for filename in os.listdir(resized_nobox_dir):
        file_path = os.path.join(resized_nobox_dir, filename)
        nobox_list.append(utils.image_to_binary(file_path, [0]))
    assert len(nobox_list) == 6100, "Wrong appending nobox filenames"
    random.shuffle(nobox_list)

    train_data_list = []
    eval_data_list = []
    for i in range(len(box_list)):
        if i < 18000:
            train_data_list.append(box_list.pop(0))
        else:
            eval_data_list.append(box_list.pop(0))
    assert len(eval_data_list) == 4394, "Wrong eval count"

    for i in range(len(nobox_list)):
        if i < 5000:
            train_data_list.append(nobox_list.pop(0))
        else:
            eval_data_list.append(nobox_list.pop(0))
    assert len(train_data_list) == 23000, "Wrong total train count"
    assert len(eval_data_list) == 5494, "Wrong total eval count"

    random.shuffle(train_data_list)
    train_data_array = np.concatenate(tuple(train_data_list), axis=0)
    train_data_array.tofile(os.path.join(bin_train_dir, "data_batch_1.bin"))

    random.shuffle(eval_data_list)
    eval_data_array = np.concatenate(tuple(eval_data_list), axis=0)
    eval_data_array.tofile(os.path.join(bin_eval_dir, "test_batch.bin"))
