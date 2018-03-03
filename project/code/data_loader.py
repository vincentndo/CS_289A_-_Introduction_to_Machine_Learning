import os
import numpy as np
import random
import cv2

data_dir = "../data"

class data_loader(object):

    def __init__(self,classes,image_size):

        self.classes = classes
        self.num_class = len(self.classes)
        self.image_size = image_size
        self.load_data()


    def compute_feature(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        image = (image / 255.0) * 2.0 - 1.0
        return image


    def compute_label(self,label):
        one_hot = np.zeros(self.num_class)
        idx = self.classes.index(label)
        one_hot[idx] = 1.0
        return one_hot


    def load_data(self):

        temp_dir = os.path.join(data_dir, "tmp")
        resized_box_dir = os.path.join(temp_dir, "resized_box")
        resized_nobox_dir = os.path.join(temp_dir, "resized_nobox")

        box_list = []
        count = 0
        for filename in os.listdir(resized_box_dir):
            label = "package"
            label_vec = self.compute_label(label)

            file_path = os.path.join(resized_box_dir, filename)
            img = cv2.imread(file_path)
            features = self.compute_feature(img)
            box_list.append({'c_img': img, 'label': label_vec, 'features': features})
        assert len(box_list) == 22394, "Wrong appending box filenames"
        random.shuffle(box_list)

        nobox_list = []
        for filename in os.listdir(resized_nobox_dir):
            label = "no_package"
            label_vec = self.compute_label(label)

            file_path = os.path.join(resized_nobox_dir, filename)
            img = cv2.imread(file_path)
            features = self.compute_feature(img)
            nobox_list.append({'c_img': img, 'label': label_vec, 'features': features})
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
        self.train_data = train_data_list

        random.shuffle(eval_data_list)
        self.val_data = eval_data_list
