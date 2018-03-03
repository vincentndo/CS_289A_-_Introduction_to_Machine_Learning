from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import random
import cv2
import IPython
import numpy as np



class Viz_Feat(object):


    def __init__(self,val_data,train_data, class_labels,sess):

        self.val_data = val_data
        self.train_data = train_data
        self.CLASS_LABELS = class_labels
        self.sess = sess





    def vizualize_features(self,net):

        images = [0,10,100]
        '''
        Compute the response map for the index images
        '''
        for i in images:
            features = np.array( [self.val_data[i]["features"] for _ in range(1) ] )
            feature_map_1 = self.sess.run(net.response_map_1, feed_dict={net.images: features})

            s = feature_map_1.shape[1]
            image = np.zeros( [s, s * 5, 3] )
            for j in range(5):
                image[:, j*s : (j+1)*s, :] = self.revert_image(feature_map_1[0, :, :, j])
                plt.imshow(image)
                plt.imsave("image_" + str(i) + "_response_map_1.png", image)

            feature_map_2 = self.sess.run(net.response_map_2, feed_dict={net.images: features})
            s = feature_map_2.shape[1]
            image = np.zeros( [s, s * 5, 3] )
            for j in range(5):
                image[:, j*s : (j+1)*s, :] = self.revert_image(feature_map_2[0, :, :, j])
                plt.imshow(image)
                plt.imsave("image_" + str(i) + "_response_map_2.png", image)





    def revert_image(self,img):
        '''
        Used to revert images back to a form that can be easily visualized
        '''

        img = (img+1.0)/2.0*255.0

        img = np.array(img,dtype=int)

        blank_img = np.zeros([img.shape[0],img.shape[1],3])

        blank_img[:,:,0] = img
        blank_img[:,:,1] = img
        blank_img[:,:,2] = img

        img = blank_img.astype("uint8")

        return img

        




