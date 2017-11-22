import numpy as np
import numpy as np
import os
import matplotlib.image as mpimg
from Helpers import show_img
from ExpectationMaximization import ExpectationMaximization
import cv2

class ImageCompression(ExpectationMaximization):
    def compress(self, img, colours):
        try:
            reshaped = img.reshape(img.shape[0] * img.shape[1], 4)  # in my case it was 4, rgb + alpha
        except:
            reshaped = img.reshape(img.shape[0] * img.shape[1], 3)  # could be 3 as well, who cares

        self.cluster(reshaped, colours)

        for r_idx, row in enumerate(img):
            for c_idx, pixel in enumerate(row):
                img[r_idx][c_idx] = self.get_color(pixel)

        return img

    def get_color(self, pixel):
        distances_to_clusters = [self.get_distance_mahalanobis_to_cluster(pixel, i) for i in self.cluster_indexes]
        closest_cluster_idx = np.argmin(distances_to_clusters)

        return self.cluster_centers[closest_cluster_idx]

    # performance is important here, overwrite to remove math.sqrt
    def get_distance_mahalanobis_to_cluster(self, x, i_cluster):
        # the square root could be removed, but helps for better plotting
        return (x - self.cluster_centers[i_cluster]).dot(
            self.inv_covariances_per_cluster[i_cluster]).dot((x - self.cluster_centers[i_cluster]).T)




ic = ImageCompression()
img = mpimg.imread('./Dataset/image.png')
show_img(img)

compressed = ic.compress(img, 30)  # 30 colors
show_img(compressed, True) # True for save
