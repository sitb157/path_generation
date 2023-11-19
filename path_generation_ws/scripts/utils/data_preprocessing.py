import pykitti
import numpy as np
import cv2
import os

class Data_Preprocessing:
    """
        @brief Class for preprocessing data.
    """
    def __init__(self, base_dir, date, drive, save_dir):

        # The 'frames' argument is optional - default: None, which loads the whole dataset.
        # Calibration, timestamps, and IMU data are read automatically. 
        # Camera and velodyne data are available via properties that create generators
        # when accessed, or through getter methods that provide random access.
        self.data = pykitti.raw(base_dir, date, drive)
        self.save_dir = save_dir

    def lidar_Preprocessing(self, idx, datas):
        lidar_z_to_image = np.zeros((600, 600))
        lidar_r_to_image = np.zeros((600, 600))

        bias_x = 299
        bias_y = 299

        for data in datas: 
            x = int(data[0] * 10) + bias_x
            y = int(data[1] * 10) + bias_y
            z = int((data[2] + 1.73) * 70) # Velodyne height is 1.73 from ground
            r = int(data[3] * 70) # reflectivity
            if (0 <= x < 600) and (0 <= y < 600):
                lidar_z_to_image[x][y] = max(lidar_z_to_image[x][y], z)
                lidar_r_to_image[x][y] = max(lidar_r_to_image[x][y], r)

        file_name_z = os.path.join(self.save_dir, "lidar_z", f"{idx}.jpg")
        file_name_r = os.path.join(self.save_dir, "lidar_r", f"{idx}.jpg")

        cv2.imwrite(file_name_z, lidar_z_to_image) 
        cv2.imwrite(file_name_r, lidar_r_to_image) 

    def run(self):
        for idx, (time, oxt, velo) in enumerate(zip(self.data.timestamps, self.data.oxts, self.data.velo)):
            self.lidar_Preprocessing(idx, velo)

if __name__ == "__main__":
    data_preprocessing = Data_Preprocessing("/root/datas", "2011_09_26", "0001", "/root/test_datas")
    data_preprocessing.run()
