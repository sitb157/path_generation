import pykitti

class Data_Preprocessing:
    """
        @brief Class for preprocessing data.
    """
    def __init__(self, basedir, date, drive):

        # The 'frames' argument is optional - default: None, which loads the whole dataset.
        # Calibration, timestamps, and IMU data are read automatically. 
        # Camera and velodyne data are available via properties that create generators
        # when accessed, or through getter methods that provide random access.
        self.data = pykitti.raw(basedir, date, drive, frames=range(0, 50, 5))
        print(self.data.oxts)


if __name__ == "__main__":
    data_preprocessing = Data_Preprocessing("/root/datas", "2011_09_26", "0001")
