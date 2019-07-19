import os
import sys
import math
import random
import numpy as np
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils

class YCBDataset(utils.Dataset):
     """Generates the YCB dataset. The Dataset needs to be present in the root folder of 
     the repository. For example: /root_dir/YCB_Video_Dataset/data
    """
    def load_ycb(self, count):
        """Generate the requested number of images from the dataset.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        pass
     def load_image(self, image_id):
         """Load the image from the file with the image id
        """
        pass
    def load_mask(self, image_id):
        """Generate instance masks for the image of the given image ID.
        """
        pass
     def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)