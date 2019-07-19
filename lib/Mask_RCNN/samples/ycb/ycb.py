import os
import sys
import math
import random
import numpy as np
import cv2
import fnmatch
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
dataset_folder = 'ycb/data'
class YCBDataset(utils.Dataset):
    """

    """
    def load_ycb(self, count_images = 0):
        """Generate the requested number of images from the dataset.
        count: number of images to generate. If set to 0, get all images
        TODO-Add more functionality here later
        """
        self.dataset_folder = 'ycb/data'
        root_folders = os.listdir(dataset_folder)
        i = 0
        for folder in root_folders:
            files = os.listdir(dataset_folder+'/'+ folder)
            for file in files:
                if fnmatch.fnmatch(file, '*color*'):
                    i+=1
                    if i > count_images:
                        break
                    image_file_id = file.split('-')[0]
                    self.add_image("ycb", image_id = i, path = folder + '/' + image_file_id) 
            if i > count_images:
                break
         
    def load_image(self, image_id):
        """Load the image from the file with the image id
        """
        image = cv2.imread(dataset_folder+'/'+self.image_info[image_id]['path']+'-color.png')
        print(dataset_folder+'/'+self.image_info[image_id]['path']+'-color.png')
        return image
        
    def load_mask(self, image_id):
        """Generate instance masks for the image of the given image ID.
        """
        mask_image = cv2.imread(dataset_folder+'/'+self.image_info[1]['path']+'-label.png')[:,:,0]  
        classes = np.unique(mask_image)
        classes = np.delete(classes,0)
        mask = np.zeros([480, 640,len(classes)], dtype=np.uint8)
        i = 0
        for obj_class in classes :
            if obj_class> 0
            mask[:,:,i] = mask_image == obj_class
            i += 1
        return mask,classes

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)