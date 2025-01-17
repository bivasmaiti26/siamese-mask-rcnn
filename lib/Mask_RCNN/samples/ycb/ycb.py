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
class YCBDataset(utils.Dataset):
    """

    """
    def load_ycb(self, count_images = 0):
        """Generate the requested number of images from the dataset.
        count: number of images to generate. If set to 0, get all images
        TODO-Add more functionality here later
        """
        if self.type == 'train':
            self.dataset_folder = 'ycb_train' 
        else:
            self.dataset_folder = 'ycb_val'
        for i in range(21):
            self.add_class('ycb',i,chr(65+i))
        root_folders = os.listdir(self.dataset_folder)
        i = -1
        for folder in root_folders:
            files = os.listdir(self.dataset_folder+'/'+ folder)
            for file in files:
                if fnmatch.fnmatch(file, '*color*'):
                    i+=1
                    if i > count_images and count_images>0:
                        break
                    image_file_id = file.split('-')[0]
                    self.add_image("ycb", image_id = i, path = folder + '/' + image_file_id) 
                    
            if i > count_images and count_images>0:
                break
         
    def load_image(self, image_id):
        """Load the image from the file with the image id
        """
        image = cv2.imread(self.dataset_folder+'/'+self.image_info[image_id]['path']+'-color.png')
        return image
        
    def load_mask(self, image_id):
        """Generate instance masks for the image of the given image ID.
        """
        
        mask_image = cv2.imread(self.dataset_folder+'/'+self.image_info[image_id]['path']+'-label.png')[:,:,0]  
        classes = np.unique(mask_image)
        classes = np.delete(classes,0)
        mask = np.zeros([480, 640,len(classes)], dtype=np.uint8)
        i = 0
        for obj_class in classes :
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
