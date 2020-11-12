# +
import pydicom
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import numpy as np
import os
import PIL
import matplotlib.pyplot as plt

class DCMDatasetLoader(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.classes, self.class_to_idx = self._find_classes(self.root)

        print(self.classes)
        self.img_names = []
        self.labels = []
        for c in self.classes:
            if os.path.isdir(os.path.join(self.root, c)):
                path =os.path.join(self.root, c)
                self.img_names += os.listdir(path)
                self.labels += ([c]*len(os.listdir(path)))
        
        print("> Found %d images..." % (len(self.img_names)))
        
        self.trans = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((512,512)),
                        transforms.Normalize(mean=[128], std=[128])
                    ])

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        img = pydicom.dcmread(os.path.join(self.root, self.labels[idx],self.img_names[idx]))
        img = img.pixel_array.astype(np.float32)
        img = self.trans(img)
        
        return img, self.class_to_idx[self.labels[idx]]
    
    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx


# -

class DCMDatasetLoader_3windows(data.Dataset):
    def __init__(self, root):
        self.root = root
        self.classes, self.class_to_idx = self._find_classes(self.root)

        print(self.classes)
        self.img_names = []
        self.labels = []
        for c in self.classes:
            if os.path.isdir(os.path.join(self.root, c)):
                path =os.path.join(self.root, c)
                self.img_names += os.listdir(path)
                self.labels += ([c]*len(os.listdir(path)))
        
        print("> Found %d images..." % (len(self.img_names)))
        
        self.trans = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((512,512)),
                        #transforms.Normalize(mean=[128], std=[128])
                    ])

    def __len__(self):
        return len(self.img_names)
    
    def __getitem__(self, idx):
        origin_img = pydicom.dcmread(os.path.join(self.root, self.labels[idx],self.img_names[idx]))
        img = origin_img.pixel_array
        window_center , window_width, intercept, slope = self.get_windowing(origin_img)

        # Combine 3 channels
        img_3_windows = []
        
        img_3_windows.append(self.window_image(img, 40, 80, intercept, slope)) # Brain window (40,80)
        img_3_windows.append(self.window_image(img, 80, 200, intercept, slope)) # Subdural window (80,200)
        img_3_windows.append(self.window_image(img, 600, 2800, intercept, slope)) # Bone window (600, 2800)
        img = np.array(img_3_windows, dtype=np.float32)
        
        img = self.trans(img.transpose(1,2,0))    
        
        return img, self.class_to_idx[self.labels[idx]]
    
    def _find_classes(self, dir: str):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
    
    def get_first_of_dicom_field_as_int(self, x):
        #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
        if type(x) == pydicom.multival.MultiValue:
            return int(x[0])
        else:
            return int(x)

    def get_windowing(self, data):
        dicom_fields = [data[('0028','1050')].value, #window center
                        data[('0028','1051')].value, #window width
                        data[('0028','1052')].value, #intercept
                        data[('0028','1053')].value] #slope
        return [self.get_first_of_dicom_field_as_int(x) for x in dicom_fields]
    
    def window_image(self, img, window_center,window_width, intercept, slope):

        img = (img*slope +intercept)
        img_min = window_center - window_width//2
        img_max = window_center + window_width//2
        img[img<img_min] = img_min
        img[img>img_max] = img_max
        return img
    
    def normalize_minmax(self, img):
        mi, ma = img.min(), img.max()
        return (img - mi) / (ma - mi)
