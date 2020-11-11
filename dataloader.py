# +
import pydicom
import torch
from torch.utils import data
import torchvision
from torchvision import transforms
import numpy as np
import os

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

