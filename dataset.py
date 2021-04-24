import torch
from torch.utils.data import Dataset
import cv2
from utils import to_relative_coordinates

class NeurusCatnDogDataset(Dataset):
    
    def __init__(self, jpg_paths, label_paths, transforms):
        self.jpg_paths = jpg_paths
        self.label_paths = label_paths
        self.transforms = transforms
        
    def __len__(self):
        return len(self.label_paths)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.jpg_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        labels = open(self.label_paths[idx]).read().split(' ')
        class_label = int(labels[0]) - 1
        coords_labels = [int(coord) for coord in labels[1:]]
        coords_labels = to_relative_coordinates(coords_labels, image)
        transformed = self.transforms(image=image, bboxes=[coords_labels + [class_label]])
        image = transformed["image"]
        bboxes = torch.Tensor(transformed["bboxes"][0])
        return image, bboxes
