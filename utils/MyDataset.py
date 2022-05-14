import os
import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, dataframe, label) -> None:
        self.dataframe = dataframe
        self.label = label
        
    def __len__(self) -> int:
        return len(self.dataframe)
    
    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        filepath = os.getcwd() + "/patch_images/"
        if self.label:
            _label = row[self.label]
        return (
            torchvision.transforms.functional.to_tensor(Image.open(filepath + row["ImageName"])),
            _label if self.label else -1
        )

    