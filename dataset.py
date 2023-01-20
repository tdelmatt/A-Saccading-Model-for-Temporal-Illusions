import os
import os.path as osp
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as F


class TroxlerDataset(Dataset):

    def __init__(
        self,
        config,
        split='train',
        dataset_root='../TROXLER'
    ):
        super().__init__()
        dataset_root = osp.join(dataset_root, split)

        self.all_paths = list(map(lambda x: osp.join(dataset_root, x), os.listdir(dataset_root)))#[:2]
        self.grayscale = config.TASK.CHANNELS == 1

    def preprocess(self, img):
        img = F.resize(img, (64, 64))
        if self.grayscale:
            img = F.to_grayscale(img)
        # * Note, uniformity images are encoded 0-1, ensure this is true in other datasets
        return F.to_tensor(img) - 0.5 # 0-center.

    @staticmethod
    def unpreprocess(view):
        # For viz
        # * Note, uniformity images are encoded 0-1, ensure this is true in other datasets
        return torch.clamp(view + 0.5, 0, 1) # 0 - 1

    def __len__(self):
        return len(self.all_paths)

    def __getitem__(self, index):
        img = Image.open(self.all_paths[index])
        return self.preprocess(img)