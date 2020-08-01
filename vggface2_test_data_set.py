import os
import cv2
import sys
import torch
from tqdm import tqdm
from PIL import Image

from torch.utils.data import Dataset


class VGGFace2TestDataset(Dataset):
    def __init__(self, root, transforms):
        self._root = root
        self.transforms = transforms
        self._classes, self._class_to_idx = self._find_classes()
        self._samples = self._make_dataset()
        self._loader = self._get_loader
        self._print_info()

    def _print_info(self):
        print('\n\tVGGFace2 custom data set info:'
              '\n\t\tRoot folder:         {}'
              '\n\t\tTest images:         {}'
              .format(len(self._samples), self._root))

    @staticmethod
    def _get_loader(path):
        return Image.fromarray(cv2.imread(path))

    def _find_classes(self):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(self._root) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(self._root) if os.path.isdir(os.path.join(self._root, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def _make_dataset(self):
        images = []
        dir = os.path.expanduser(self._root)
        progress_bar = tqdm(sorted(self._class_to_idx.keys()),
                            desc='Making for features extraction',
                            total=len(self._class_to_idx.keys()),
                            leave=False)
        for target in progress_bar:
            d = os.path.join(dir, target)
            if not os.path.isdir(d):
                continue
            for root, _, fnames in sorted(os.walk(d)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    item = (path, self._class_to_idx[target])
                    images.append(item)
            progress_bar.update(n=1)
        progress_bar.close()
        return images

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx):
        path, label = self._samples[idx]
        img = self._loader(path)
        if self.transforms:
            img = self.transforms(img)
        return img, path.split('/')[-2]+'/'+path.split('/')[-1]
