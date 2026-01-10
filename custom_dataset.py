import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, params, is_train, inputsize, transform=None, seg_mode=None):
        self.params = params
        self.root = params.train_set if is_train else params.val_set
        self.img_path = os.path.join(self.root, 'images')
        self.label_path = os.path.join(self.root, 'labels')
        self.mask_path = os.path.join(self.root, 'segmentation')
        self.img_list = os.listdir(self.img_path)
        self.transform = transform
        self.inputsize = inputsize

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img_name = self.img_list[index]
        base_name = os.path.splitext(img_name)[0]

        # 이미지 로드
        img = cv2.imread(os.path.join(self.img_path, img_name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 라벨 로드 (Bbox)
        label_file = os.path.join(self.label_path, base_name + '.txt')
        if os.path.exists(label_file):
            annot = np.loadtxt(label_file).reshape(-1, 5)
        else:
            annot = np.zeros((0, 5))

        # 마스크 로드 (Segmentation)
        mask = cv2.imread(os.path.join(self.mask_path, base_name + '.png'), 0)

        # 전처리 (Resize 등)
        img = cv2.resize(img, (self.inputsize[1], self.inputsize[0]))
        mask = cv2.resize(mask, (self.inputsize[1], self.inputsize[0]), interpolation=cv2.INTER_NEAREST)

        if self.transform:
            img = self.transform(img)

        return {'img': img, 'annot': torch.from_numpy(annot), 'segmentation': torch.from_numpy(mask).long()}

    @staticmethod
    def collate_fn(batch):
        img = torch.stack([x['img'] for x in batch])
        annot = [x['annot'] for x in batch]
        seg = torch.stack([x['segmentation'] for x in batch])

        # Annotations padding (YOLO 형식 대응)
        max_annots = max(len(x) for x in annot)
        if max_annots > 0:
            annot_padded = torch.ones((len(annot), max_annots, 5)) * -1
            for i, ann in enumerate(annot):
                if len(ann) > 0:
                    annot_padded[i, :len(ann), :] = ann
        else:
            annot_padded = torch.ones((len(annot), 1, 5)) * -1

        return {'img': img, 'annot': annot_padded, 'segmentation': seg}