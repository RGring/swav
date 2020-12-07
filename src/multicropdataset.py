# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import random
from logging import getLogger
import json

import matplotlib.pyplot as plt
import cv2
from PIL import ImageFilter
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import albumentations as A
from albumentations.pytorch import ToTensorV2

logger = getLogger()


class MultiCropDataset(datasets.ImageFolder):
    def __init__(
        self,
        args,
        size_dataset=-1,
        return_index=False,
    ):
        super(MultiCropDataset, self).__init__(args.data_path)
        assert len(args.size_crops) == len(args.nmb_crops)
        assert len(args.min_scale_crops) == len(args.nmb_crops)
        assert len(args.max_scale_crops) == len(args.nmb_crops)
        if size_dataset >= 0:
            self.samples = self.samples[:size_dataset]
        self.return_index = return_index

        # AUGMENTATIONS
        # Get datasets normalization values
        with open(f"{args.data_path}/norm.json", "r+") as f:
            norm = json.load(f)
        trans = []
        for i in range(len(args.size_crops)):
            randomresizedcrop = A.RandomResizedCrop(width=args.size_crops[i], height=args.size_crops[i], scale=(args.min_scale_crops[i], args.max_scale_crops[i]))
            trans.extend([A.Compose([
                A.Rotate(180, p=args.aug_rotation_prob),
                randomresizedcrop,
                A.HorizontalFlip(p=0.5),
                A.ColorJitter(args.aug_jitter_vector[0], args.aug_jitter_vector[1], args.aug_jitter_vector[2], args.aug_jitter_vector[3], p=args.aug_jitter_prob),
                A.RGBShift(r_shift_limit=args.aug_rgb_shift_vector[0], g_shift_limit=args.aug_rgb_shift_vector[1], b_shift_limit=args.aug_rgb_shift_vector[2], p=args.aug_rgb_shift_prob),
                A.ToGray(p=args.aug_grey_prob),
                A.GaussianBlur(blur_limit=(23, 23), sigma_limit=np.random.rand() * 1.9 + 0.1, p=args.aug_gaussian_blur_prob),
                A.Normalize(mean=norm["mean"], std=norm["std"]),
                ToTensorV2(),
            ])
            ] * args.nmb_crops[i])
        self.trans = trans

    def __getitem__(self, index):
        path, _ = self.samples[index]
        image = self.loader(path)
        # plt.imshow(image)
        # plt.show()
        # Generate different views for same image.
        multi_crops = list(map(lambda trans: trans(image=np.array(image))["image"], self.trans))
        # for img in multi_crops:
        #     plt.imshow(np.swapaxes(np.swapaxes(img.cpu().numpy(), 0, 1), 1, 2))
        #     plt.show()
        if self.return_index:
            return index, multi_crops
        return multi_crops


# class RandomGaussianBlur(object):
#     def __call__(self, img):
#         do_it = np.random.rand() > 0.5
#         if not do_it:
#             return img
#         sigma = np.random.rand() * 1.9 + 0.1
#         return cv2.GaussianBlur(np.asarray(img), (23, 23), sigma)
#
#
# class PILRandomGaussianBlur(object):
#     """
#     Apply Gaussian Blur to the PIL image. Take the radius and probability of
#     application as the parameter.
#     This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
#     """
#
#     def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
#         self.prob = p
#         self.radius_min = radius_min
#         self.radius_max = radius_max
#
#     def __call__(self, img):
#         do_it = np.random.rand() <= self.prob
#         if not do_it:
#             return img
#
#         return img.filter(
#             ImageFilter.GaussianBlur(
#                 radius=random.uniform(self.radius_min, self.radius_max)
#             )
#         )
#
#
# def get_color_distortion(s=1.0):
#     # s is the strength of color distortion1.
#     rnd_color_jitter = A.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s, p=0.8)
#     rnd_gray = A.ToGray(p=0.2)
#     color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
#     return color_distort
