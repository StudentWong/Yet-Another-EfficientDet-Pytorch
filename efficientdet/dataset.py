import os
import torch
import numpy as np
import random
import math
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2


class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None, mosaic=True, augment=True):

        self.degrees = 1.98 * 3  # image rotation (+/- deg)
        self.translate = 0.05 * 3  # image translation (+/- fraction)
        self.scale = 0.05 * 3  # image scale (+/- gain)
        self.shear = 0.641 * 3
        self.mosaic = mosaic
        self.augment = augment
        self.root_dir = root_dir
        self.set_name = set
        # print(self.set_name)
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()
        # print(self.image_ids)

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        if self.mosaic and np.random.rand() < 0.5:
            img, annot = self.load_mosaic(idx)
        else:
            img = self.load_image(idx)
            annot = self.load_annotations(idx)

            if self.augment:
                # print(img.shape)
                #labels = annot[:, 0:4]
                img, annot = self.random_affine(img, annot, degrees=self.degrees,
                                                 translate=self.translate,
                                                 scale=self.scale,
                                                 shear=self.shear)
                # print(labels.shape)
                #annot[:, 0:4] = labels[:, 0:4]

        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)

        return sample

    def load_mosaic(self, index):
        # loads images in a mosaic

        labels4 = []
        #s = self.imgsize
        indices = [index] + [random.randint(0, len(self.image_ids) - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            # Load image

            img = self.load_image(index)
            # print(img.shape)
            h = img.shape[0]
            w = img.shape[1]
            xc, yc = [int(random.uniform(w * 0.5, h * 1.5)) for _ in range(2)]  # mosaic center x, y

            #h0, w0 h, w
            # print(img)
            annot = self.load_annotations(index)
            # place img in img4
            if i == 0:  # top left
                img4 = np.full((h * 2, w * 2, img.shape[2]), 114/225.0, dtype=np.float32)  # base image with 4 tiles
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, w * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(h * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, w * 2), min(h * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            # print([y1a, y2a, y1b, y2b])
            # print([x1a, x2a, x1b, x2b])
            # print("img4")
            # print(img4.shape)
            # print("img")
            # print(img.shape)
            # print(img4[y1a:y2a, x1a:x2a, :].shape)
            # print(img[y1b:y2b, x1b:x2b, :].shape)
            img4[y1a:y2a, x1a:x2a, :] = img[y1b:y2b, x1b:x2b, :]  # img4[ymin:ymax, xmin:xmax]
            # cv2.imshow("1", img4)
            # cv2.waitKey(0)
            padw = x1a - x1b
            padh = y1a - y1b

            # Labels
            # print
            #x = annot
            # labels_xywh = annot.copy()
            labels_out = annot.copy()
            # # print(annot)
            # labels_xywh[:, 0] = ((annot[:, 0] + annot[:, 2])/2) * w / w0
            # labels_xywh[:, 1] = ((annot[:, 1] + annot[:, 3])/2) * h / h0
            # labels_xywh[:, 2] = (annot[:, 2] - annot[:, 0]) * w / w0
            # labels_xywh[:, 3] = (annot[:, 3] - annot[:, 1]) * h / h0

            if annot.size > 0:  # Normalized xywh to pixel xyxy format
                labels_out[:, 0] = annot[:, 0] + padw
                labels_out[:, 1] = annot[:, 1] + padh
                labels_out[:, 2] = annot[:, 2] + padw
                labels_out[:, 3] = annot[:, 3] + padh

            labels4.append(labels_out)
        # print('Concat/cliplabels')
        # print(labels4)
        # Concat/clip labels
        if len(labels4):
            labels4 = np.concatenate(labels4, 0)
            # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
            # np.clip(labels4[:, :4], 0, 2 * s, out=labels4[:, :4])  # use with random_affine
            np.clip(labels4[:, 0], 0, 2 * w, out=labels4[:, 0])
            np.clip(labels4[:, 1], 0, 2 * h, out=labels4[:, 1])
            np.clip(labels4[:, 2], 0, 2 * w, out=labels4[:, 2])
            np.clip(labels4[:, 3], 0, 2 * h, out=labels4[:, 3])

        # Augment
        # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
        img4, labels4 = self.random_affine(img4, labels4,
                                           degrees=self.degrees,
                                           translate=self.translate,
                                           scale=self.scale,
                                           shear=self.shear,
                                           border=(-w // 2, -h // 2))  # border to remove
        # print('before return')
        # print(labels4)
        return img4, labels4


    def random_affine(self, img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=(0, 0)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        # targets = [cls, xyxy]

        height = img.shape[0] + border[1] * 2
        width = img.shape[1] + border[0] * 2

        # Rotation and Scale
        R = np.eye(3)
        a = random.uniform(-degrees, degrees)
        # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
        s = random.uniform(1 - scale, 1 + scale)
        # s = 2 ** random.uniform(-scale, scale)
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border[0]  # x translation (pixels)
        T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border[1]  # y translation (pixels)

        # Shear
        S = np.eye(3)
        S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
        S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

        # Combined rotation matrix
        M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
        if (border[0] != 0 and border[1] != 0) or (M != np.eye(3)).any():  # image changed
            img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

        # Transform label coordinates
        n = len(targets)
        # print(n)
        if n:
            # warp points
            xy = np.ones((n * 4, 3))
            xy[:, :2] = targets[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
            xy = (xy @ M.T)[:, :2].reshape(n, 8)

            # create new boxes
            x = xy[:, [0, 2, 4, 6]]
            y = xy[:, [1, 3, 5, 7]]
            xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

            # reject warped points outside of image
            xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
            xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
            w = xy[:, 2] - xy[:, 0]
            h = xy[:, 3] - xy[:, 1]
            area = w * h
            area0 = (targets[:, 2] - targets[:, 0]) * (targets[:, 3] - targets[:, 1])
            ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
            i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

            targets = targets[i]
            targets[:, 0:4] = xy[i]

        return img, targets

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        # print(image_info)
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        # print(path)
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        # print(coco_annotations)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            '''
            4分类模型中狗改成3
            '''
            # if annotation[0, 4] == 16.0:
            #     annotation[0, 4] = 3.0

            '''
            3分类模型去掉狗
            '''
            if annotation[0, 4] == 16.0:
                continue

            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
