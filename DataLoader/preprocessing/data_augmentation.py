import torch
import cv2
import numpy as np


class ConvertFromInts(object):
    """
    uint8 -> float32
    """
    def __call__(self, image, boxes=None, labels=None):
        return image.astype(np.float32), boxes, labels

class ToAbsoluteCoords(object):
    """
    バウンディングボックスの規格化をもとに戻す
    """
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] *= width
        boxes[:, 2] *= width
        boxes[:, 1] *= height
        boxes[:, 3] *= height

        return image, boxes, labels

class PhotometricDistort(object):
    """
    画像のデータオーグメンテーション
    """
    def __init__(self):
        self.distort_list = [
            RandomContrast(),
            ConvertColor(transform='HSV'),
            RandomSaturation(),
            RandomHue(),
            RandomBrightness(),
            ConvertColor(current='HSV', transform='BGR'),
            RandomContrast()
        ]
        self.rand_light_noise = RandomSwap()

    def __call__(self, image, boxes, labels):
        img = image.copy()
        
        # コントラスの変更を最初か最後どちらで行うかを決定する
        if np.random.randint(2):
            distort = Compose(self.distort_list[:-1])
        else:
            distort = Compose(self.distort_list[1:])
        img, boxes, labels = distort(img, boxes, labels)
        return self.rand_light_noise(img, boxes, labels)

class RandomContrast(object):
    """
    コントラストをランダムに変更する
    """
    def __init__(self, lower=0.5, upper=1.5):
        self.lower = lower
        self.upper = upper
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image *= np.random.uniform(self.lower, self.upper)
        return np.clip(image, 0, 255), boxes, labels

class ConvertColor(object):
    """
    BGR <-> HSV
    """
    def __init__(self, current='BGR', transform='HSV'):
        self.transform = transform  # 変換後の形式
        self.current = current  # 変換前の形式
    def __call__(self, image, boxes=None, labels=None):
        if self.current == 'BGR' and self.transform == 'HSV':
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_BGR2HSV)
        elif self.current == 'HSV' and self.transform == 'BGR':
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_HSV2BGR)
        else:
            raise NotImplementedError  # BGRかHSV以外はエラーとする
        return image.astype(np.float32), boxes, labels

class RandomSaturation(object):
    """
    ランダムに彩度を変更する
    """
    def __init__(self, lower=0, upper=1):
        self.lower = lower
        self.upper = upper
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 1] *= np.random.uniform(self.lower, self.upper)
        return image, boxes, labels

class RandomHue(object):
    """
    ランダムに色相を変更する
    """
    def __init__(self, delta=30.0):
        self.delta = delta
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            # 色相の要素に[-delta, delta]の範囲のランダムな値を足すことで色相を変換させる
            image[:, :, 0] += np.random.uniform(-self.delta, self.delta)

            # [0, 180]の範囲に収まるように調整する
            image[:, :, 0][image[:, :, 0] > 180.0] -= 180.0
            image[:, :, 0][image[:, :, 0] < 0.0] += 180.0
        return image, boxes, labels

class RandomBrightness(object):
    """
    ランダムに明度を変更する
    """
    def __init__(self, lower=0.8, upper=1.):
        self.lower = lower
        self.upper = upper
    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            image[:, :, 2] *= np.random.uniform(self.lower, self.upper)
        return image, boxes, labels

class RandomSwap(object):
    """
    ランダムにチャネルを入れ替える
    """
    def __init__(self):
        self.perms = ((0, 1, 2), (0, 2, 1),
                      (1, 0, 2), (1, 2, 0),
                      (2, 0, 1), (2, 1, 0))

    def __call__(self, image, boxes=None, labels=None):
        if np.random.randint(2):
            swap = self.perms[np.random.randint(len(self.perms))]
            image = image[:, :, swap]
        return image, boxes, labels

# ボックスも含めたデータオーグメンテーション

class Expand(object):
    """
    画像を縮小させて、周りを画像のBGR値の平均値でパディングする
    """
    def __init__(self, mean):
        """
        Parameters
        ---------
        mean : int
            BGR値の平均値
            (mean, mean, mean)のBGR値で画像の周りをパディングする
        """
        self.mean = mean

    def __call__(self, image, boxes, labels):
        if np.random.randint(2):
            return image, boxes, labels

        height, width, depth = image.shape
        ratio = np.random.uniform(1, 4)
        left = np.random.uniform(0, width*ratio - width)
        top = np.random.uniform(0, height*ratio - height)

        expand_image = np.zeros((int(height*ratio), int(width*ratio), depth), dtype=image.dtype)
        expand_image[:, :, :] = self.mean
        expand_image[int(top):int(top + height), int(left):int(left + width)] = image
        image = expand_image

        boxes = boxes.copy()
        boxes[:, :2] += (int(left), int(top))
        boxes[:, 2:] += (int(left), int(top))

        return image, boxes, labels

    
# 切り抜き
def intersect(box_a, box_b):
    """
    2つのボックスの共通面積を求める
    RandomSampleCropで使用する
    """
    max_xy = np.minimum(box_a[:, 2:], box_b[2:])
    min_xy = np.maximum(box_a[:, :2], box_b[:2])
    inter = np.clip((max_xy - min_xy), a_min=0, a_max=np.inf)
    return inter[:, 0] * inter[:, 1]

def jaccard_numpy(box_a, box_b):
    """
    Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.
    RandomSampleCropで使用する
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: 
            Multiple bounding boxes
            Shape: [num_boxes, 4]
        box_b: 
            Single bounding box
            Shape: [4]
    Return:
        jaccard overlap: list
        [box_a.shape[0], box_a.shape[1]]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))  # [A,B]
    area_b = ((box_b[2]-box_b[0]) *
              (box_b[3]-box_b[1]))  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]

class RandomSampleCrop(object):
    """
    画像を切り抜く
    前処理された画像サイズは必ずしも元のサイズには調整されない
    """
    def __init__(self):
        self.sample_options = (
            None,
            (0.1, None),
            (0.3, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        )

    def __call__(self, image, boxes=None, labels=None):
        height, width, _ = image.shape
        while True:
            mode = np.random.choice(self.sample_options)
            if mode is None:
                return image, boxes, labels

            min_iou, max_iou = mode
            if min_iou is None:
                min_iou = float('-inf')
            if max_iou is None:
                max_iou = float('inf')

            for _ in range(50):
                current_image = image

                w = np.random.uniform(0.3 * width, width)
                h = np.random.uniform(0.3 * height, height)

                if h / w < 0.5 or h / w > 2:
                    continue

                left = np.random.uniform(width - w)
                top = np.random.uniform(height - h)
                rect = np.array([int(left), int(top), int(left+w), int(top+h)])
                overlap = jaccard_numpy(boxes, rect)

                if overlap.min() < min_iou and max_iou < overlap.max():
                    continue

                current_image = current_image[rect[1]:rect[3], rect[0]:rect[2], :]

                centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                mask = m1 * m2

                if not mask.any():
                    continue

                current_boxes = boxes[mask, :].copy()
                current_labels = labels[mask]
                current_boxes[:, :2] = np.maximum(current_boxes[:, :2], rect[:2])
                current_boxes[:, :2] -= rect[:2]
                current_boxes[:, 2:] = np.minimum(current_boxes[:, 2:], rect[2:])
                current_boxes[:, 2:] -= rect[:2]

                return current_image, current_boxes, current_labels

class RandomMirror(object):
    """
    反転
    """
    def __call__(self, image, boxes, classes):
        _, width, _ = image.shape
        if np.random.randint(2):
            image = image[:, ::-1]
            boxes = boxes.copy()
            boxes[:, 0::2] = width - boxes[:, 2::-2]
        return image, boxes, classes

# 調整

class ToPercentCoords(object):
    """
    バウンディングボックスの規格化
    """
    def __call__(self, image, boxes=None, labels=None):
        height, width, channels = image.shape
        boxes[:, 0] /= width
        boxes[:, 2] /= width
        boxes[:, 1] /= height
        boxes[:, 3] /= height

        return image, boxes, labels

class Resize(object):
    """
    画像サイズの変形
    """
    def __init__(self, size=300):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size, self.size))
        return image, boxes, labels

class SubtractMeans(object):
    """
    BGR値の規格化
    """
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels

# まとめ

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels

class DataTransform:
    """
    学習用と検証用の画像とアノテーションを前処理する
    data_transform : dict
        {'train' : 学習用の前処理クラス, 'val' : 検証用の前処理クラス}
    """
    def __init__(self, transform_list_train, transform_list_val):
        """
        Parameters
        ----------
        transform_list_train : list
            学習用の様々な前処理クラスを格納しているリスト
            [Resize(), ..., Augmentation()]
        transform_list_train : list
            検証用の様々な前処理クラスを格納しているリスト
            [Resize(), ...,]
        """
        self.data_transform = {"train": Compose(transform_list_train),
                               "val": Compose(transform_list_val)}

    def __call__(self, img, boxes, labels, phase):
        return self.data_transform[phase](img, boxes, labels)