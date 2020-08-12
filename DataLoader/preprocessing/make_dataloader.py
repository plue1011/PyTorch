import cv2
import xml.etree.ElementTree as ET
import numpy as np
import torch
from torch.utils.data import Dataset

def make_data_path_list(root_path, train_val, img_ext='jpg'):
    """
    Parameters
    ----------
    root_path : str
        データセットの親パス
    train_val : str
        'train' or 'val'
    img_ext : str
        画像データの拡張子
        
    Returns
    -------
    img_path_list : list of str
        画像データのパスリスト
        ['path/img1.jpg', 'path/img2.jpg']
    anno_path_list : list of str
        アノテーションデータのパスリスト
        ['path/img1.xml', 'path/img2.xml']
    """
    # 学習データの画像ファイルとアノテーションファイルへのパスリストを作成
    img_path_list = []
    anno_path_list = []

    id_names = root_path + 'ImageSets/Main/' + train_val + '.txt'

    for line in open(id_names):
        file_id = line.strip()  # 空白スペースと改行を除去
        img_path = root_path + 'JPEGImages/' + file_id + '.' + img_ext
        anno_path = root_path + 'Annotations/' + file_id + '.xml'
        img_path_list.append(img_path)
        anno_path_list.append(anno_path)

    return img_path_list, anno_path_list

class Anno_xml2list(object):
    """
    1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

    Attributes
    ----------
    classes : リスト
        データセットで登場するクラス名を格納したリスト
    """

    # 矩形のクラスリストを入力する
    def __init__(self, classes):
        self.classes = classes

    def __call__(self, xml_path, width, height):
        """
        1枚の画像に対する「XML形式のアノテーションデータ」を、画像サイズで規格化してからリスト形式に変換する。

        Parameters
        ----------
        xml_path : str
            xmlファイルへのパス。
        width : int
            対象画像の幅。
        height : int
            対象画像の高さ。

        Returns
        -------
        ret : [[xmin, ymin, xmax, ymax, label_ind], ... ]
            物体のアノテーションデータを格納したリスト。画像内に存在する物体数分のだけ要素を持つ。
        """

        # 画像内の全ての物体のアノテーションをこのリストに格納
        ret = []

        # xmlファイルを読み込む
        xml = ET.parse(xml_path).getroot()

        # 画像内にある物体（object）の数だけループする
        for obj in xml.iter('object'):
            # 1つの物体に対するアノテーションを格納するリスト
            bndbox = []

            name = obj.find('name').text.strip()  # 物体名
            bbox = obj.find('bndbox')  # バウンディングボックスの情報

            # アノテーションの xmin, ymin, xmax, ymaxを取得し、0～1に規格化
            pts = ['xmin', 'ymin', 'xmax', 'ymax']

            for pt in (pts):
                # damagedatasetは原点が(1,1)なので1を引き算して（0, 0）
                cur_pixel = int(bbox.find(pt).text) - 1

                # 幅、高さで規格化
                if pt == 'xmin' or pt == 'xmax':  # x方向のときは幅で割算
                    cur_pixel /= width
                else:  # y方向のときは高さで割算
                    cur_pixel /= height

                bndbox.append(cur_pixel)

            # アノテーションのクラス名のindexを取得して追加
            label_idx = self.classes.index(name)
            bndbox.append(label_idx)

            # retに[xmin, ymin, xmax, ymax, label_ind]を追加
            ret += [bndbox]

        return np.array(ret)  # [[xmin, ymin, xmax, ymax, label_idx], ... ]
    
class ImageDetectionDataset(Dataset):
    """
    物体検出用のDatasetを作成するクラス
    PyTorchのDatasetクラスを継承

    Attributes
    ----------
    img_list : list
        画像のパスを格納したリスト
    anno_list : list
        アノテーションへのパスを格納したリスト
    transform : object
        前処理クラスのインスタンス
    transform_anno : object
        xmlのアノテーションをリストに変換するインスタンス
    """

    def __init__(self, img_path_list, anno_path_list, transform, transform_anno, phase):
        """
        Parameters
        ----------
        img_list : list
            画像のパスを格納したリスト
        anno_list : list
            アノテーションへのパスを格納したリスト
        transform : object
            前処理クラスのインスタンス
        transform_anno : object
            xmlのアノテーションをリストに変換するインスタンス
        phase : str
            'train' or 'val'
        """
        self.img_path_list = img_path_list
        self.anno_path_list = anno_path_list
        self.transform = transform
        self.transform_anno = transform_anno
        self.phase = phase

    def __len__(self):
        """
        データセット中の画像の枚数を返す
        """
        return len(self.img_path_list)

    def __getitem__(self, index):
        """
        前処理をした画像のテンソル形式のデータとアノテーションを出力
        
        Parameters
        ----------
        index : int
            データセットの何番目のデータを出力するか
        
        Returns
        -------
        img : tensor
            画像データ
            torch.Size([channls, h, w])
        gt : array
            バウンディングボックスの情報(ground truth)
            [xmin, ymin, xmax, ymax, class]
        """
        img, gt, h, w = self.pull_item(index)
        return img, gt

    def pull_item(self, index):
        """
        前処理をした画像のテンソル形式のデータ、アノテーション、画像の高さ、幅を取得する
        
        Returns
        -------
        img : tensor
            画像データ
            torch.Size([channls, h, w])
        gt : array
            バウンディングボックスの情報(ground truth)
            [xmin, ymin, xmax, ymax, class]
        """

        # 画像読み込み
        image_file_path = self.img_path_list[index]
        img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
        height, width, channels = img.shape  # 画像のサイズを取得

        # xml形式のアノテーション情報をリストに
        anno_file_path = self.anno_path_list[index]
        anno_list = self.transform_anno(anno_file_path, width, height)

        # 前処理を実施
        img, boxes, labels = self.transform(img, anno_list[:, :4], anno_list[:, 4], self.phase)

        # 色チャネルの順番がBGRになっているので、RGBに順番変更
        # さらに（高さ、幅、色チャネル）の順を（色チャネル、高さ、幅）に変換
        img = torch.from_numpy(img[:, :, (2, 1, 0)]).permute(2, 0, 1)

        # BBoxとラベルをセットにしたnp.arrayを作成、変数名「gt」はground truth（答え）の略称
        gt = np.hstack((boxes, np.expand_dims(labels, axis=1)))

        return img, gt, height, width
    
def od_collate_fn(batch):
    targets = []
    imgs = []
    for sample in batch:
        imgs.append(sample[0])  # sample[0]：画像データ
        # アノテーションはnumpyの配列で入力されるためtensorに変換している
        targets.append(torch.FloatTensor(sample[1]))  # sample[1]：アノテーションgt

    # imgsはミニバッチサイズのリストになっている
    # imgsの要素はtorch.Size([3, 300, 300])
      # 3：RGB
      #300：画像の縦サイズ
      #300：画像の横サイズ
    # imgsをtorch.Size([batch_num, 3, 300, 300])のテンソルに変換する
    imgs = torch.stack(imgs, dim=0)

    # targetsはアノテーションデータの正解であるgtのリスト
    # リストのサイズはミニバッチサイズ
    # リストtargetsの要素は [n, 5] となっている
    # nは画像ごとに異なり、画像内にある物体の数となる
    # 5：[xmin, ymin, xmax, ymax, class_index] 

