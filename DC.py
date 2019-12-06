from mxnet.gluon.data import Dataset
from transform import transform
from mxnet import image
import os.path as osp
import os
import re

class DC(Dataset):
    def __init__(self, args, mode):
        super(DC, self).__init__()
        self.args = args
        self.animal_label_dict = {}
        self.animal_label_num = 0.0
        self.animal_label_sum_dict = {}
        if mode == 'train':
            self.data_dir = args.train_dir
        elif mode == 'test':
            self.data_dir = args.test_dir
        self.imgs = [path for path in self.list_pictures()]
        self.imgs_class = [self.label2id(self.imgs[i]) for i in range(len(self.imgs))]

    def __getitem__(self, idx):
        img = self.imgs[idx]
        label = self.imgs_class[idx]
        item = image.imread(img)
        return transform(item, args=self.args), label

    def __len__(self):
        return len(self.imgs)

    def list_pictures(self):
        ext = 'jpg|jpeg|bmp|png|ppm'
        data_dir = self.data_dir
        assert osp.isdir(data_dir), 'dataset is not exists!{}'.format(data_dir)
        return sorted([osp.join(root, f)
                       for root, _, files in os.walk(data_dir) for f in files
                       if re.match(r'([\w]+\.+[\w]+\.(?:' + ext + '))', f)])


    def label2id(self, img):
        animal_label = img.replace('\\', '/').split('/')[-1].split('.')[0]
        if animal_label not in self.animal_label_dict:
            self.animal_label_dict[animal_label] = self.animal_label_num
            self.animal_label_num += 1.0
            self.animal_label_sum_dict[animal_label] = 0.0
        self.animal_label_sum_dict[animal_label] += 1.0
        return self.animal_label_dict[animal_label]


