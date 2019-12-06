#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

def list_pictures(directory, ext='jpg|jpeg|bmp|png|ppm'):
    assert os.path.isdir(directory), 'dataset is not exists!{}'.format(directory)

    return sorted([os.path.join(root, f)
                   for root, _, files in os.walk(directory) for f in files
                   if re.match(r'([\w]+\.(?:' + ext + '))', f)])

if __name__ == '__main__':
    # f = '1.1'
    # if re.match(r'[\w]+\.+[\w]', f):
    #     print('success')
    #
    # f = '1.1'
    # if re.match(r'([\w]+\.(?:[\w]))', f):
    #     print('success')
    #
    ext = 'jpg'
    f_dir = 'D:/datasets/dogs-vs-cats/train'
    for root, _, files in os.walk(f_dir):
        for f in files:
            print(f)
            if re.match(r'([\w]+\.+[\w]+\.(?:' + ext + '))', f):
                print('success')
