#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : show_bboxes.py
#   Author      : YunYang1994
#   Created date: 2019-05-29 01:18:24
#   Description :
#
#================================================================

import cv2
import numpy as np
from PIL import Image


label_txt = "./data/recce_data.txt"
num_imgs = len(open(label_txt).readlines())

for img_ind in range(num_imgs):
    image_info = open(label_txt).readlines()[img_ind].split()
    image_path = image_info[0]
    image = cv2.imread(image_path)
    for bbox in image_info[1:]:
        bbox = bbox.split(",")
        image = cv2.rectangle(image,(int(float(bbox[0])),
                                    int(float(bbox[1]))),
                                    (int(float(bbox[2])),
                                  int(float(bbox[3]))), (255,0,0), 2)

    image = Image.fromarray(np.uint8(image))
    print(image_path.split('/')[-1])
    image.show()
