import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2
import os
from core.utils import draw_gt_bbox

###################################################################################
## This script preorganizes the Taggings of Yolov3 baseline 2 format [f_data_in_path]
## into a desired [patch_w, patch_h] cropped Tagging file [f_data_out_path] and images [write_image_path]
## main dir of cropped outputs
write_image_path = '/home/tamar/DBs/Reccelite/CroppedDB/croppedImgs/'

# the Tagging to be cropped:
f_data_in_path = '/home/tamar/RecceLite_code_packages/yolo3_baseline2/data/dataset/recce_all_Tagging_1_2_3_img.txt'

# the outout of this script: the cropped Tagging
f_data_out_path = '/home/tamar/DBs/Reccelite/CroppedDB/cropped_Tagging_1_2_3.txt'

patch_w = 1*800
patch_h = 1*640
###################################################################################



if not os.path.exists(write_image_path):
    os.makedirs(write_image_path)
with open(f_data_in_path, 'r') as f_in:
    txt = f_in.readlines()
    annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
img_ind = 0
# loop over each img in original list: crop each img and add the cropped_img txt to the new datatxt
with open(f_data_out_path, 'w') as f_out:
    for img_ind in range(len(annotations)):
        annotation = annotations[img_ind]
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        bboxes = np.array([list(map(float, box.split(','))) for box in line[1:]])
        image = np.array(cv2.imread(image_path))
        print("original image size is:", image.shape)
        image_format = image_path.split('.')[-1]
        image_name = image_path.split('.')[0].split('/')[-1]
        orig_image_h = image.shape[0]
        orig_image_w = image.shape[1]
        # the number of cropped parts:
        div_w = int(np.ceil(orig_image_w/patch_w)) # num of w sections in w
        div_h = int(np.ceil(orig_image_h/patch_h)) # num of sections in h
        cropped_bboxes = np.zeros_like(bboxes)
        # draw_gt_bbox(image, bboxes)
        for ind_w in range(div_w):
            for ind_h in range(div_h):
                cropped_bboxes[:, -1] = bboxes[:, -1]    # the category needs no coordinate adjustment
                if (ind_w!=div_w-1 and ind_h!=div_h-1):  # a nominal cropped part (i.e: not an end patch
                    cropped_image = image[ind_h*patch_h:(ind_h+1)*patch_h, ind_w*patch_w:(ind_w+1)*patch_w]
                    cropped_bboxes[:, [0, 2]] = bboxes[:, [0,2]] - ind_w*patch_w
                    cropped_bboxes[:, [1, 3]] = bboxes[:, [1,3]] - ind_h*patch_h
                    print('ind_h=', ind_h, 'ind_w=', ind_w)
                    print('image size is: ', cropped_image.shape)
                    cropped_image_name = image_name + '_' + str(ind_w) + '_' + str(ind_h) + '.' + image_format
                    cv2.imwrite(write_image_path + cropped_image_name, cropped_image)
                    if img_ind:
                        f_out.write('\n' + write_image_path + cropped_image_name + ' ')
                    else:
                        f_out.write(write_image_path + cropped_image_name + ' ')
                    for bb_ind in range(len(cropped_bboxes)):
                        f_out.write(str(cropped_bboxes[bb_ind, 0]) + ',' + str(cropped_bboxes[bb_ind, 1]) + ',' + str(cropped_bboxes[bb_ind, 2]) + ',' + str(cropped_bboxes[bb_ind, 3]) + ',' + str(cropped_bboxes[bb_ind, 4]) + ' ')
                    img_ind = 1
                    # draw_gt_bbox(cropped_image, cropped_bboxes)
                elif (ind_w==div_w-1 and ind_h==div_h-1):
                    cropped_image = image[-1-(patch_h-1):, -1-(patch_w-1):]
                    cropped_bboxes[:, [0, 2]] = bboxes[:, [0,2]] - (orig_image_w-patch_w)
                    cropped_bboxes[:, [1, 3]] = bboxes[:, [1,3]] - (orig_image_h-patch_h)
                    print('ind_h=', ind_h, 'ind_w=', ind_w)
                    print('image size is: ', cropped_image.shape)
                    cropped_image_name = image_name + '_' + str(ind_w) + '_' + str(ind_h) + '.' + image_format
                    cv2.imwrite(write_image_path + cropped_image_name, cropped_image)
                    if img_ind:
                        f_out.write('\n' + write_image_path + cropped_image_name + ' ')
                    else:
                        f_out.write(write_image_path + cropped_image_name + ' ')
                    for bb_ind in range(len(cropped_bboxes)):
                        f_out.write(str(cropped_bboxes[bb_ind, 0]) + ',' + str(cropped_bboxes[bb_ind, 1]) + ',' + str(
                            cropped_bboxes[bb_ind, 2]) + ',' + str(cropped_bboxes[bb_ind, 3]) + ',' + str(
                            cropped_bboxes[bb_ind, 4]) + ' ')
                    img_ind = 1
                    # draw_gt_bbox(cropped_image, cropped_bboxes)
                elif ind_w != div_w-1:                     # in case this is a vertical-end patch
                    cropped_image = image[-1-(patch_h-1):, ind_w*patch_w:(ind_w+1)*patch_w]
                    cropped_bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - ind_w * patch_w
                    cropped_bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - (orig_image_h-patch_h)
                    print('ind_h=', ind_h, 'ind_w=', ind_w)
                    print('image size is: ', cropped_image.shape)
                    cropped_image_name = image_name + '_' + str(ind_w) + '_' + str(ind_h) + '.' + image_format
                    cv2.imwrite(write_image_path + cropped_image_name, cropped_image)
                    if img_ind:
                        f_out.write('\n' + write_image_path + cropped_image_name + ' ')
                    else:
                        f_out.write(write_image_path + cropped_image_name + ' ')
                    for bb_ind in range(len(cropped_bboxes)):
                        f_out.write(str(cropped_bboxes[bb_ind, 0]) + ',' + str(cropped_bboxes[bb_ind, 1]) + ',' + str(
                            cropped_bboxes[bb_ind, 2]) + ',' + str(cropped_bboxes[bb_ind, 3]) + ',' + str(
                            cropped_bboxes[bb_ind, 4]) + ' ')
                    img_ind = 1
                    # draw_gt_bbox(cropped_image, cropped_bboxes)
                else:                                    # in case this is a horizontal-end patch
                    cropped_image = image[ind_h*patch_h:(ind_h+1)*patch_h, -1-(patch_w-1):]
                    cropped_bboxes[:, [0, 2]] = bboxes[:, [0,2]] - ind_w*patch_w
                    cropped_bboxes[:, [1, 3]] = bboxes[:, [1,3]] - (orig_image_h-patch_h)
                    print('ind_h=', ind_h, 'ind_w=', ind_w, (orig_image_h-patch_h), ind_w*patch_w)
                    print('image size is: ', cropped_image.shape)
                    cropped_image_name = image_name + '_' + str(ind_w) + '_' + str(ind_h) + '.' + image_format
                    cv2.imwrite(write_image_path + cropped_image_name, cropped_image)
                    if img_ind:
                        f_out.write('\n' + write_image_path + cropped_image_name + ' ')
                    else:
                        f_out.write(write_image_path + cropped_image_name + ' ')
                    for bb_ind in range(len(cropped_bboxes)):
                        f_out.write(str(cropped_bboxes[bb_ind, 0]) + ',' + str(cropped_bboxes[bb_ind, 1]) + ',' + str(
                            cropped_bboxes[bb_ind, 2]) + ',' + str(cropped_bboxes[bb_ind, 3]) + ',' + str(
                            cropped_bboxes[bb_ind, 4]) + ' ')
                    img_ind = 1
