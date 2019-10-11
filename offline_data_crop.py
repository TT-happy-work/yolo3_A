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

###################################################################################

write_image_path = '/home/tamar/DBs/Reccelite/CroppedDB/croppedImgs_1_2_3_4_5_Th06/'

# the Tagging to be cropped:
f_data_in_path = '/home/tamar/RecceLite_code_packages/yolo3_baseline2/data/individual_datasets_without_24/recce_data_Tagging_1_2_3_4_5.txt'

# the outout of this script: the cropped Tagging
f_data_out_path = '/home/tamar/DBs/Reccelite/CroppedDB/cropped_1_2_3_4_5_Th06.txt'

patch_w = 1*800
patch_h = 1*640

cropTH = 0.6 # above this percentage of the bbox included in the path - clip the box. else - ignore the bbox.

###################################################################################



if not os.path.exists(write_image_path):
    os.makedirs(write_image_path)
with open(f_data_in_path, 'r') as f_in:
    txt = f_in.readlines()
    annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
new_line_flag = False
# loop over each img in original list: crop each img and add the cropped_img txt to the new datatxt
with open(f_data_out_path, 'w') as f_out:
    for img_ind in range(len(annotations)):
        annotation = annotations[img_ind]
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        print(image_path)
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
        # draw_gt_bbox(image, bboxes)
        for ind_w in range(div_w):
            for ind_h in range(div_h):
                new_img_flag = True
                cropped_bboxes = []
                cropped_image_name = image_name + '_' + str(ind_w) + '_' + str(ind_h) + '.' + image_format
                # patches that are not last row  or last column
                if (ind_w!=div_w-1 and ind_h!=div_h-1):
                    minW = ind_w*patch_w; minH = ind_h*patch_h;
                    maxW = (ind_w + 1)*patch_w; maxH = (ind_h + 1)*patch_h
                # patch that is last row and last column
                elif (ind_w == div_w - 1 and ind_h == div_h - 1):
                    minW = orig_image_w - patch_w; minH = orig_image_h - patch_h;
                    maxW = orig_image_w - 1; maxH = orig_image_w - 1
                # patches that are last row
                elif ind_w != div_w - 1:
                    minW = ind_w*patch_w; minH =  orig_image_h - patch_h
                    maxW = (ind_w + 1)*patch_w; maxH = orig_image_w - 1
                # patches that are last column
                else:
                    minW =orig_image_w - patch_w; minH = ind_h*patch_h;
                    maxW = orig_image_w - 1; maxH = (ind_h + 1)*patch_h
                cropped_image = image[minH:maxH, minW:maxW]
                cv2.imwrite(write_image_path + cropped_image_name, cropped_image)
                for box in bboxes:
                    cropped1 = []; croppedOne = []
                    # boxes that are completely out of this patch: ignore
                    if box[0]>=maxW or box[1]>=maxH or box[2]<=minW or (box[3]<=minH):
                        continue
                    # boxes that are entirely included in patch: keep
                    elif box[0]>=minW and box[1]>=minH and box[2]<=maxW and box[3]<=maxW:
                        # print('entire')
                        cropped1 = box
                    # boxes that are partially included: check the extent of overlap
                    else:
                        # print('overlap')
                        areaBox = (box[2] - box[0]) * (box[3] - box[1])
                        npbox = np.array(box)
                        nppatch = np.array([minW, minH, maxW-1, maxH-1])
                        left_up = np.maximum(npbox[:2], nppatch[:2])
                        right_down = np.minimum(npbox[2:4], nppatch[2:4])
                        inter_section = np.maximum(right_down - left_up, 0.0)
                        inter_area = inter_section[0] * inter_section[1]
                        if inter_area >= cropTH*areaBox:
                            cropped1 = (np.concatenate([left_up, right_down])).tolist()
                            np.array(cropped1.append(box[4]))
                            cropped1 = np.array(cropped1)
                            # print('Successoverlap')
                        else:
                            # print('Failoverlap')
                            continue
                    # adjust coordinates
                    croppedOne[:2] = cropped1[:2] - [minW, minH]
                    croppedOne[2:4] = cropped1[2:4] - [minW, minH]
                    croppedOne.append(cropped1[4])
                    # print(ind_w, ind_h)
                    # print(minW, minH, maxW, maxH)
                    # print(box)
                    # print(cropped1)
                    # print(croppedOne)
                    # write box to new dataTxt:
                    if new_img_flag:
                        if new_line_flag:
                            f_out.write('\n')
                        new_line_flag = True
                        f_out.write(write_image_path + cropped_image_name + ' ')
                    new_img_flag = False
                    f_out.write(str(croppedOne[0]) + ',' + str(croppedOne[1]) + ',' + str(croppedOne[2]) + ',' + str(croppedOne[3]) + ',' + str(croppedOne[4]) + ' ')
                    cropped_bboxes.append(croppedOne)
                # print(image_name, 'minW: ', minW, 'minH: ', minH, 'maxW: ', maxW, 'maxH: ', maxH, 'len(cropped_bboxes): ', len(cropped_bboxes))
                # draw_gt_bbox(cropped_image, cropped_bboxes)
                pass
