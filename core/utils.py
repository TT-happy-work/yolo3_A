#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : utils.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 13:14:19
#   Description :
#
#================================================================

import os
import cv2
import glob
import random
import colorsys
import numpy as np
import tensorflow as tf
from core.config import cfg
import matplotlib.pyplot as plt

def read_class_names(class_file_name):
    '''loads class name from a file'''
    names = {}
    with open(class_file_name, 'r') as data:
        for ID, name in enumerate(data):
            names[ID] = name.strip('\n')
    return names

def read_conf_th(conf_th_file_name):
    '''loads class name from a file'''
    names = {}
    with open(conf_th_file_name, 'r') as data:
        for name in data:
            names[name.split(',')[0]] = float(name.split(',')[1].split('\n')[0])
    return names

def get_checkpoint_file_path():
    if not cfg.TEST.USE_WEIGHTS_DIR:
        return cfg.TEST.WEIGHT_FILE
    weight_dir = cfg.TEST.WEIGHT_DIR
    if not os.path.isdir(weight_dir):
        raise Exception("USE_WEIGHTS_DIR requested but WEIGHT_DIR (%s) is not a directory" % weight_dir)
    checkpoints_file = os.path.join(weight_dir, 'checkpoint')
    if os.path.isfile(checkpoints_file):
        with open(checkpoints_file, 'r') as fd:
            line = fd.readline()
            f = line.split(':')[1].strip().strip('"')
            checkpoint_file = os.path.join(weight_dir,f)
            return checkpoint_file
    # failed to read the checkpoint file - looking for the newest file in dir
    list_of_files = glob.glob(weight_dir + '/*.ckpt-[0-9]*.meta')  # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    return latest_file.rstrip('.meta')


def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = np.array(anchors.split(), dtype=np.float32)
    return anchors.reshape(3, 3, 2)


def image_preporcess(image, target_height, target_width, gt_boxes=None):

    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

    ih, iw    = target_height, target_width
    h,  w, _  = image.shape

    scale = min(iw/w, ih/h)
    nw, nh  = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))
    image_cropped = image[:target_height, :target_width]

    image_paded = np.full(shape=[ih, iw, 3], fill_value=128.0)
    dw, dh = (iw - nw) // 2, (ih-nh) // 2
    image_paded[dh:nh+dh, dw:nw+dw, :] = image_resized
    image_paded = image_paded / 255.

    if gt_boxes is None:
        return image_paded
    elif cfg.YOLO.IMAGE_HANDLE=='scale':
        gt_boxes[:, [0, 2]] = gt_boxes[:, [0, 2]] * scale + dw
        gt_boxes[:, [1, 3]] = gt_boxes[:, [1, 3]] * scale + dh
        return image_paded, gt_boxes
    elif cfg.YOLO.IMAGE_HANDLE=='crop':
        gt_boxes_cropped = []
        for gt in gt_boxes:
            if (gt[0] >= target_width and gt[2] >= target_width) or (gt[1] >= target_height and gt[2] >= target_height):   # entire box is outside of cropped image
                continue
            elif gt[0] < target_width and gt[2] < target_width and gt[3] < target_height and gt[1] < target_height: # entire bbox is included in cropped image
                gt_boxes_cropped.append(gt)
            else:
                if gt[0] >= target_width: gt[0] = target_width - 1
                if gt[1] >= target_height: gt[1] = target_height - 1
                if gt[2]>=target_width: gt[2]=target_width-1
                if gt[3] >= target_height: gt[3] = target_height - 1
                gt_boxes_cropped.append(gt)
        return np.array(image_cropped), np.array(gt_boxes_cropped)

    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    cropTH = 0.6
    # im = image
    # plt.figure(); plt.imshow(image/255); plt.title('full org'); plt.show()
    # ih, iw    = target_height, target_width
    # h,  w, _  = image.shape
    # image_resized = cv2.resize(image/255, (target_width, target_height))
    # image_cropped = image[:target_height, :target_width] /255
    # plt.figure(); plt.imshow(image_cropped); plt.show()
    # bboxes = np.array([172.2945,  2266.1885,   211.48303, 2301.762 , 0.37, 6.])
    # image_draw_full = draw_bbox(im/255, bboxes)
    # plt.figure(); plt.imshow(image_draw_full); plt.title('full'); plt.show()
    # bboxes = np.array([172.2945,  639,   211.48303, 639 , 0.37, 6.])
    # image_draw_crop = draw_bbox(image_cropped, bboxes)
    # plt.figure(); plt.imshow(image_draw_crop); plt.title('crop'); plt.show()

    ih, iw = target_height, target_width
    h, w, _ = image.shape
    image_scaled = cv2.resize(image / 255, (target_width, target_height))
    image_cropped = image[:target_height, :target_width] / 255

    #draw_gt_bbox(image, gt_boxes)
    if gt_boxes is None:
        if cfg.YOLO.IMAGE_HANDLE=='scale':
            return image_scaled
        elif cfg.YOLO.IMAGE_HANDLE=='crop':
            return image_cropped
        else:
            return image_cropped
    elif cfg.YOLO.IMAGE_HANDLE=='scale':
        gt_boxes_scaled = np.zeros_like(gt_boxes)
        gt_boxes_scaled[:, [0, 2]] = gt_boxes[:, [0, 2]] * iw/w
        gt_boxes_scaled[:, [1, 3]] = gt_boxes[:, [1, 3]] * ih/h
        gt_boxes_scaled[:, -1] = gt_boxes[:, -1]
        #draw_gt_bbox(image_scaled*255, gt_boxes_scaled)
        return image_scaled, gt_boxes_scaled
    elif cfg.YOLO.IMAGE_HANDLE=='crop':
        gt_boxes_cropped = []
        for gt in gt_boxes:
            if (gt[0] >= target_width and gt[2] >= target_width) or (gt[1] >= target_height and gt[3] >= target_height):   # entire box is outside of cropped image
                continue
            elif gt[0] < target_width and gt[2] < target_width and gt[3] < target_height and gt[1] < target_height:        # entire bbox is included in cropped image
                gt_boxes_cropped.append(gt)
            elif (gt[0] < target_width and gt[2] >= target_width and (target_width-gt[0])/(gt[2]-gt[0])< cropTH) or (gt[1] < target_height and gt[3] >= target_height and (target_height-gt[1])/(gt[3]-gt[1])< cropTH): # if only a small portion of target is in the frame then ignore it
                continue
            else:
                if gt[0] >= target_width:  gt[0] = target_width - 1
                if gt[1] >= target_height: gt[1] = target_height - 1
                if gt[2] >= target_width:  gt[2] = target_width-1
                if gt[3] >= target_height: gt[3] = target_height - 1
                gt_boxes_cropped.append(gt)
        #draw_gt_bbox(image_cropped*255, gt_boxes_cropped)
        return image_cropped, np.array(gt_boxes_cropped)


def draw_bbox(image, bboxes, classes=[], show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
    """

    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    # random.seed(None)

    # print('len(bboxes) = ', len(bboxes))
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        score = bbox[4]
        class_ind = int(bbox[5])
        # print('class_ind = ', class_ind)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], score)
            t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick//2)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

            cv2.putText(image, bbox_mess, (c1[0], c1[1]-2), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, (0, 0, 0), bbox_thick//2, lineType=cv2.LINE_AA)

    return image


def draw_gt_bbox(image, gt_boxes, classes=[], show_label=True):
    """
    bboxes: [x_min, y_min, x_max, y_max, cls_id] format coordinates.
    """

    from PIL import Image
    num_classes = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / num_classes, 1., 1.) for x in range(num_classes)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))

    random.seed(0)
    random.shuffle(colors)
    # random.seed(None)

    # print('len(bboxes) = ', len(bboxes))
    #plt.figure()
    if gt_boxes==[]: return
    for i, bbox in enumerate(gt_boxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        fontScale = 0.5
        class_ind = int(bbox[4])
        # print('class_ind = ', class_ind)
        bbox_color = colors[class_ind]
        bbox_thick = int(0.6 * (image_h + image_w) / 600)
        c1, c2 = (coor[0], coor[1]), (coor[2], coor[3])
        img = cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)

        bbox_mess = '%s' % (classes[class_ind])
        t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
        cv2.rectangle(img, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled

        cv2.putText(img, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)

    image_box = Image.fromarray(np.uint8(img))
    image_box.show()

    return


def bboxes_iou(boxes1, boxes2):

    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up       = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down    = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area    = inter_section[..., 0] * inter_section[..., 1]
    union_area    = boxes1_area + boxes2_area - inter_area
    ious          = np.maximum(1.0 * inter_area / union_area, np.finfo(np.float32).eps)

    return ious



def read_pb_return_tensors(graph, pb_file, return_elements):

    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        frozen_graph_def = tf.GraphDef()
        frozen_graph_def.ParseFromString(f.read())

    with graph.as_default():
        return_elements = tf.import_graph_def(frozen_graph_def,
                                              return_elements=return_elements)
    return return_elements


def nms(bboxes, iou_threshold, sigma=0.3, method='nms'):
    """
    :param bboxes: (xmin, ymin, xmax, ymax, score, class)

    Note: soft-nms, https://arxiv.org/pdf/1704.04503.pdf
          https://github.com/bharatsingh430/soft-nms
    """
    classes_in_img = list(set(bboxes[:, 5]))
    best_bboxes = []

    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5] == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            iou = bboxes_iou(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            weight = np.ones((len(iou),), dtype=np.float32)

            assert method in ['nms', 'soft-nms']

            if method == 'nms':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0

            if method == 'soft-nms':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > 0.
            cls_bboxes = cls_bboxes[score_mask]

    return best_bboxes


def postprocess_boxes(pred_bbox, org_img_shape, input_h, input_w, score_threshold):

    valid_scale=[0, np.inf]
    pred_bbox = np.array(pred_bbox)

    pred_xywh = pred_bbox[:, 0:4]
    pred_conf = pred_bbox[:, 4]
    pred_prob = pred_bbox[:, 5:]

    # # (1) (x, y, w, h) --> (xmin, ymin, xmax, ymax)
    pred_coor = np.concatenate([pred_xywh[:, :2] - pred_xywh[:, 2:] * 0.5,
                                pred_xywh[:, :2] + pred_xywh[:, 2:] * 0.5], axis=-1)
    # # (2) (xmin, ymin, xmax, ymax) -> (xmin_org, ymin_org, xmax_org, ymax_org)
    org_h, org_w = org_img_shape
    resize_ratio = min(input_w / org_w, input_h / org_h)

    dw = (input_w - resize_ratio * org_w) / 2
    dh = (input_h - resize_ratio * org_h) / 2

    if cfg.YOLO.IMAGE_HANDLE == 'scale':
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2]) * org_w / input_w
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2]) * org_h / input_h
    elif cfg.YOLO.IMAGE_HANDLE == 'crop':
        pass

    # # (3) clip some boxes those are out of range
    pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
    invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
    pred_coor[invalid_mask] = 0
            # true_positive_dict[c] += p if t >= p else t

    # # (4) discard some invalid boxes
    bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
    scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

    # # (5) discard some boxes with low scores
    classes = np.argmax(pred_prob, axis=-1)
    scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
    score_mask = scores > score_threshold
    mask = np.logical_and(scale_mask, score_mask)
    coors, scores, classes = pred_coor[mask], scores[mask], classes[mask]

    return np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)



