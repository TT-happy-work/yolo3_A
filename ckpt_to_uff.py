import os
import argparse

import uff
import numpy as np
import tensorflow as tf

from core import utils
from core.config import cfg
from core.yolov3 import YOLOV3


class ModelLoader():
    def __init__(self):
        self.target_height = cfg.TEST.IMAGE_H
        self.target_width = cfg.TEST.IMAGE_W
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.anchors = np.array(utils.get_anchors(cfg.YOLO.ANCHORS))
        self.score_threshold = cfg.TEST.SCORE_THRESHOLD
        self.iou_threshold = cfg.TEST.IOU_THRESHOLD
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.annotation_path = cfg.TEST.ANNOT_PATH
        self.weight_file = utils.get_checkpoint_file_path()
        self.write_image = cfg.TEST.WRITE_IMAGE
        self.write_image_path = cfg.TEST.WRITE_IMAGE_PATH
        self.show_label = cfg.TEST.SHOW_LABEL
        self.epilog_logics = cfg.YOLO.EPILOG_LOGICS
        self.specifc_conf_file = cfg.YOLO.CONF_TH_FILE

        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable = tf.constant(dtype=tf.bool, name='trainable', value=False)

        self.model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = \
            self.model.pred_sbbox, self.model.pred_mbbox, self.model.pred_lbbox

        self.output_nodes = [self.model.pred_sbbox, self.model.pred_mbbox, self.model.pred_lbbox]
        self.input_nodes = [self.input_data, self.trainable]

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)


def _node_name(n):
  if n.startswith("^"):
    return n[1:]
  else:
    return n.split(":")[0]


def sizeof_fmt(num, suffix='B'):
    numf = float(num)
    units = ['', 'K', 'M', 'G']
    for ex, unit in enumerate(units):
        div = 2 ** (10*(ex))
        if abs(numf) < 1024.0 * div:
            return "%3.1f%s%s" % (numf/div, unit, suffix)
    return "%.1f%s%s" % (numf, 'T', suffix)


def yolo_checkpoint_to_uff(uff_file='out.uff'):
    tf_model = ModelLoader()
    out_node_names = [x.name.split(':')[0] for x in tf_model.output_nodes]
    const_graph = tf.graph_util.convert_variables_to_constants(tf_model.sess, tf_model.sess.graph_def, out_node_names)
    inference_graph = tf.graph_util.remove_training_nodes(const_graph)
    ret = uff.from_tensorflow(inference_graph, out_node_names, output_filename=uff_file)
    print("Model converted to file %s (size: %s)" % (uff_file, sizeof_fmt(os.stat(uff_file).st_size)))
    return ret


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert Yolo3 checkpoint file to uff")
    parser.add_argument('--uff_out_file_name', default='out.uff',
                        help='The output file of the converted model [default: out.uff]')
    parser.add_argument('--ckpt_file_name', default=None,
                        help='Optional, explicit checkpoint file to load. otherwise the cfg.TEST.WEIGHT_FILE/DIR '\
                             'parameters of the config file are used for determining which checkpoint file to use.')
    opts = parser.parse_args()
    if opts.ckpt_file_name is not None:
        cfg.TEST.USE_SPECIFIED_WEIGHT_FILE = True
        cfg.TEST.WEIGHT_FILE = opts.ckpt_file_name
    yolo_checkpoint_to_uff(opts.uff_out_file_name)
