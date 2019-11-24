import os
import argparse
import time

import uff
import numpy as np
from tqdm import tqdm
import tensorflow as tf

from core import utils
from core.config import cfg
from core.yolov3 import YOLOV3


class ModelLoader():
    def __init__(self):
        utils.modify_config_for_tensorrt()
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
            self.input_data = tf.placeholder(dtype=tf.float32,
                                             shape=(cfg.TEST.BATCH_SIZE, cfg.TEST.IMAGE_H, cfg.TEST.IMAGE_W, 3),
                                             name='input_data')
            self.trainable = tf.constant(dtype=tf.bool, name='trainable', value=False)

        self.model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = \
            self.model.pred_sbbox, self.model.pred_mbbox, self.model.pred_lbbox

        #self.output_nodes = [self.model.pred_sbbox, self.model.pred_mbbox, self.model.pred_lbbox]
        self.output_nodes = [self.model.conv_sbbox, self.model.conv_mbbox, self.model.conv_lbbox]
        self.input_nodes = [self.input_data, self.trainable]

        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        # In order to propose tensor-rt compatibility, we have modified the graph and therefore some of the variables
        # in the checkpoint may not be present in our graph. In order to avoid any error during the checkpoint restore
        # we must avoid the attempt to load variables that does not exist in the graph.
        vars_map = ema_obj.variables_to_restore()
        ckpt_var_names = dict(tf.train.list_variables(self.weight_file))
        keys_to_remove = [k for k in vars_map if k not in ckpt_var_names]
        tensors_to_init = [vars_map[k] for k in keys_to_remove]
        for k in keys_to_remove:
            vars_map.pop(k)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(vars_map)
        self.saver.restore(self.sess, self.weight_file)
        self.sess.run(tf.variables_initializer(tensors_to_init))


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


def tf_measure_timings(n_iterations=10, n_samples_per_iteration=100):
    # build graph:
    tf_model = ModelLoader()
    out_node_names = [x.name.split(':')[0] for x in tf_model.output_nodes]
    const_graph = tf.graph_util.convert_variables_to_constants(tf_model.sess, tf_model.sess.graph_def, out_node_names)
    inference_graph = tf.graph_util.remove_training_nodes(const_graph)
    # generate random input:
    in_shape = (cfg.TEST.BATCH_SIZE, cfg.TEST.IMAGE_H, cfg.TEST.IMAGE_W, 3)
    rand_batch_data = np.random.random(in_shape)
    durations = np.full((n_iterations,n_samples_per_iteration), np.finfo(np.float32).max)
    # measure
    for i in range(n_iterations):
        pbar = tqdm(range(n_samples_per_iteration))
        for j in pbar:
            t_start = time.time()
            _ = tf_model.sess.run(tf_model.output_nodes, feed_dict={tf_model.input_data: rand_batch_data})
            t_end = time.time()
            it_dur = t_end - t_start
            pbar.set_description("last iteration took: %.3f seconds." % it_dur)
            durations[i, j] = it_dur
    for i in range(n_iterations):
        print('==> Average duration for last %s itarations was %.3f seconds/iteration' %
              (n_samples_per_iteration, np.average(durations[i])))
    return None


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
    # tf_measure_timings()
