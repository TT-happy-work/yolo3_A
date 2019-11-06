#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
# ================================================================

import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOV3
from core.config import cfg
import datetime
import subprocess as sp
import json

from tensorflow.python.client import timeline


class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = cfg.YOLO.ANCHOR_PER_SCALE
        self.classes = utils.read_class_names(cfg.YOLO.CLASSES)
        self.num_classes = len(self.classes)
        self.learn_rate_init = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        # nadav_wp_pruning-
        self.pruning_epoch_freq = cfg.TRAIN.PRUNING_EPOCH_FREQ
        # -nadav_wp_pruning
        self.warmup_periods = cfg.TRAIN.WARMUP_EPOCHS
        self.chkpnt_to_restore = cfg.TRAIN.RESTORE_CHKPT
        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        self.moving_ave_decay = cfg.YOLO.MOVING_AVE_DECAY
        self.max_bbox_per_scale = 150
        self.train_logdir = "./data/log/train"
        self.trainset = Dataset('train')
        self.testset = Dataset('test')
        self.steps_per_period = len(self.trainset)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.folder_name = cfg.YOLO.ROOT_DIR + cfg.YOLO.EXP_DIR
        self.upsample_method     = cfg.YOLO.UPSAMPLE_METHOD
        self.data_format = cfg.YOLO.DATA_FORMAT
        self.max_to_keep = cfg.TRAIN.MAX_TO_KEEP

        with tf.name_scope('output_folder'):
            timestr = datetime.datetime.now().strftime('%d%h%y_%H%M')
            for i in range(0, len(sp.getstatusoutput('git branch')[1].split())):
                if sp.getstatusoutput('git branch')[1].split()[i] == '*':
                    gitBranch = sp.getstatusoutput('git branch')[1].split()[i + 1]
            gitCommitID = sp.getstatusoutput('git rev-parse --short HEAD')[1]
            self.output_folder = os.path.join(
                self.folder_name[0] + self.folder_name[1] + '_' + gitBranch + '_' + timestr + '_' + gitCommitID)
            if not os.path.exists(self.output_folder):
                os.makedirs(self.output_folder)
            cfg_new_path = os.path.join(self.output_folder, 'configFile.txt')
            shutil.copyfile('core/config.py', cfg_new_path)

        with tf.name_scope('define_input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):

            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                self.label_sbbox, self.label_mbbox, self.label_lbbox,
                self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                       dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant((self.first_stage_epochs + self.second_stage_epochs) * self.steps_per_period,
                                      dtype=tf.float64, name='train_steps')
            self.learn_rate = tf.cond(
                pred=self.global_step < warmup_steps,
                true_fn=lambda: self.global_step / warmup_steps * self.learn_rate_init,
                false_fn=lambda: self.learn_rate_end + 0.5 * (self.learn_rate_init - self.learn_rate_end) *
                                 (1 + tf.cos(
                                     (self.global_step - warmup_steps) / (train_steps - warmup_steps) * np.pi))
            )
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_first_stage_train"):
            self.first_stage_trainable_var_list = []
            for var in tf.trainable_variables():
                var_name = var.op.name
                var_name_mess = str(var_name).split('/')
                if var_name_mess[0] in ['conv_sbbox', 'conv_mbbox', 'conv_lbbox']:
                    self.first_stage_trainable_var_list.append(var)

            first_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                     var_list=self.first_stage_trainable_var_list)
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([first_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_frozen_variables = tf.no_op()

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()

        with tf.name_scope('loader_and_saver'):
            # nadav_wp_pruning-
            # self.net_var = tf.global_variables()  # net variables in graph
            ckpt_net_var = [tup[0] for tup in
                            tf.train.list_variables(self.chkpnt_to_restore)]  # net variables in checkpoint
            # restore only variables that exist both in the ckpt and the graph
            variables_to_restore = [var for var in self.net_var if var.name.split(':')[0] in ckpt_net_var]
            self.loader = tf.train.Saver(variables_to_restore)
            # -nadav_wp_pruning
            self.saver = tf.train.Saver(tf.global_variables(), self.max_to_keep)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate", self.learn_rate)
            tf.summary.scalar("giou_loss", self.giou_loss)
            tf.summary.scalar("conf_loss", self.conf_loss)
            tf.summary.scalar("prob_loss", self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            logdir = self.output_folder + "/log/"
            if os.path.exists(logdir): shutil.rmtree(logdir)
            os.mkdir(logdir)
            self.write_op = tf.summary.merge_all()
            self.summary_writer_train = tf.summary.FileWriter(logdir + "/train", graph=self.sess.graph)
            self.summary_writer_test = tf.summary.FileWriter(logdir + "/test", graph=self.sess.graph)

        # nadav_wp_pruning-
        ih = self.trainset.input_height
        iw = self.trainset.input_width
        # g = tf.get_default_graph()
        # run_meta = tf.RunMetadata()
        # opts = tf.profiler.ProfileOptionBuilder.float_operation()
        # flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        # -nadav_wp_pruning

    def train(self):
        self.sess.run(tf.global_variables_initializer())
        try:
            print('=> Restoring weights from: %s ... ' % self.chkpnt_to_restore)
            if self.upsample_method == 'resize':
                # Load all from coco
                self.loader.restore(self.sess, self.chkpnt_to_restore)

            elif self.upsample_method == 'deconv':
                # Load selectively from coco
                layers_to_restore = self.net_var[0:297]
                layer_to_restore = [v for v in layers_to_restore]
                saver = tf.train.Saver(layer_to_restore)
                saver.restore(self.sess, self.chkpnt_to_restore)

                print('Last restored layer is: %s' % self.net_var[297-1])
                print('Next un-restored layer is: %s' % self.net_var[297])

                layers_to_restore = self.net_var[300:336]
                layer_to_restore = [v for v in layers_to_restore]
                saver = tf.train.Saver(layer_to_restore)
                saver.restore(self.sess, self.chkpnt_to_restore)

                print('Last restored layer is: %s' % self.net_var[336-1])
                print('Next un-restored layer is: %s' % self.net_var[336])

                layers_to_restore = self.net_var[339:371]
                layer_to_restore = [v for v in layers_to_restore]
                saver = tf.train.Saver(layer_to_restore)
                saver.restore(self.sess, self.chkpnt_to_restore)

        except:
            print('=> %s does not exist !!!' % self.chkpnt_to_restore)
            print('=> Now it starts to train YOLOV3 from scratch ...')
            self.first_stage_epochs = 0

        for epoch in range(1, 1 + self.first_stage_epochs + self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            j = (epoch - 1) * 85

            for train_data in pbar:
                if self.data_format == "NCHW":
                    input_data = train_data[0].transpose([0, 3, 1, 2])  # switch from NHWC to NCHW
                else:
                    input_data = train_data[0]

                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],
                    #run_metadata=run_metadata, options=run_options,
                                                feed_dict={
                                                self.input_data:   input_data,
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True
                })
                self.summary_writer_train.add_summary(summary, global_step_val)
                train_epoch_loss.append(train_step_loss)
                pbar.set_description("train loss: %.2f" %train_step_loss)
                # # nadav_wp_pruning-
                # # check inference time (of a batch)
                # run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                # run_metadata = tf.RunMetadata()
                # iter_start_time = time.time()
                # _, _, _, global_step_val = self.sess.run(
                #     [self.model.pred_lbbox, self.model.pred_mbbox, self.model.pred_sbbox, self.global_step], feed_dict={
                #         self.input_data: input_data,
                #         self.trainable: False
                #     },
                #     run_metadata=run_metadata, options=run_options
                # )
                # iter_runtime = time.time() - iter_start_time
                #
                # iter_runtime = tf.Summary(
                #     value=[tf.Summary.Value(tag="iteration runtime - only feed forward", simple_value=iter_runtime), ])
                # self.summary_writer_train.add_summary(iter_runtime, j)  # global_step_val)
                #
                # j += 1
                # self.summary_writer_train.add_run_metadata(run_metadata, ',step %d' % j)
                #
                # ####### timeline object for runtime analysis #######
                # tl = timeline.Timeline(run_metadata.step_stats)
                # ctf = tl.generate_chrome_trace_format()
                # with open('timeline.json', 'w') as f:
                #     f.write(ctf)
                # f.close()
                #
                # ####### timeline json #######
                # with open('timeline.json') as json_file:
                #     data = json.load(json_file)
                #
                # prefixes = ['define_loss/darknet/', 'define_loss/']
                # gpu_proccess_id = 5
                # # total_dur = 0
                # # conv_events = []
                # # conv_dur = 0
                # conv_layers = {}
                #
                # for event in data['traceEvents']:
                #     if event['pid'] == gpu_proccess_id and 'dur' in event:
                #             # total_dur += event['dur']
                #             if 'name' in event['args'] and 'Conv2D' in event['args']['name']:
                #                 conv_layer = event['args']['name']
                #                 for prefix in prefixes:
                #                     conv_layer = conv_layer.replace(prefix, '')
                #
                #                 conv_layer, _ = conv_layer.split('Conv2D', 1)
                #                 if conv_layer not in conv_layers:
                #                     conv_layers[conv_layer] = event['dur']
                #                 else:
                #                     conv_layers[conv_layer] += event['dur']
                #                 # conv_events.append(event)
                #                 # conv_dur += event['dur']
                #
                # # how to prune convolution layers before residual blocks?
                # # conv 1, conv 4, conv 9, conv 26, conv 43 -> all conv layers in darknet
                # # -nadav_wp_pruning

            for test_data in self.testset:
                if self.data_format == "NCHW":
                    input_data = test_data[0].transpose([0, 3, 1, 2])  # switch from NHWC to NCHW
                else:
                    input_data = test_data[0]

                summary, test_step_loss = self.sess.run(
                    [self.write_op, self.loss], feed_dict={
                        self.input_data: input_data,
                        self.label_sbbox: test_data[1],
                        self.label_mbbox: test_data[2],
                        self.label_lbbox: test_data[3],
                        self.true_sbboxes: test_data[4],
                        self.true_mbboxes: test_data[5],
                        self.true_lbboxes: test_data[6],
                        self.trainable: False,
                    })
                test_epoch_loss.append(test_step_loss)
                self.summary_writer_test.add_summary(summary, global_step_val)

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "/checkpoints/yolov3_epoch=%s_test_loss=%.4f.ckpt" % (epoch, test_epoch_loss)
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                  % (epoch, log_time, train_epoch_loss, test_epoch_loss, self.output_folder + ckpt_file))
            self.saver.save(self.sess, save_path=os.path.join(self.output_folder + ckpt_file), global_step=epoch)

            # nadav_wp_pruning-
            # prune_epoch = True if (epoch % self.pruning_epoch_freq == 0) else False
            # if prune_epoch:
            #     masks = tf.get_collection('masks')
            #     for mask in masks:
            #         indices = [0, 1]  # decide which filters to zero, current just for debug
            #         # selecting indices:
            #         # 1.
            #         #
            #         #
            #
            #
            #         self.sess.run(tf.scatter_update(mask, indices, 0))
            #
            # # -nadav_wp_pruning

    # calculate complexity per layer
    def calc_net_complexity(self):
        ih = self.trainset.input_height
        iw = self.trainset.input_width

        return


if __name__ == '__main__':
    np.random.seed(0)
    tf.set_random_seed(0)
    YoloTrain().train()
