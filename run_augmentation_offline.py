#! /usr/bin/env python
# coding=utf-8
#================================================================
import argparse
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from core.augment_offline import AugmentOffline
from core.yolov3 import YOLOV3
from core.config import cfg

#################################################################
# This code creates augmentations (images + bboxes) offline.
# Possible augmentations: horizontal flip, vertical flip,
# crop, translate, color channels swap.
# Plase see input, output and possibilities under 'main'.
# The code was based on train.py of YOLO_V3 but is done offline.
#################################################################


class AugmentationsOffline(object):
    def __init__(self, annot_path, output_path, aug_type):
        self.learn_rate_init     = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end      = cfg.TRAIN.LEARN_RATE_END
        self.first_stage_epochs  = cfg.TRAIN.FISRT_STAGE_EPOCHS
        self.second_stage_epochs = cfg.TRAIN.SECOND_STAGE_EPOCHS
        self.warmup_periods      = cfg.TRAIN.WARMUP_EPOCHS
        self.chkpnt_to_restore   = cfg.TRAIN.RESTORE_CHKPT
        self.moving_ave_decay    = cfg.YOLO.MOVING_AVE_DECAY
        self.trainset            = AugmentOffline(annot_path, output_path, aug_type)
        self.steps_per_period    = len(self.trainset)
        self.sess                = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.annot_path          = annot_path
        self.aug_type            = aug_type

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')

        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
            warmup_steps = tf.constant(self.warmup_periods * self.steps_per_period,
                                        dtype=tf.float64, name='warmup_steps')
            train_steps = tf.constant( (self.first_stage_epochs + self.second_stage_epochs)* self.steps_per_period,
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
            self.loader = tf.train.Saver(self.net_var)

        with tf.name_scope('summary'):
            tf.summary.scalar("learn_rate",      self.learn_rate)
            tf.summary.scalar("giou_loss",  self.giou_loss)
            tf.summary.scalar("conf_loss",  self.conf_loss)
            tf.summary.scalar("prob_loss",  self.prob_loss)
            tf.summary.scalar("total_loss", self.loss)

            self.write_op = tf.summary.merge_all()

# this function calls AugmentOffline and creates the needed augmentation (image + bbox)
    def augment(self):
        self.sess.run(tf.global_variables_initializer())

        try:
            self.loader.restore(self.sess, self.chkpnt_to_restore)
        except:
            self.first_stage_epochs = 0

        for epoch in range(1, 1+self.first_stage_epochs+self.second_stage_epochs):
            if epoch <= self.first_stage_epochs:
                train_op = self.train_op_with_frozen_variables
            else:
                train_op = self.train_op_with_all_variables

            pbar = tqdm(self.trainset)
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in pbar:
                _, summary, train_step_loss, global_step_val = self.sess.run(
                    [train_op, self.write_op, self.loss, self.global_step],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                })

def main(annot_path, output_path, aug_type):
    np.random.seed(0)
    tf.set_random_seed(0)
    AugmentationsOffline(annot_path, output_path, aug_type).augment()


# input: annotations path with txt files which look like: /home/mayarap/tamar_pc_DB/DBs/Reccelite_iter1/Taggings/all_imgs/0b0407e2-130c-d5a2-371c-041ca42ceb75_0_van.jpg 111.58578,5.8808684,173.12396,35.28521,1.0 370.00107,8.354597,429.82354999999995,39.444653,15.0 440.31635000000006,0.1369606,497.3369,29.455198,1.0
# output: path for folder of augmentation images and txt file of their bbox annotations.
# possible augmentation types: horizontal flip, vertical flip, crop, translate, color channels swap.

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Offline augmentations",
                                     description="")

    parser.add_argument('--annot_path',
                        default="/home/mayarap/Reccelite/ourRep/yolo3_A/data/croppedtxt/Tagging4_cropReg.txt",
                        help="recce data tagging txt file")
    parser.add_argument('--output_path',
                        default="/home/mayarap/tamar_pc_DB/DBs/Reccelite_iter1/Tagging4_cropReg_out",
                        help="directory for augmentations output")
    parser.add_argument('--aug_type',
                        default='hflip',
                        help="options are: 'hflip', 'vflip', 'crop', 'translate', 'color' ")

    opts = parser.parse_args()

    main(annot_path=opts.annot_path, output_path = opts.output_path, aug_type=opts.aug_type)
