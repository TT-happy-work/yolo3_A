import os
import sys
import train
import argparse
import numpy as np
import tensorflow as tf
from core.config import cfg


def compare_results(expected_dict, observed_dict, tolerance = 0.005):
    n_errors = 0
    for key in ['train', 'test']:
        if abs(expected_dict[key] - observed_dict[key]) > tolerance:
            print("Error at %s: expected(%s) != observed(%s)" % (key, expected_dict[key], observed_dict[key]))
            n_errors += 1
    return n_errors


def parse_cmdline():
    parser = argparse.ArgumentParser(description='Check that train remains consistent (in terms of test/train loss)')
    parser.add_argument('--seed', default=2019, type=int,
                        help='an integer seed value to seed the graph-level seed with')
    parser.add_argument('--tolerance', default=0.005, type=float,
                        help='The maximum allowed error between expected and observed results')
    parser.add_argument('--first_stage_epochs', default=2, type=int,
                        help='The number of epochs for the first stage of the train')
    parser.add_argument('--second_stage_epochs', default=3, type=int,
                        help='The number of epochs for the second stage of the train')
    parser.add_argument('--train_annot_path', default='./data/dataset/recce_short_train_Tagging.txt', type=str,
                        help='The path of the annotation file used for train')
    parser.add_argument('--test_annot_path', default='./data/dataset/recce_short_test_Tagging.txt', type=str,
                        help='The path of the annotation file used for test (eval)')
    parser.add_argument('--expected_train_loss', default=9177.667, type=float,
                        help='The expected loss of the end of the train process')
    parser.add_argument('--expected_test_loss', default=5458.8022, type=float,
                        help='The expected loss of the end of the test process')
    parser.epilog = 'By default this test should be run as-is and pass. ' \
                    'NOTE(!!!) however that when some non-default args are provided, ' \
                    'a new expected_train/test_loss arguments should be probably provided as well.'
    return parser.parse_args(), parser


if __name__ == '__main__':
    opts, arg_parser = parse_cmdline()
    # prepare to test:
    np.random.seed(opts.seed)
    tf.set_random_seed(opts.seed)
    expected = {'test': opts.expected_test_loss, 'train': opts.expected_train_loss}

    cfg.TRAIN.FISRT_STAGE_EPOCHS = opts.first_stage_epochs
    cfg.TRAIN.SECOND_STAGE_EPOCHS = opts.second_stage_epochs
    cfg.TRAIN.BE_REPRODUCIBLE = True
    cfg.TRAIN.ANNOT_PATH = opts.train_annot_path
    cfg.TEST.ANNOT_PATH = opts.test_annot_path
    cwd = os.getcwd()
    test_dir = os.path.dirname(__file__)
    project_dir = os.path.dirname(test_dir)

    # the train itself
    os.chdir(project_dir)  # train.YoloTrain() is using relative paths
    yt = train.YoloTrain()
    yt.train()

    # reproducibility test:
    observed = {'test': yt.test_epoch_loss, 'train': yt.train_epoch_loss}
    num_errors = compare_results(expected, observed, opts.tolerance)
    if num_errors == 0:
        print('SUCCESS: Expected results were observed')
    os.chdir(cwd)  # going back to the work-dir which we started from
    sys.exit(num_errors)  # return an error code on failure

