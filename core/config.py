from easydict import EasyDict as edict


__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.EXP_DIR                = "dbg_crop_1320_",
__C.YOLO.ROOT_DIR               = "/home/tamar/RecceLite_code_packages/yolo3_baseline2/Runs/",
__C.YOLO.CLASSES                = "./data/classes/recce.names"
__C.YOLO.ANCHORS                = "./data/anchors/recce_anchors_2.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize" # interpolation
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"
__C.YOLO.IMAGE_HANDLE           = 'crop'  # 'crop' or 'scale'

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.INITIAL_WEIGHT        = "./checkpoint/yolov3_coco_demo.ckpt"
#__C.TRAIN.RESTORE_CHKPT          = "./checkpoint/yolov3_test_loss=14.7788.ckpt-2"
__C.TRAIN.RESTORE_CHKPT          = "./checkpoint/yolov3_coco_demo.ckpt"
__C.TRAIN.ANNOT_PATH            = "./data/dataset/recce_all_Tagging_1_2_img.txt"
__C.TRAIN.BATCH_SIZE            = 1
__C.TRAIN.IMAGE_H               = 1*640 #2464
__C.TRAIN.IMAGE_W               = 1*800 #3296
__C.TRAIN.DATA_AUG              = False
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 300
__C.TRAIN.SECOND_STAGE_EPOCHS   = 1000


# TEST options
__C.TEST                        = edict()

__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_epoch=69_test_loss=82.6984.ckpt-69"
__C.TEST.ANNOT_PATH             = "./data/dataset/recce_all_Tagging_1_2_img.txt"
__C.TEST.BATCH_SIZE             = 2
__C.TEST.IMAGE_H                = 1*640 #2464
__C.TEST.IMAGE_W                = 1*800 #3296
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45