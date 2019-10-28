from easydict import EasyDict as edict

__C                             = edict()
# Consumers can get config by: from config import cfg

cfg                             = __C

# YOLO options
__C.YOLO                        = edict()

# Set the class name
__C.YOLO.EXP_DIR                = "recce_anchors1-5",
__C.YOLO.ROOT_DIR               = "./Runs/",
__C.YOLO.CLASSES                = "./data/classes/recce.names"
__C.YOLO.ANCHORS                = "./data/anchors/anchors_1-5_cropped.txt"
__C.YOLO.MOVING_AVE_DECAY       = 0.9995
__C.YOLO.STRIDES                = [8, 16, 32]
__C.YOLO.ANCHOR_PER_SCALE       = 3
__C.YOLO.IOU_LOSS_THRESH        = 0.5
__C.YOLO.UPSAMPLE_METHOD        = "resize" # "resize" # "deconv"
__C.YOLO.ORIGINAL_WEIGHT        = "./checkpoint/yolov3_coco.ckpt"
__C.YOLO.DEMO_WEIGHT            = "./checkpoint/yolov3_coco_demo.ckpt"
__C.YOLO.IMAGE_HANDLE           = 'crop'  # 'crop' or 'scale'
__C.YOLO.DATA_FORMAT            = 'NHWC'  # 'NCHW'(=opt for RT) or 'NHWC'(=default) (N-batch size, C-channels, H-height, W-width)
__C.YOLO.EPILOG_LOGICS          = True
__C.YOLO.CONF_TH_FILE           = "./data/classes/recce.confidence_th.txt"

# Train options
__C.TRAIN                       = edict()

__C.TRAIN.RESTORE_CHKPT          = "./checkpoint/yolov3_coco_demo.ckpt"
__C.TRAIN.ANNOT_PATH            = "/home/tamar/DBs/Reccelite/CroppedDB/croppedImgs_1_2_3_4_5_Th06_reg_rare.txt"
__C.TRAIN.BATCH_SIZE            = 2
__C.TRAIN.IMAGE_H               = 1*640 #1*640#2464
__C.TRAIN.IMAGE_W               = 1*800 #1*800#3296
__C.TRAIN.DATA_AUG              = False
__C.TRAIN.LEARN_RATE_INIT       = 1e-4
__C.TRAIN.LEARN_RATE_END        = 1e-6
__C.TRAIN.WARMUP_EPOCHS         = 2
__C.TRAIN.FISRT_STAGE_EPOCHS    = 20
__C.TRAIN.SECOND_STAGE_EPOCHS   = 150
__C.TRAIN.WEIGHTED_LOSS         = True
__C.TRAIN.WEIGHTED_LOSS_MAP     = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                   1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # weight vector with length=number_of_classes
__C.TRAIN.PRUNING_EPOCH_FREQ    = 100
__C.TRAIN.MAX_TO_KEEP           = 100

# TEST options
__C.TEST                        = edict()
__C.TEST.WEIGHT_FILE            = "./checkpoint/yolov3_epoch=53_test_loss=21.5429.ckpt-53"
# if USE_WEIGHTS_DIR is False - WEIGHT_FILE is used, otherwise the latest file in the WEIGHT_DIR will be used
__C.TEST.WEIGHT_DIR             = "../Runs/recce__trt_compatibility_23Oct19_1720_1b15729/checkpoints/"
__C.TEST.USE_WEIGHTS_DIR        = True
__C.TEST.ANNOT_PATH            = '/home/tamar/DBs/Reccelite/CroppedDB/croppedImgs_3_4_5_Th06_reg.txt' #cropped_1_2_3_5_Th06_reg_rare.txt'
__C.TEST.BATCH_SIZE             = 1
__C.TEST.IMAGE_H                = 640 #1*640#2464
__C.TEST.IMAGE_W                = 800 #1*800#3296
__C.TEST.DATA_AUG               = False
__C.TEST.WRITE_IMAGE            = True
__C.TEST.WRITE_IMAGE_PATH       = "./data/detection/"
__C.TEST.WRITE_IMAGE_SHOW_LABEL = True
__C.TEST.SHOW_LABEL             = True
__C.TEST.SCORE_THRESHOLD        = 0.3
__C.TEST.IOU_THRESHOLD          = 0.45
