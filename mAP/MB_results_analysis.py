import glob
import json
import os
import shutil
import operator
import sys
import argparse
import cv2
from collections import defaultdict, OrderedDict
import numpy as np
import random
import pandas as pd
import seaborn as sn
import matplotlib
from matplotlib.backends.backend_pdf import PdfPages
import pylab as plt
import six

random.seed(0)
np.random.seed(0)

MINOVERLAP = 0.5  # default value (defined in the PASCAL VOC2012 challenge)

parser = argparse.ArgumentParser()
parser.add_argument('-na', '--no-animation', help="no animation is shown.", action="store_true")
parser.add_argument('-np', '--no-plot', help="no plot is shown.", action="store_true")
parser.add_argument('-q', '--quiet', help="minimalistic console output.", action="store_true")
# argparse receiving list of classes to be ignored
parser.add_argument('-i', '--ignore', nargs='+', type=str, help="ignore a list of classes.")
# argparse receiving list of classes with specific IoU
parser.add_argument('--set-class-iou', nargs='+', type=str, help="set IoU for a specific class.")
args = parser.parse_args()

# if there are no classes to ignore then replace None by empty list
if args.ignore is None:
  args.ignore = []

specific_iou_flagged = False
if args.set_class_iou is not None:
  specific_iou_flagged = True

# if there are no images then no animation can be shown
home_dir = os.path.expanduser('~')
if home_dir == '~':
  home_dir = ''
# If user or $HOME is unknown, do nothing
img_path = os.path.join(home_dir, 'DBs/Reccelite/Tagging_1_2_img') #'images'
if os.path.exists(img_path):
  for dirpath, dirnames, files in os.walk(img_path):
    if not files:
      # no image files found
      args.no_animation = True
else:
  args.no_animation = True

# try to import OpenCV if the user didn't choose the option --no-animation
show_animation = False
if not args.no_animation:
  try:
    import cv2
    show_animation = False
  except ImportError:
    print("\"opencv-python\" not found, please install to visualize the results.")
    args.no_animation = True

# try to import Matplotlib if the user didn't choose the option --no-plot
draw_plot = False
if not args.no_plot:
  try:
    import matplotlib.pyplot as plt
    draw_plot = True
  except ImportError:
    print("\"matplotlib\" not found, please install it to get the resulting plots.")
    args.no_plot = True

"""
 throw error and exit
"""
def error(msg):
  print(msg)
  sys.exit(0)

"""
 check if the number is a float between 0.0 and 1.0
"""
def is_float_between_0_and_1(value):
  try:
    val = float(value)
    if val > 0.0 and val < 1.0:
      return True
    else:
      return False
  except ValueError:
    return False

"""
 Calculate the AP given the recall and precision array
  1st) We compute a version of the measured precision/recall curve with
       precision monotonically decreasing
  2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
  """
  --- Official matlab code VOC2012---
  mrec=[0 ; rec ; 1];
  mpre=[0 ; prec ; 0];
  for i=numel(mpre)-1:-1:1
      mpre(i)=max(mpre(i),mpre(i+1));
  end
  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  rec.insert(0, 0.0) # insert 0.0 at begining of list
  rec.append(1.0) # insert 1.0 at end of list
  mrec = rec[:]
  prec.insert(0, 0.0) # insert 0.0 at begining of list
  prec.append(0.0) # insert 0.0 at end of list
  mpre = prec[:]
  """
   This part makes the precision monotonically decreasing
    (goes from the end to the beginning)
    matlab:  for i=numel(mpre)-1:-1:1
                mpre(i)=max(mpre(i),mpre(i+1));
  """
  # matlab indexes start in 1 but python in 0, so I have to do:
  #   range(start=(len(mpre) - 2), end=0, step=-1)
  # also the python function range excludes the end, resulting in:
  #   range(start=(len(mpre) - 2), end=-1, step=-1)
  for i in range(len(mpre)-2, -1, -1):
    mpre[i] = max(mpre[i], mpre[i+1])
  """
   This part creates a list of indexes where the recall changes
    matlab:  i=find(mrec(2:end)~=mrec(1:end-1))+1;
  """
  i_list = []
  for i in range(1, len(mrec)):
    if mrec[i] != mrec[i-1]:
      i_list.append(i) # if it was matlab would be i + 1
  """
   The Average Precision (AP) is the area under the curve
    (numerical integration)
    matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
  """
  ap = 0.0
  for i in i_list:
    ap += ((mrec[i]-mrec[i-1])*mpre[i])
  return ap, mrec, mpre


"""
 Convert the lines of a file to a list
"""
def file_lines_to_list(path):
  # open txt file lines to a list
  with open(path) as f:
    content = f.readlines()
  # remove whitespace characters like `\n` at the end of each line
  content = [x.strip() for x in content]
  return content

"""
 Draws text in image
"""
def draw_text_in_image(img, text, pos, color, line_width):
  font = cv2.FONT_HERSHEY_PLAIN
  fontScale = 1
  lineType = 1
  bottomLeftCornerOfText = pos
  cv2.putText(img, text,
      bottomLeftCornerOfText,
      font,
      fontScale,
      color,
      lineType)
  text_width, _ = cv2.getTextSize(text, font, fontScale, lineType)[0]
  return img, (line_width + text_width)

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
  # get text width for re-scaling
  bb = t.get_window_extent(renderer=r)
  text_width_inches = bb.width / fig.dpi
  # get axis width in inches
  current_fig_width = fig.get_figwidth()
  new_fig_width = current_fig_width + text_width_inches
  propotion = new_fig_width / current_fig_width
  # get axis limit
  x_lim = axes.get_xlim()
  axes.set_xlim([x_lim[0], x_lim[1]*propotion])


"""
 Find for a given bounding box a corresponding bounding box with maximal IOU out of list of bounding boxes 
"""
def find_bbox_with_max_IOU(src_bbox, dst_bbox_list):
  maxIOU = 0
  maxIOU_bbox = []

  src = [float(x) for x in src_bbox["bbox"].split()]
  for dst_bbox in dst_bbox_list:
    dst = [float(x) for x in dst_bbox["bbox"].split()]
    # calc iou, if larger than max replace
    bi = [max(src[0], dst[0]), max(src[1], dst[1]), min(src[2], dst[2]), min(src[3], dst[3])]  # intersection bb
    iw = bi[2] - bi[0] + 1  # intersection width
    ih = bi[3] - bi[1] + 1  # intersection height
    if iw > 0 and ih > 0:
      # compute overlap (IoU) = area of intersection / area of union
      ua = (src[2] - src[0] + 1) * (src[3] - src[1] + 1) + (dst[2] - dst[0]
                                                        + 1) * (dst[3] - dst[1] + 1) - iw * ih  # union area
      IOU = iw * ih / ua
      if IOU > maxIOU:
        maxIOU = IOU
        maxIOU_bbox = dst_bbox

  return maxIOU_bbox, maxIOU


"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
  # sort the dictionary by decreasing value, into a list of tuples
  sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
  # unpacking the list of tuples into two lists
  print(sorted_dic_by_value)
  sorted_keys, sorted_values = zip(*sorted_dic_by_value)
  #
  if true_p_bar != "":
    """
     Special case to draw in (green=true predictions) & (red=false predictions)
    """
    fp_sorted = []
    tp_sorted = []
    gt_sorted = []
    for key in sorted_keys:
      fp_sorted.append(dictionary[key] - true_p_bar[key])
      tp_sorted.append(true_p_bar[key])
      gt_sorted.append(gt_counter_per_class[key])
    plt.barh(range(n_classes), gt_sorted, align='center', color='forestgreen', label='True Predictions')
    plt.barh(range(n_classes), list(np.array(gt_sorted)-np.array(tp_sorted)), align='center', color='orange', label='Mis-Detections', left=gt_sorted)
    plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Predictions', left=list(2*np.array(gt_sorted)-np.array(tp_sorted)))
    # add legend
    plt.legend(loc='lower right')
    """
     Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
      fp_val = fp_sorted[i] # red
      gt_minus_tp_val = gt_sorted[i]-tp_sorted[i] # orange
      gt_val = gt_sorted[i] # green
      green_val = " " + str(gt_val)
      orange_val = green_val + " " + str(gt_minus_tp_val)
      red_val = orange_val + " " + str(fp_val)
      # trick to paint multicolor with offset:
      #   first paint everything and then repaint the first number
      t = plt.text(fp_val+gt_minus_tp_val+gt_val, i, red_val, color='crimson', va='center', fontweight='bold')
      plt.text(fp_val+gt_minus_tp_val+gt_val, i, orange_val, color='orange', va='center', fontweight='bold')
      plt.text(fp_val+gt_minus_tp_val+gt_val, i, green_val, color='forestgreen', va='center', fontweight='bold')

      if i == (len(sorted_values)-1): # largest bar
        adjust_axes(r, t, fig, axes)
  else:
    plt.barh(range(n_classes), sorted_values, color=plot_color)
    """
     Write number on side of bar
    """
    fig = plt.gcf() # gcf - get current figure
    axes = plt.gca()
    r = fig.canvas.get_renderer()
    for i, val in enumerate(sorted_values):
      str_val = " " + str(val) # add a space before
      if val < 1.0:
        str_val = " {0:.2f}".format(val)
      t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
      # re-set axes to show number inside the figure
      if i == (len(sorted_values)-1): # largest bar
        adjust_axes(r, t, fig, axes)
  # set window title
  fig.canvas.set_window_title(window_title)
  # write classes in y axis
  tick_font_size = 12
  plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
  """
   Re-scale height accordingly
  """
  init_height = fig.get_figheight()
  # comput the matrix height in points and inches
  dpi = fig.dpi
  height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
  height_in = height_pt / dpi
  # compute the required figure height
  top_margin = 0.15    # in percentage of the figure height
  bottom_margin = 0.05 # in percentage of the figure height
  figure_height = height_in / (1 - top_margin - bottom_margin)
  # set new height
  if figure_height > init_height:
    fig.set_figheight(figure_height)

  # set plot title
  plt.title(plot_title, fontsize=14)
  # set axis titles
  # plt.xlabel('classes')
  plt.xlabel(x_label, fontsize='large')
  # adjust size of window
  fig.tight_layout()
  # save the plot
  fig.savefig(output_path)
  # show image
  if to_show:
    plt.show()
  # close the plot
  plt.close()


# nadav - from magic box analysis script
def render_mpl_table(data, col_width=3.0, row_height=0.625, font_size=14,
                     header_color='#40466e', row_colors=['#f1f1f2','w'], edge_color='w',
                     bbox=[0,0,1,1], header_columns=0, ax=None, title=None, **kwargs):
  # turns a pandas dataframe to a plt.table
  for col in data.columns:
    data[col] = data[col].map('{:.2f}'.format)

  if ax is None:
    size = (np.array(data.shape[::-1])+np.array([1, 1]) * np.array([col_width, row_height]))
    fig, ax = plt.subplots(figsize=size)
    ax.axis('off')

  mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, rowLabels=data.index, **kwargs)
  mpl_table.auto_set_font_size(False)
  mpl_table.set_fontsize(font_size)


  for k,cell in six.iteritems(mpl_table._cells):
    cell.set_edgecolor(edge_color)
    if k[0] == 0 or k[1] < header_columns:
      cell.set_text_props(weight='bold', color = 'w')
      cell.set_facecolor(header_color)
    else:
      cell.set_facecolor(row_colors[k[0] % len(row_colors)])
    if title:
      plt.title(title)

  return fig

def maximize_figure(backend=None, fullscreen=False):
  """Maximize window independently on backend.
  Fullscreen sets fullscreen mode, that is same as maximized, but it doesn't have title bar (press key F to toggle full screen mode)."""
  if backend is None:
    backend = matplotlib.get_backend()
  mng = plt.get_current_fig_manager()

  if fullscreen:
    mng.full_screen_toggle()
  else:
    if backend == 'wxAgg':
      mng.frame.Maximize(True)
    elif backend == 'Qt4Agg' or backend == 'Qt5Agg':
      mng.window.showMaximized()
    elif backend == 'TkAgg':
      mng.resize(*mng.window.maxsize()) # works on ubuntu
      #mng.window.state('zoomed')  # works fine on Windows!
    else:
      print("Unrecognized backend: ", backend)  # not tested on different backends (only Qt)
  plt.show()

  plt.pause(0.1)  # this is needed to make sure following processing gets applied (e.g. tight_layout)





## main

"""
 Create a "tmp_files/" and "results/" directory
"""
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path): # if it doesn't exist already
  os.makedirs(tmp_files_path)
tmp_files_path = "tmp_files"
if not os.path.exists(tmp_files_path): # if it doesn't exist already
  os.makedirs(tmp_files_path)

results_files_path = "results"
if os.path.exists(results_files_path): # if it exist already
  # reset the results directory
  shutil.rmtree(results_files_path)

os.makedirs(results_files_path)
if draw_plot:
  os.makedirs(results_files_path + "/classes")
if show_animation:
  os.makedirs(results_files_path + "/images")
  os.makedirs(results_files_path + "/images/single_predictions")

"""
 Ground-Truth
   Load each of the ground-truth files into a temporary ".json" file.
   Create a list of all the class names present in the ground-truth (gt_classes).
"""
# get a list with the ground-truth files
ground_truth_files_list = glob.glob('ground-truth/*.txt')
if len(ground_truth_files_list) == 0:
  error("Error: No ground-truth files found!")
ground_truth_files_list.sort()
# dictionary with counter per class
gt_counter_per_class = {}

for txt_file in ground_truth_files_list:
  #print(txt_file)
  file_id = txt_file.split(".txt",1)[0]
  file_id = os.path.basename(os.path.normpath(file_id))
  # check if there is a correspondent predicted objects file
  if not os.path.exists('predicted/' + file_id + ".txt"):
    error_msg = "Error. File not found: predicted/" +  file_id + ".txt\n"
    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
    error(error_msg)
  lines_list = file_lines_to_list(txt_file)
  # create ground-truth dictionary
  bounding_boxes = []
  is_difficult = False
  for line in lines_list:
    try:
      if "difficult" in line:
          class_name, left, top, right, bottom, _difficult = line.split()
          is_difficult = True
      else:
          class_name, left, top, right, bottom = line.split()
    except ValueError:
      error_msg = "Error: File " + txt_file + " in the wrong format.\n"
      error_msg += " Expected: <class_name> <left> <top> <right> <bottom> ['difficult']\n"
      error_msg += " Received: " + line
      error_msg += "\n\nIf you have a <class_name> with spaces between words you should remove them\n"
      error_msg += "by running the script \"remove_space.py\" or \"rename_class.py\" in the \"extra/\" folder."
      error(error_msg)
    # check if class is in the ignore list, if yes skip
    if class_name in args.ignore:
      continue
    bbox = left + " " + top + " " + right + " " +bottom
    if is_difficult:
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False, "difficult":True})
        is_difficult = False
    else:
        bounding_boxes.append({"class_name":class_name, "bbox":bbox, "used":False})
        # count that object
        if class_name in gt_counter_per_class:
          gt_counter_per_class[class_name] += 1
        else:
          # if class didn't exist yet
          gt_counter_per_class[class_name] = 1
  # dump bounding_boxes into a ".json" file
  with open(tmp_files_path + "/" + file_id + "_ground_truth.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)

gt_classes = list(gt_counter_per_class.keys())
# let's sort the classes alphabetically
gt_classes = sorted(gt_classes)
n_classes = len(gt_classes)
#print(gt_classes)
#print(gt_counter_per_class)

"""
 Check format of the flag --set-class-iou (if used)
  e.g. check if class exists
"""
if specific_iou_flagged:
  n_args = len(args.set_class_iou)
  error_msg = \
    '\n --set-class-iou [class_1] [IoU_1] [class_2] [IoU_2] [...]'
  if n_args % 2 != 0:
    error('Error, missing arguments. Flag usage:' + error_msg)
  # [class_1] [IoU_1] [class_2] [IoU_2]
  # specific_iou_classes = ['class_1', 'class_2']
  specific_iou_classes = args.set_class_iou[::2] # even
  # iou_list = ['IoU_1', 'IoU_2']
  iou_list = args.set_class_iou[1::2] # odd
  if len(specific_iou_classes) != len(iou_list):
    error('Error, missing arguments. Flag usage:' + error_msg)
  for tmp_class in specific_iou_classes:
    if tmp_class not in gt_classes:
          error('Error, unknown class \"' + tmp_class + '\". Flag usage:' + error_msg)
  for num in iou_list:
    if not is_float_between_0_and_1(num):
      error('Error, IoU must be between 0.0 and 1.0. Flag usage:' + error_msg)

"""
 Predicted
   Load each of the predicted files into a temporary ".json" file.
"""
# get a list with the predicted files
predicted_files_list = glob.glob('predicted/*.txt')
predicted_files_list.sort()

for txt_file in predicted_files_list:
  #print(txt_file)
  file_id = txt_file.split(".txt",1)[0]
  file_id = os.path.basename(os.path.normpath(file_id))
  # check if there is a correspondent gt objects file
  if not os.path.exists('ground-truth/' + file_id + ".txt"):
    error_msg = "Error. File not found: ground-truth/" + file_id + ".txt\n"
    error_msg += "(You can avoid this error message by running extra/intersect-gt-and-pred.py)"
    error(error_msg)
  lines_list = file_lines_to_list(txt_file)
  # create ground-truth dictionary
  bounding_boxes = []
  is_difficult = False
  for line in lines_list:
    try:
      class_name, confidence, left, top, right, bottom = line.split()
    except ValueError:
      error_msg = "Error: File " + txt_file + " in the wrong format.\n"
      error_msg += " Expected: <class_name> <confidence> <left> <top> <right> <bottom>\n"
      error_msg += " Received: " + line
      error(error_msg)
    # check if class is in the ignore list, if yes skip
    bbox = left + " " + top + " " + right + " " +bottom
    bounding_boxes.append({"confidence": confidence, "class_name":class_name, "bbox": bbox})

  # dump bounding_boxes into a ".json" file
  with open(tmp_files_path + "/" + file_id + "_predictions.json", 'w') as outfile:
    json.dump(bounding_boxes, outfile)





# at this point we have GT json and predictions json for each image
# nadav
# create two dictionarties sorted by frames for the gt and the predictions data
predictions_data = {}
gt_data = {}
frame_list = []

for frameFile in os.listdir(tmp_files_path):
  """
     Load predictions and gt of each image
  """
  if frameFile.endswith("_predictions.json"):
    frameID = frameFile.split("_predictions.json")[0]
    frame_list.append(frameID)
    pred_frameFile = tmp_files_path + '/' + frameFile
    predictions_data[frameID] = json.load(open(pred_frameFile))
    gt_frameFile = tmp_files_path + '/' + frameFile.replace("_predictions", "_ground_truth")
    gt_data[frameID] = json.load(open(gt_frameFile))


# lists of zeros length of num_class
template_class_dict = {}
for gt_class in gt_classes:
  template_class_dict[gt_class] = 0

a = 1
TP = dict(template_class_dict)  # True positive predictions per class (correct class predicted)
FA = dict(template_class_dict)  # False Alarm predictions per class   (no gt object with IOU higher than threshold)
predictions_occurrences = dict(template_class_dict)
gt_occurrences = dict(template_class_dict)
miss_detections = dict(template_class_dict)


confusion_matrix = dict(template_class_dict)
for class_name in confusion_matrix:
  confusion_matrix[class_name] = dict(template_class_dict)


# iterate over the predictions, for each predictions bbox:
# if there is no corresponding gt bbox (of any class) mark as FA (False Alarm)
# if the corresponding gt bbox is of the same class mark as TP (True Positive)
# save FAs and TPs sorted by class
# possible to add later - FA per class depending on size range
for frameID in predictions_data:
  possible_gt_bbox_list = gt_data[frameID]
  for bbox_pred in predictions_data[frameID]:
    predictions_occurrences[bbox_pred['class_name']] += 1
    maxIOU_gt_bbox, IOU = find_bbox_with_max_IOU(bbox_pred, possible_gt_bbox_list)
    if IOU < MINOVERLAP:  # if IOU smaller than threshold the prediction is a FA
      FA[bbox_pred['class_name']] += 1
    elif bbox_pred['class_name'] == maxIOU_gt_bbox['class_name']:
      # if IOU larger than threshold and correct class is predicted than it's TP
      TP[bbox_pred['class_name']] += 1


# iterate over the ground truth, for each gt bbox:
# if there is no corresponding prediction bbox (of any class) mark as miss detection
# if the bbox was detected, save the prediction in the confusion matrix
for frameID in gt_data:
  possible_pred_bbox_list = predictions_data[frameID]
  for bbox_gt in gt_data[frameID]:
    gt_occurrences[bbox_gt['class_name']] += 1
    maxIOU_pred_bbox, IOU = find_bbox_with_max_IOU(bbox_gt, possible_pred_bbox_list)
    if IOU < MINOVERLAP:  # if IOU smaller than threshold the prediction is a False Alarm
      miss_detections[bbox_gt['class_name']] += 1
    else:
      confusion_matrix[bbox_gt['class_name']][maxIOU_pred_bbox['class_name']] += 1.0

# normalize confusion matrix to [0,1] (per gt class)
for gt_class_name in confusion_matrix:
  num_detected = gt_occurrences[gt_class_name] - miss_detections[gt_class_name]
  if num_detected != 0:
    for prediction_class_name in confusion_matrix:
      confusion_matrix[gt_class_name][prediction_class_name] = float("{0:.3f}".format(confusion_matrix[gt_class_name][prediction_class_name]/float(num_detected)))

# plot to pdf
figs = []
report_path = './mAP'
# predictions info figures
df = pd.DataFrame()
df['Occurrences'] = pd.Series(predictions_occurrences)
df['FA'] = pd.Series(FA)
df['FA [%]'] = df['FA'].div(other=df['Occurrences'])
df['TP'] = pd.Series(TP)
df['TP [%]'] = df['TP'].div(other=df['Occurrences'])
df['Misclassified'] = df['Occurrences'].sub(other=df['FA']).sub(other=df['TP'])
df['Misclassified [%]'] = df['Misclassified'].div(other=df['Occurrences'])

# graph figure
fig = df.plot(kind='bar', stacked=True, y=['TP', 'FA', 'Misclassified'], title='predictions data').get_figure()
#fig = df.plot(kind='bar', stacked=True, y=['TP [%]', 'FA [%]', 'Misclassified [%]'], title='predictions data').get_figure()
maximize_figure()
figs.append(fig)
plt.close()
# table figure
fig = render_mpl_table(df,title='predictions data')
maximize_figure()
figs.append(fig)
plt.close()


# gt info figure
df = pd.DataFrame()
df['Occurrences'] = pd.Series(gt_occurrences)
df['Not Detected'] = pd.Series(miss_detections)
df['Not Detected [%]'] = df['Not Detected'].div(other=df['Occurrences'])

# graph figure
fig = df.plot(kind='bar',y=['Occurrences', 'Not Detected'], title='ground truth data').get_figure()
#fig = df.plot(kind='bar',y=['Not Detected [%]'], title='ground truth data').get_figure()
maximize_figure()
figs.append(fig)
plt.close()
# table figure
fig = render_mpl_table(df,title='ground truth data')
maximize_figure()
figs.append(fig)
plt.close()
# confusion matrix figure
df = pd.DataFrame()
for gt_cls in sorted(confusion_matrix):
    gt_class_data = pd.Series(confusion_matrix[gt_cls])
    cls_index = gt_cls + ' (' + str(gt_occurrences[gt_cls] - miss_detections[gt_cls]) + ')'  # number of detected bbox
    df[cls_index] = gt_class_data
fig = sn.set()
sn.heatmap(df, annot=True)
maximize_figure()
figs.append(fig)
with PdfPages(report_path) as pdf:
  for fig in figs:
    pdf.savefig(fig, bbox_inches='tight')
plt.close('all')

# sn.catplot(data=df)
#
# indices = ['FA', 'TP']
# df = pd.DataFrame(index=indices)
# for class_name in sorted(gt_classes):
#   data = [FA[class_name], TP[class_name]]
#   df[class_name] = pd.Series(data=data, index=indices)
#
# sn.catplot(data=df)















# From Tamars' main



# """
#  Calculate the AP for each class
# """
# sum_AP = 0.0
# ap_dictionary = {}
# conf_T = defaultdict(list)
# conf_F = defaultdict(list)
# # open file to store the results
# with open(results_files_path + "/results.txt", 'w') as results_file:
#   results_file.write("# AP and precision/recall per class\n")
#   count_true_positives = {}
#   for class_index, class_name in enumerate(gt_classes):
#     count_true_positives[class_name] = 0
#     """
#      Load predictions of that class
#     """
#     predictions_file = tmp_files_path + "/" + class_name + "_predictions.json"
#     predictions_data = json.load(open(predictions_file))
#
#     """
#      Assign predictions to ground truth objects
#     """
#     nd = len(predictions_data)
#     tp = [0] * nd # creates an array of zeros of size nd
#     fp = [0] * nd
#     for idx, prediction in enumerate(predictions_data):
#       file_id = prediction["file_id"]
#       if show_animation:
#         # find ground truth image
#         ground_truth_img = glob.glob1(img_path, file_id + ".*")
#         #tifCounter = len(glob.glob1(myPath,"*.tif"))
#         print('\nimg_path =',img_path, '; ground_truth_img =',ground_truth_img, '; class_name =',class_name)
#         if len(ground_truth_img) == 0:
#           error("Error. Image not found with id: " + file_id)
#         elif len(ground_truth_img) > 1:
#           error("Error. Multiple image with id: " + file_id)
#         else: # found image
#           #print(img_path + "/" + ground_truth_img[0])
#           # Load image
#           img = cv2.imread(img_path + "/" + ground_truth_img[0])
#           # load image with draws of multiple detections
#           img_cumulative_path = results_files_path + "/images/" + ground_truth_img[0]
#           if os.path.isfile(img_cumulative_path):
#             img_cumulative = cv2.imread(img_cumulative_path)
#           else:
#             img_cumulative = img.copy()
#           # Add bottom border to image
#           bottom_border = 60
#           BLACK = [0, 0, 0]
#           img = cv2.copyMakeBorder(img, 0, bottom_border, 0, 0, cv2.BORDER_CONSTANT, value=BLACK)
#       # assign prediction to ground truth object if any
#       #   open ground-truth with that file_id
#       gt_file = tmp_files_path + "/" + file_id + "_ground_truth.json"
#       ground_truth_data = json.load(open(gt_file))
#       ovmax = -1
#       gt_match = -1
#       # load prediction bounding-box
#       bb = [ float(x) for x in prediction["bbox"].split() ]
#       for obj in ground_truth_data:
#         # look for a class_name match
#         if obj["class_name"] == class_name:
#           bbgt = [ float(x) for x in obj["bbox"].split() ]
#           bi = [max(bb[0],bbgt[0]), max(bb[1],bbgt[1]), min(bb[2],bbgt[2]), min(bb[3],bbgt[3])]
#           iw = bi[2] - bi[0] + 1
#           ih = bi[3] - bi[1] + 1
#           if iw > 0 and ih > 0:
#             # compute overlap (IoU) = area of intersection / area of union
#             ua = (bb[2] - bb[0] + 1) * (bb[3] - bb[1] + 1) + (bbgt[2] - bbgt[0]
#                     + 1) * (bbgt[3] - bbgt[1] + 1) - iw * ih
#             ov = iw * ih / ua
#             if ov > ovmax:
#               ovmax = ov
#               gt_match = obj
#
#       # assign prediction as true positive/don't care/false positive
#       if show_animation:
#         status = "NO MATCH FOUND!" # status is only used in the animation
#       # set minimum overlap
#       min_overlap = MINOVERLAP
#       if specific_iou_flagged:
#         if class_name in specific_iou_classes:
#           index = specific_iou_classes.index(class_name)
#           min_overlap = float(iou_list[index])
#       if ovmax >= min_overlap:
#         if "difficult" not in gt_match:
#             if not bool(gt_match["used"]):
#               # true positive
#               tp[idx] = 1
#               conf_T[class_name].append(float(prediction['confidence']))
#               gt_match["used"] = True
#               count_true_positives[class_name] += 1
#               # update the ".json" file
#               with open(gt_file, 'w') as f:
#                   f.write(json.dumps(ground_truth_data))
#               if show_animation:
#                 status = "MATCH!"
#             else:
#               # false positive (multiple detection)
#               fp[idx] = 1
#               if show_animation:
#                 status = "REPEATED MATCH!"
#       else:
#         # false positive
#         conf_F[class_name].append(float(prediction['confidence']))
#         fp[idx] = 1
#         if ovmax > 0:
#           status = "INSUFFICIENT OVERLAP"
#
#       """
#        Draw image to show animation
#       """
#       if show_animation:
#         height, widht = img.shape[:2]
#         # colors (OpenCV works with BGR)
#         white = (255,255,255)
#         light_blue = (255,200,100)
#         green = (0,255,0)
#         light_red = (30,30,255)
#         # 1st line
#         margin = 10
#         v_pos = int(height - margin - (bottom_border / 2))
#         text = "Image: " + ground_truth_img[0] + " "
#         img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
#         text = "Class [" + str(class_index) + "/" + str(n_classes) + "]: " + class_name + " "
#         img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), light_blue, line_width)
#         if ovmax != -1:
#           color = light_red
#           if status == "INSUFFICIENT OVERLAP":
#             text = "IoU: {0:.2f}% ".format(ovmax*100) + "< {0:.2f}% ".format(min_overlap*100)
#           else:
#             text = "IoU: {0:.2f}% ".format(ovmax*100) + ">= {0:.2f}% ".format(min_overlap*100)
#             color = green
#           img, _ = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
#         # 2nd line
#         v_pos += int(bottom_border / 2)
#         rank_pos = str(idx+1) # rank position (idx starts at 0)
#         text = "Prediction #rank: " + rank_pos + " confidence: {0:.2f}% ".format(float(prediction["confidence"])*100)
#         img, line_width = draw_text_in_image(img, text, (margin, v_pos), white, 0)
#         color = light_red
#         if status == "MATCH!":
#           color = green
#         text = "Result: " + status + " "
#         img, line_width = draw_text_in_image(img, text, (margin + line_width, v_pos), color, line_width)
#
#         font = cv2.FONT_HERSHEY_SIMPLEX
#         if ovmax > 0: # if there is intersections between the bounding-boxes
#           bbgt = [ int(float(x)) for x in gt_match["bbox"].split() ]
#           cv2.rectangle(img,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
#           cv2.rectangle(img_cumulative,(bbgt[0],bbgt[1]),(bbgt[2],bbgt[3]),light_blue,2)
#           cv2.putText(img_cumulative, class_name, (bbgt[0],bbgt[1] - 5), font, 0.6, light_blue, 1, cv2.LINE_AA)
#         bb = [int(i) for i in bb]
#         cv2.rectangle(img,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
#         cv2.rectangle(img_cumulative,(bb[0],bb[1]),(bb[2],bb[3]),color,2)
#         cv2.putText(img_cumulative, class_name, (bb[0],bb[1] - 5), font, 0.6, color, 1, cv2.LINE_AA)
#         # show image
#         cv2.imshow("Animation", img)
#         cv2.waitKey(20) # show for 20 ms
#         # save image to results
#         output_img_path = results_files_path + "/images/single_predictions/" + class_name + "_prediction" + str(idx) + ".jpg"
#         cv2.imwrite(output_img_path, img)
#         # save the image with all the objects drawn to it
#         cv2.imwrite(img_cumulative_path, img_cumulative)
#
#     #print(tp)
#     # compute precision/recall
#     cumsum = 0
#     for idx, val in enumerate(fp):
#       fp[idx] += cumsum
#       cumsum += val
#     cumsum = 0
#     for idx, val in enumerate(tp):
#       tp[idx] += cumsum
#       cumsum += val
#     #print(tp)
#     rec = tp[:]
#     for idx, val in enumerate(tp):
#       rec[idx] = float(tp[idx]) / gt_counter_per_class[class_name]
#     #print(rec)
#     prec = tp[:]
#     for idx, val in enumerate(tp):
#       prec[idx] = float(tp[idx]) / (fp[idx] + tp[idx])
#     #print(prec)
#
#     ap, mrec, mprec = voc_ap(rec, prec)
#     sum_AP += ap
#     text = "{0:.2f}%".format(ap*100) + " = " + class_name + " AP  " #class_name + " AP = {0:.2f}%".format(ap*100)
#     """
#      Write to results.txt
#     """
#     rounded_prec = [ '%.2f' % elem for elem in prec ]
#     rounded_rec = [ '%.2f' % elem for elem in rec ]
#     results_file.write(text + "\n Precision: " + str(rounded_prec) + "\n Recall   :" + str(rounded_rec) + "\n\n")
#     if not args.quiet:
#       print(text)
#     ap_dictionary[class_name] = ap
#
#     """
#      Draw plot
#     """
#     if draw_plot:
#       plt.plot(rec, prec, '-o')
#       # add a new penultimate point to the list (mrec[-2], 0.0)
#       # since the last line segment (and respective area) do not affect the AP value
#       area_under_curve_x = mrec[:-1] + [mrec[-2]] + [mrec[-1]]
#       area_under_curve_y = mprec[:-1] + [0.0] + [mprec[-1]]
#       plt.fill_between(area_under_curve_x, 0, area_under_curve_y, alpha=0.2, edgecolor='r')
#       # set window title
#       fig = plt.gcf() # gcf - get current figure
#       fig.canvas.set_window_title('AP ' + class_name)
#       # set plot title
#       plt.title('class: ' + text)
#       #plt.suptitle('This is a somewhat long figure title', fontsize=16)
#       # set axis titles
#       plt.xlabel('Recall')
#       plt.ylabel('Precision')
#       # optional - set axes
#       axes = plt.gca() # gca - get current axes
#       axes.set_xlim([0.0,1.0])
#       axes.set_ylim([0.0,1.05]) # .05 to give some extra space
#       # Alternative option -> wait for button to be pressed
#       #while not plt.waitforbuttonpress(): pass # wait for key display
#       # Alternative option -> normal display
#       #plt.show()
#       # save the plot
#       fig.savefig(results_files_path + "/classes/mAP of " + class_name + ".png")
#       plt.cla() # clear axes for next plot
#
#   if show_animation:
#     cv2.destroyAllWindows()
#
#   results_file.write("\n# mAP of all classes\n")
#   mAP = sum_AP / n_classes
#   text = "mAP = {0:.2f}%".format(mAP*100)
#   results_file.write(text + "\n")
#   print(text)
#
# # remove the tmp_files directory
# shutil.rmtree(tmp_files_path)
#
# """
#  Count total of Predictions
# """
# # iterate through all the files
# pred_counter_per_class = {}
# #all_classes_predicted_files = set([])
# for txt_file in predicted_files_list:
#   # get lines to list
#   lines_list = file_lines_to_list(txt_file)
#   for line in lines_list:
#     class_name = line.split()[0]
#     # check if class is in the ignore list, if yes skip
#     if class_name in args.ignore:
#       continue
#     # count that object
#     if class_name in pred_counter_per_class:
#       pred_counter_per_class[class_name] += 1
#     else:
#       # if class didn't exist yet
#       pred_counter_per_class[class_name] = 1
# #print(pred_counter_per_class)
# pred_classes = list(pred_counter_per_class.keys())
#
#
# """
#  Plot the total number of occurences of each class in the ground-truth
# """
# if draw_plot:
#   window_title = "Ground-Truth Info"
#   plot_title = "Ground-Truth\n"
#   plot_title += "(" + str(len(ground_truth_files_list)) + " files and " + str(n_classes) + " classes)"
#   x_label = "Number of objects per class"
#   output_path = results_files_path + "/Ground-Truth Info.png"
#   to_show = False
#   plot_color = 'forestgreen'
#   draw_plot_func(
#     gt_counter_per_class,
#     n_classes,
#     window_title,
#     plot_title,
#     x_label,
#     output_path,
#     to_show,
#     plot_color,
#     '',
#     )
#
#
# """
#  Write number of ground-truth objects per class to results.txt
# """
# with open(results_files_path + "/results.txt", 'a') as results_file:
#   results_file.write("\n# Number of ground-truth objects per class\n")
#   for class_name in sorted(gt_counter_per_class):
#     results_file.write(class_name + ": " + str(gt_counter_per_class[class_name]) + "\n")
#
# """
#  Finish counting true positives
# """
# for class_name in pred_classes:
#   # if class exists in predictions but not in ground-truth then there are no true positives in that class
#   if class_name not in gt_classes:
#     count_true_positives[class_name] = 0
# #print(count_true_positives)
#
# """
#  Plot the total number of occurences of each class in the "predicted" folder
#  Plot the amount of true/false positives Vs. Confidence - per class and total
#
# """
# if draw_plot:
#   window_title = "Predicted Objects Info"
#   # Plot title
#   plot_title = "Predicted Objects\n"
#   plot_title += "(" + str(len(predicted_files_list)) + " files and "
#   count_non_zero_values_in_dictionary = sum(int(x) > 0 for x in list(pred_counter_per_class.values()))
#   plot_title += str(count_non_zero_values_in_dictionary) + " detected classes)"
#   # end Plot title
#   x_label = "Number of objects per class"
#   output_path = results_files_path + "/Predicted Objects Info.png"
#   to_show = False
#   plot_color = 'forestgreen'
#   true_p_bar = count_true_positives
#   draw_plot_func(
#     pred_counter_per_class,
#     len(pred_counter_per_class),
#     window_title,
#     plot_title,
#     x_label,
#     output_path,
#     to_show,
#     plot_color,
#     true_p_bar
#     )
#
#   for cls in pred_counter_per_class.keys():
#     print('\n'+cls)
#     if len(conf_T[cls] + conf_T[cls]) == 0: continue
#     output_path = results_files_path + "/Confidence reliability of " + cls + ".png"
#     #fig, ax = plt.subplots()
#     n_equal_bins = max(len(conf_T[cls] + conf_T[cls]), int(len(conf_T[cls] + conf_T[cls])/15))
#     print(n_equal_bins)
#     opacity = 0.75
#     plt.hist(conf_T[cls], n_equal_bins, alpha=opacity, color='g', label='True Predictions')
#     plt.hist(conf_F[cls], n_equal_bins, alpha=opacity, color='r', label='False Predictions')
#     plt.grid(axis='y', alpha=0.75)
#     plt.xlabel('Confidence', fontsize='large')
#     plt.ylabel('Amount of Predictions', fontsize='large')
#     plot_title = "Relevancy of Confidence\n"
#     plot_title += str(pred_counter_per_class[cls]) + " predictions, "
#     plot_title += "Class " + cls
#     plt.title(plot_title, fontsize=14)
#     plt.xlim(0, 1)
#     handles, labels = plt.gca().get_legend_handles_labels()
#     by_label = OrderedDict(zip(labels, handles))
#     plt.legend(by_label.values(), by_label.keys())
#     plt.tight_layout()
#     plt.savefig(output_path)
#     #plt.show()
#
#
# """
#  Write number of predicted objects per class to results.txt
# """
# with open(results_files_path + "/results.txt", 'a') as results_file:
#   results_file.write("\n# Number of predicted objects per class\n")
#   for class_name in sorted(pred_classes):
#     n_pred = pred_counter_per_class[class_name]
#     text = class_name + ": " + str(n_pred)
#     text += " (tp:" + str(count_true_positives[class_name]) + ""
#     text += ", fp:" + str(n_pred - count_true_positives[class_name]) + ")\n"
#     results_file.write(text)
#
# """
#  Draw mAP plot (Show AP's of all classes in decreasing order)
# """
# if draw_plot:
#   plt.figure()
#   window_title = "mAP"
#   plot_title = "mAP = {0:.2f}%".format(mAP*100)
#   x_label = "Average Precision"
#   output_path = results_files_path + "/mAP.png"
#   to_show = True
#   plot_color = 'royalblue'
#   draw_plot_func(
#     ap_dictionary,
#     n_classes,
#     window_title,
#     plot_title,
#     x_label,
#     output_path,
#     to_show,
#     plot_color,
#     ""
#     )
