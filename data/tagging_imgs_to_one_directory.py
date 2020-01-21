import os
from shutil import copyfile

##########################################################################################################
# This script takes all images from folders "Tagindg1" to "Tagging8" (uncropped images + casper data)
# and copies them to a joined destination folder of all images without sub folders
##########################################################################################################

main_db_dir = '/home/mayarap/tamar_pc_DB/DBs/Reccelite_iter1/Taggings'
dest_dir = '/home/mayarap/tamar_pc_DB/DBs/Reccelite_iter1/all_full_img'

for mid_dir in os.listdir(main_db_dir):
    if not (mid_dir.endswith('1') or mid_dir.endswith('2') or mid_dir.endswith('3') or mid_dir.endswith('4') or mid_dir.endswith('5') or mid_dir.endswith('6') or mid_dir.endswith('7') or mid_dir.endswith('8')):
        continue

    for img_dir in os.listdir(os.path.join(main_db_dir,mid_dir)):
        for f in  os.listdir(os.path.join(main_db_dir,mid_dir,img_dir)):
            if f.endswith('.jpg') or f.endswith('.png') or f.endswith('.jp2'):
                img_name = os.path.join(main_db_dir,mid_dir,img_dir,f)
                copyfile(img_name, os.path.join(dest_dir,f))
                print(f)
