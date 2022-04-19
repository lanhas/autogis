import os

import numpy as np
import scipy.misc
from tqdm import tqdm


def annotation2color(input_path, output_path):
    img = scipy.misc.imread(input_path)

    color = np.zeros((*(2448, 2448), 3), dtype=np.uint8)

    color[img == 0] = [0, 255, 255]         # 人造建筑      
    color[img == 1] = [255, 255, 0]         # 农田          
    color[img == 2] = [255, 0, 255]         # 牧场          
    color[img == 3] = [0, 255, 0]           # 森林          
    color[img == 4] = [0, 0, 255]           # 水系          
    color[img == 5] = [255, 255, 255]       # 荒地          
    color[img == 6] = [0, 0, 0]             # 未知区域      

    scipy.misc.imsave(output_path, color)
    pass


one_channel_label_path = 'datasets/train/landCover_voc/SegmentationClass'
test_mask_path = 'dataset/test/mask'

filelist = os.listdir(one_channel_label_path)
file_names = np.array([file.split('_')[0] for file in filelist if file.endswith('.png')], dtype=object)

for filename in tqdm(file_names):
    label_path = os.path.join(one_channel_label_path, filename + '_label.png')
    mask_path = os.path.join(test_mask_path, filename + '_mask.png')
    annotation2color(label_path, mask_path)
