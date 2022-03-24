import os
import utils
import torch
import network
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datasets.road import RoadSegm
from torchvision import transforms as T


def main():

    model_name = 'unet_small'   # choose model for training
    ckpt = Path('')
    data = Path('')             # path to the test dataset
    save_dir = Path('')         # path to the predicted results (default: ./test_predict)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    batch_size = 10             # batch size for prediction process (default: 10)
    crop_size = 80              # cropped size from the patch of prediction (default: 80)
    patch_size = 112            # patch size for image cropped from orig image (default: 112)')


    # Setup model
    model_map = {
        'mtss_resnet50': network.mtss_resnet50,
        'mtss_resnet101': network.mtss_resnet101,
        'mtss_mobilenet': network.mtss_mobilenet,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[model_name]
    if ckpt.is_file():
        print("=> loading checkpoint '{}".format(ckpt))
        checkpoint = torch.load(ckpt)
        model.load_state_dict
    else:
        print("=> no checkpoint found at '{}'".format(ckpt))

    check_base_name = ckpt.stem
    save_subdir = save_dir / check_base_name
    if not save_subdir.is_dir():
        save_subdir.mkdir()

    model.eval()
    if data.is_dir():
        fileNames_predict = list(filter(lambda x: x.endswith('.jpg'), data.iterdir()))
    elif data.is_file():
        fileNames_predict = [data]
    else:
        print("data path error! please check!")

    # stride = int(crop_size / 2)
    # stride_idx = list(range(0, 1024, stride))

    # miro_margin = int((patch_size-crop_size)/2)
    batch_num = len(fileNames_predict) // batch_size + 1

    for batch_idx in tqdm(range(0, batch_num)):
        batch_img_path_list = fileNames_predict[batch_idx*batch_size:(batch_idx+1)*batch_size]
        # batch_img_name_list = [path.name for path in batch_img_path_list]

        batch_img_list = [np.array(Image.open(path)) for path in batch_img_path_list]
        batch_img_array = np.array(batch_img_list)

        # for i, strt_row in enumerate(stride_idx):
        #     for j, start_col in enumerate(stride_idx):
        batch_imgs = torch.Tensor(np.transpose(batch_img_array, axes=(0, 3, 1, 2)) / 255.0).to(device)
        output_logist = model(batch_imgs)
        output_maps = np.squeeze(output_logist.item())

        for img_idx, img_path in enumerate(batch_img_path_list):
            save_path = save_subdir / img_path.name
            Image.fromarray(output_maps[img_idx, :]).save(save_path)
            print('saved predicted image {}'.format(img_path.name))


if __name__ == "__main__":
    main()
