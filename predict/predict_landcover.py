import torch
import models.village_segm as models
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from labelme import utils


# -----------------------------------------------------------------#
#                     mask color                                   #
#       name        color          r       g       b     mask      #
#       未知        黑色            0       0       0       0       #
#       荒地        红色           128      0       0       1       #
#       森林        绿色            0      128      0       2       #
#       农田        黄色           128     128      0       3       #
#       水系        深蓝色         0        0       128     4       #
#       草地        紫色           128      0       128     5       #
#       村落        浅蓝色          0      128      128     6       #
# ------------------------------------------------------------------#


def main():
    model_name = 'deeplab-v3p'
    ckpt = Path('save/village_segm-deeplab-v3p_resnet50/max-vi.pth')
    remote_data = Path(r'F:\Dataset\tradition_villages\remote')
    save_dir = Path(r'F:\Dataset\results\tradition_villages')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    separable_conv = False  # apply separable conv to decoder and aspp
    batch_size = 1
    crop_val = True
    crop_size = 500
    patch_size = 512
    img_size = (2560, 2560)

    model_args = {'encoder': 'resnet50',
                  'encoder_args': {'output_stride': 16, 'pretrained': True},
                  'classifier_args': {'n_classes': 7}}

    model = models.make(model_name, **model_args)

    if separable_conv:
        models.convert_to_separable_conv(model.classifier)

    if ckpt.is_file():
        print("=> loading checkpoint '{}".format(ckpt))
        model = models.load(torch.load(ckpt))
    else:
        print("=> no checkpoint found at '{}'".format(ckpt))

    check_base_name = ckpt.stem
    save_subdir = save_dir / check_base_name
    if not save_subdir.is_dir():
        save_subdir.mkdir()

    model.eval()
    if remote_data.is_dir():
        file_names_remote = list(filter(lambda x: x.suffix == '.tif', remote_data.iterdir()))
    elif remote_data.is_file():
        file_names_remote = [remote_data]
    else:
        raise ValueError('data path error! please check!')

    batch_num = len(file_names_remote) // batch_size + 1

    # crop_val
    stride = crop_size * 1
    stride_idx_width = list(range(0, img_size[1], stride))
    stride_idx_height = list(range(0, img_size[0], stride))
    miro_margin = int((patch_size - crop_size) / 2)

    for batch_idx in tqdm(range(0, batch_num)):
        batch_remote_path_list = file_names_remote[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        batch_remote_list = [np.array(Image.open(path).convert('RGB')) for path in batch_remote_path_list]
        batch_remote_array = np.array(batch_remote_list)

        if not crop_val:
            batch_remote_array = torch.Tensor(np.transpose(batch_remote_array,
                                              axes=(0, 3, 1, 2)) / 255.0).to(device)
            outputs_maps = model(batch_remote_array).max(1)[1].cpu().detach().numpy()
        else:
            batch_predict_test_maps = np.zeros((len(batch_remote_path_list), img_size[0], img_size[1]))
            test_remote_miro_array = np.pad(batch_remote_array, pad_width=[(0, 0),
                                            (miro_margin, miro_margin),
                                            (miro_margin, miro_margin),
                                            (0, 0)], mode='reflect')

            for i, start_row in enumerate(stride_idx_height):
                for j, start_col in enumerate(stride_idx_height):
                    if start_row + crop_size > img_size[0]:
                        start_row = img_size[0] - crop_size
                    if start_col + crop_size > img_size[1]:
                        start_col = img_size[1] - crop_size

                    batch_patch_test_remote = test_remote_miro_array[:, start_row:start_row + patch_size,
                                                                     start_col:start_col + patch_size, :]

                    batch_patch_test_remote = torch.Tensor(np.transpose(batch_patch_test_remote,
                                                                        axes=(0, 3, 1, 2)) / 255.0).to(device)
                    output_logist = model(batch_patch_test_remote).softmax(dim=1)

                    outputs_maps_patch = output_logist.max(1)[1].cpu().detach().numpy()
                    outputs_maps_crops = outputs_maps_patch[:, miro_margin:miro_margin+crop_size,
                                                            miro_margin:miro_margin+crop_size]

                    batch_predict_test_maps[:, start_row:start_row + crop_size,
                                            start_col:start_col + crop_size] = outputs_maps_crops
            outputs_maps = batch_predict_test_maps

        for img_idx, img_path in enumerate(batch_remote_path_list):
            save_path = save_subdir / (img_path.stem + '.png')
            utils.lblsave(str(save_path), outputs_maps[img_idx, :].astype('uint8'))
            print('saved predicted image {}'.format(img_path.stem + '.png'))


if __name__ == "__main__":
    main()

