import torch
import network
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datasets.mtsd import villageFactorsSegm

# -----------------------------------------------------------------#
#                     mask color                                   #
#       name        color          r       g       b     mask      #
#       未知        黑色            0       0       0       0       #
#       山体        红色           128      0       0       1       #
#       森林/植被   绿色            0      128      0       2       #
#       农田        黄色           128     128      0       3       #
#       水系        深蓝色         0        0       128     4       #
#       荒地        紫色           128      0       128     5       #
#       村落        浅蓝色          0      128      128     6       #
#------------------------------------------------------------------#


def main():
    model_name = 'mtss_resnet50'
    ckpt = Path(r'checkpoints/mtss/best-mtss_resnet50-loss_0.2928-Acc_0.9008-IoU_0.7003-Epoch_42.pth')
    remote_data = Path(r'F:\Dataset\tradition_villages_old\Segmentation\JPEGImages')
    dem_data = Path(r'F:\Dataset\tradition_villages_old\Segmentation\DEMImages')
    save_dir = Path(r'F:\Dataset\results\tradition_villages_old')

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'

    separable_conv = False  # apply separable conv to decoder and aspp
    batch_size = 1
    crop_size = 960
    patch_size = 1024
    output_stride = 16      # choices=[8, 16]
    num_classes = 7
    img_size = 2448

    # setup model
    model_map = {
        'mtss_resnet50': network.mtss_resnet50,
        'mtss_resnet101': network.mtss_resnet101,
        'mtss_mobilenet': network.mtss_mobilenet,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }

    model = model_map[model_name](num_classes=num_classes, output_stride=output_stride).to(device)
    if separable_conv:
        network.convert_to_separable_conv(model.classifier)

    if ckpt.is_file():
        print("=> loading checkpoint '{}".format(ckpt))
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint["state_dict"], False)
    else:
        print("=> no checkpoint found at '{}'".format(ckpt))

    # decode function
    decode_fn = villageFactorsSegm.decode_target

    check_base_name = ckpt.stem
    save_subdir = save_dir / check_base_name
    if not save_subdir.is_dir():
        save_subdir.mkdir()

    model.eval()
    if remote_data.is_dir() and dem_data.is_dir():
        file_names_remote = list(filter(lambda x: x.suffix == '.jpg', remote_data.iterdir()))
        file_names_dem = list(filter(lambda x: x.suffix == '.jpg', dem_data.iterdir()))
        if not len(file_names_remote) == len(file_names_dem):
            raise ValueError("data error!")
    elif remote_data.is_file() and dem_data.is_file():
        file_names_remote = [remote_data]
        file_names_dem = [dem_data]
    else:
        raise ValueError('data path error! please check!')

    # setup stride
    stride = int(crop_size / 2)
    stride_idx = list(range(0, img_size, stride))
    miro_margin = int((patch_size - crop_size) / 2)
    batch_num = len(file_names_remote) // batch_size + 1

    for batch_idx in tqdm(range(0, batch_num)):
        batch_remote_path_list = file_names_remote[batch_idx*batch_size:(batch_idx+1)*batch_size]
        batch_dem_path_list = file_names_dem[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        batch_remote_list = [np.array(Image.open(path)) for path in batch_remote_path_list]
        batch_remote_array = np.array(batch_remote_list)

        batch_dem_list = [np.array(Image.open(path)) for path in batch_dem_path_list]
        batch_dem_array = np.expand_dims(np.array(batch_dem_list), 3)

        batch_predict_test_maps = np.zeros((len(batch_remote_path_list), img_size, img_size))
        predict_test_mask = np.zeros((img_size, img_size))

        test_remote_miro_array = np.pad(batch_remote_array, pad_width=[(0, 0),
                                        (miro_margin, miro_margin),
                                        (miro_margin, miro_margin),
                                        (0, 0)], mode='reflect')
        test_dem_miro_array = np.pad(batch_dem_array, pad_width=[(0, 0),
                                     (miro_margin, miro_margin),
                                     (miro_margin, miro_margin),
                                     (0, 0)], mode='reflect')
        assert test_remote_miro_array.shape[1:] == (img_size + (patch_size - crop_size),
                                                    img_size + (patch_size - crop_size),
                                                    3)
        for i, start_row in enumerate(stride_idx):
            for j, start_col in enumerate(stride_idx):
                batch_temp_test_maps = np.zeros((len(batch_remote_path_list), img_size, img_size))
                temp_test_mask = np.zeros((img_size, img_size))
                if start_row + crop_size > img_size:
                    start_row = img_size - crop_size
                if start_col + crop_size > img_size:
                    start_col = img_size - crop_size

                batch_crop_test_remote = test_remote_miro_array[:, start_row:start_row + patch_size,
                                                                start_col:start_col + patch_size, :]
                batch_crop_test_dem = test_dem_miro_array[:, start_row:start_row + patch_size,
                                                          start_col:start_col + patch_size, :]

                batch_crop_test_remote = torch.Tensor(np.transpose(batch_crop_test_remote, axes=(0, 3, 1, 2)) / 255.0).to(device)
                batch_crop_test_dem = torch.Tensor(np.transpose(batch_crop_test_dem, axes=(0, 3, 1, 2)) / 255.0).to(device)

                if model_name[:4] == 'mtss':
                    output_logist = model(batch_crop_test_remote, batch_crop_test_dem).max(1)[1]
                else:
                    batch_imgs = torch.cat((batch_crop_test_remote, batch_crop_test_dem), 1)
                    output_logist = model(batch_imgs).max(1)[1]
                outputs_maps = output_logist.cpu().numpy()
                outputs_maps_crops = outputs_maps[:, miro_margin:miro_margin + crop_size,
                                               miro_margin:miro_margin + crop_size]

                batch_temp_test_maps[:, start_row:start_row + crop_size,
                                     start_col:start_col + crop_size] = outputs_maps_crops
                temp_test_mask[start_row:start_row+crop_size,
                               start_col:start_col+crop_size] = np.ones((crop_size, crop_size))

                batch_predict_test_maps = batch_predict_test_maps + batch_temp_test_maps
                predict_test_mask = predict_test_mask + temp_test_mask

        predict_test_mask = np.expand_dims(predict_test_mask, axis=0)
        batch_predict_test_maps = batch_predict_test_maps / predict_test_mask

        for img_idx, img_path in enumerate(batch_remote_path_list):
            save_path = save_subdir / img_path.name
            colorized_predict = decode_fn(batch_predict_test_maps[img_idx, :].astype('uint8'))
            Image.fromarray(colorized_predict).save(save_path)
            print('saved predicted image {}'.format(img_path.name))


if __name__ == "__main__":
    main()

