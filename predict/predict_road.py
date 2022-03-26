import torch
import network
import numpy as np
from tqdm import tqdm
from PIL import Image
from pathlib import Path


def main():
    model_name = 'unet_small'
    ckpt = Path(r'checkpoints/mtss/best-mtss_resnet50-loss_0.2928-Acc_0.9008-IoU_0.7003-Epoch_42.pth')
    data_path = Path(r'F:\Dataset\tradition_villages_old\Segmentation\JPEGImages')
    save_dir = Path(r'F:\Dataset\results\tradition_villages_old')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 1
    crop_val = True
    crop_size = 500
    patch_size = 512
    img_size = (2448, 2448)

    # setup model
    # setup model
    model_map = {
        'unet': network.unet,
        'unet_small': network.unet_small,
        'dense_unet': network.dense_unet,
    }
    model = model_map[model_name]().to(device)

    if ckpt.is_file():
        print("=> loading checkpoint '{}".format(ckpt))
        checkpoint = torch.load(ckpt)
        model.load_state_dict(checkpoint["state_dict"], False)
    else:
        print("=> no checkpoint found at '{}'".format(ckpt))

    check_base_name = ckpt.stem
    save_subdir = save_dir / check_base_name
    if not save_subdir.is_dir():
        save_subdir.mkdir()

    model.eval()
    if data_path.is_dir():
        file_names_img = list(filter(lambda x: x.suffix == '.jpg', data_path.iterdir()))
    elif data_path.is_file():
        file_names_img = [data_path]
    else:
        raise ValueError('data path error! please check!')

    batch_num = len(file_names_img) // batch_size + 1

    # crop_val
    stride = crop_size * 1
    stride_idx_width = list(range(0, img_size[1], stride))
    stride_idx_height = list(range(0, img_size[0], stride))
    miro_margin = int((patch_size - crop_size) / 2)

    for batch_idx in tqdm(range(0, batch_num)):
        batch_img_path_list = file_names_img[batch_idx * batch_size:(batch_idx + 1) * batch_size]

        batch_img_list = [np.array(Image.open(path)) for path in batch_img_path_list]
        batch_img_array = np.array(batch_img_list)

        if not crop_val:
            batch_img_array = torch.Tensor(np.transpose(batch_img_array,
                                              axes=(0, 3, 1, 2)) / 255.0).to(device)
            outputs = model(batch_img_array).squeeze(dim=1)

            outputs_maps = outputs.cpu().numpy()

        else:
            batch_predict_test_maps = np.zeros((len(batch_img_path_list), img_size[0], img_size[1]))
            test_img_miro_array = np.pad(batch_img_array, pad_width=[(0, 0),
                                            (miro_margin, miro_margin),
                                            (miro_margin, miro_margin),
                                            (0, 0)], mode='reflect')

            for i, start_row in enumerate(stride_idx_height):
                for j, start_col in enumerate(stride_idx_height):
                    if start_row + crop_size > img_size[0]:
                        start_row = img_size[0] - crop_size
                    if start_col + crop_size > img_size[1]:
                        start_col = img_size[1] - crop_size

                    batch_patch_test_img = test_img_miro_array[:, start_row:start_row + patch_size,
                                                                     start_col:start_col + patch_size, :]

                    batch_patch_test_img = torch.Tensor(np.transpose(batch_patch_test_img,
                                                                        axes=(0, 3, 1, 2)) / 255.0).to(device)

                    outputs = model(batch_patch_test_img).squeeze(dim=1)
                    output_maps = outputs.data.cpu().numpy()
                    outputs_maps_crops = output_maps[:, miro_margin:miro_margin + crop_size,
                                                            miro_margin:miro_margin + crop_size]

                    batch_predict_test_maps[:, start_row:start_row + crop_size,
                                            start_col:start_col + crop_size] = outputs_maps_crops
            outputs_maps = batch_predict_test_maps

        for img_idx, img_path in enumerate(batch_img_path_list):
            save_path = save_subdir / img_path.name
            outputs_maps = outputs_maps[img_idx, :].astype('uint8')
            outputs_maps[outputs_maps > 0] = 255
            Image.fromarray(outputs_maps).save(save_path)
            print('saved predicted image {}'.format(img_path.name))


if __name__ == "__main__":
    main()

