import os
import utils
import torch
import network
import torch.nn as nn
from tqdm import tqdm
from PIL import Image
from pathlib import Path
from datasets.road import RoadSegm
from torchvision import transforms as T


ckpt = Path.cwd() / 'checkpoints/mtss/mtss_resnet50.pth'     # resume from checkpoint
mode = 'predict'        # choice = ['test', 'predict', 'video']
fixPredict = False
inputImg = Path.cwd() / 'datasets/mtsd_voc/JPEGImages/bagui.jpg'      # This takes effect when the selected mode is 'predict'
inputDem = Path.cwd() / 'datasets/mtsd_voc/DEMImages/bagui.jpg'      # This takes effect when the selected mode is 'predict'

# This takes effect when the selected mode is 'test'
val_file = Path.cwd() / 'datasets/mtsd_voc/ImageSets/Segmentation/val.txt'       # path to a single image or image director
res_path = Path.cwd() / 'datasets/result'     # save segmentation results to the specified dir


def predict():
    #--------------------------------#
    #-----------超参数----------------#
    #--------------------------------#
    # Network Options
    # model_name = 'deeplabv3plus_resnet50'
    model_name = 'mtss_resnet50'      # choices=['mtss_resnet50', 'mtss_resnet101',
                                                # 'mtss_mobilenet', 'deeplabv3plus_resnet50']
                                                # 'deeplabv3plus_resnet101', 'deeplabv3plus_mobilenet']
    separable_conv = False      # apply separable conv to decoder and aspp
    resize = -1
    output_stride = 16      # choices=[8, 16]
    num_classes = 7

    # predict
    decode_fn = villageFactorsSegm.decode_target
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Setup Dataloader
    if mode == 'test':
        with open(val_file, "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        image_files = [Path.cwd() / 'datasets/mtsd_voc/JPEGImages' / (x + '.jpg') for x in file_names]
        dem_files = [Path.cwd() / 'datasets/mtsd_voc/DEMImages' / (x + '.jpg') for x in file_names]
    else:
        image_files = [inputImg]
        dem_files = [inputDem]

    # Setup model
    model_map = {
        'mtss_resnet50': network.mtss_resnet50,
        'mtss_resnet101': network.mtss_resnet101,
        'mtss_mobilenet': network.mtss_mobilenet,
        'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
        'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    model = model_map[model_name](num_classes=num_classes, output_stride=output_stride)
    if separable_conv:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if ckpt is not None and os.path.isfile(ckpt):
        checkpoint = torch.load(ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"], False)
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    transform_remote = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.2737, 0.3910, 0.3276],
                            std=[0.1801, 0.1560, 0.1301]),
            ])
    transform_dem = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.4153],
                            std=[0.2405]),
            ])
    if res_path is not None:
        os.makedirs(res_path, exist_ok=True)

    def single_predict(image, dem):
        img = transform_remote(image.resize((512, 512))).unsqueeze(0) # To tensor of NCHW
        img = img.to(device)
        dem_ori = Image.open(dem_path).convert('L').resize((512, 512))
        dem = transform_dem(dem_ori).unsqueeze(0) # To tensor of NCHW
        if model_name[:4] == 'mtss':
            # 处理高程数据
            dem = dem.to(device)
            if model_name[:4] == 'mtss':
                pred = model(img, dem).max(1)[1].cpu().numpy()[0] # HW
            else:
                img = torch.cat((img, dem), 1)
                pred = model(img).max(1)[1].cpu().numpy()[0] # HW
        else:
            pred = model(img).max(1)[1].cpu().numpy()[0]
        # 解码
        colorized_preds = decode_fn(pred).astype('uint8')
        result = segm_adjust(colorized_preds, dem_ori)
        result = Image.fromarray(result)
        return result

    def fix_predict(image, dem, multiple):
        single_width = image.width // multiple
        single_height = image.height // multiple
        boxes = [[(i*single_width, j*single_height, (i+1)*single_width, (j+1)*single_height) for j in range(multiple)] for i in range(multiple)]
        images = [image.crop(box) for box in boxes]
        dems = [dem.crop(box) for box in boxes]
        results = [single_predict(image, dem) for (image, dem) in zip(images, dems)]
        result = concat_images(results, (single_width, single_height), multiple)
        return result

    with torch.no_grad():
        model = model.eval()
        for img_path, dem_path in tqdm(zip(image_files, dem_files)):
            if not img_path.stem == dem_path.stem:
                raise ValueError("遥感数据与dem数据不对应，请检查后重试！")
            # 处理遥感数据
            img = Image.open(img_path).convert('RGB')
            dem = Image.open(dem_path).convert('L')
            if fixPredict:
                result = fix_predict(img, dem, 5)
            else:
                result = single_predict(img, dem)
            result.show()
            if res_path:
                result.save(res_path / (img_path.stem+'.png'))


def concat_images(images, size, multiple):
    target = Image.new('RGB', (size[0] * multiple, size[1] * multiple))
    for row in range(multiple):
        for col in range(multiple):
            #对图片进行逐行拼接
            #paste方法第一个参数指定需要拼接的图片，第二个参数为二元元组（指定复制位置的左上角坐标）
            #或四元元组（指定复制位置的左上角和右下角坐标）
            target.paste(images[multiple*row+col], (0 + size[0]*col, 0 + size[1]*row))
    return target


def segm_adjust(image, dem):
    """
    根据dem数据对分割结果进行调整
    """
    return image


if __name__ == "__main__":
    predict()
