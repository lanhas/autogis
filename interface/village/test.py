from PIL import Image
import numpy as np
import cv2


colorMask = {'urban_land': (0, 255, 255),
             'agriculture_land': (255, 255, 0),
             'rangeland': (255, 0, 255),
             'forest_land': (0, 255, 0),
             'water': (0, 0, 255),
             'barren_land': (255, 255, 255),
             'unkonwn': (0, 0, 0),
}

def image_blend(img: Image, areaMask: np.array, alpha, beta, gamma) -> Image:
    foreground = np.array(img)
    background = np.array(img)
    for i in range(3):
        foreground[:, :, i][areaMask == 0] = 0
        background[:, :, i][areaMask > 0] = 0
    result = cv2.addWeighted(foreground, alpha, background, beta, gamma)
    result = Image.fromarray(result)
    return result

def iterboolAnd(boolList):
    result = boolList[0]
    for i in range(len(boolList)-1):
        result = np.logical_and(result, boolList[i+1])
    return result

def color2annotation(im_array: np.array) -> np.array:
    annotation = np.zeros_like(im_array[:, :, 0], dtype=np.uint8)
    annotation[cv2.inRange(im_array, colorMask['urban_land'], colorMask['urban_land'])==255 ] = 1
    annotation[cv2.inRange(im_array, colorMask['agriculture_land'], colorMask['agriculture_land'])==255] = 2
    annotation[cv2.inRange(im_array, colorMask['rangeland'], colorMask['rangeland'])==255] = 3
    annotation[cv2.inRange(im_array, colorMask['forest_land'], colorMask['forest_land'])==255] = 4
    annotation[cv2.inRange(im_array, colorMask['water'], colorMask['water'])==255] = 5
    annotation[cv2.inRange(im_array, colorMask['barren_land'], colorMask['barren_land'])==255] = 6
    annotation[cv2.inRange(im_array, colorMask['unkonwn'], colorMask['unkonwn'])==255] = 7
    return annotation


def getAreaMask(colorImg: Image, areaIndex: int) -> np.array:
    annotation = color2annotation(np.array(colorImg))
    result = np.zeros_like(annotation, dtype=np.uint8)
    if areaIndex == 0:
        result = np.array(colorImg)
    elif areaIndex == 1:
        result[annotation == 0] = 1
    elif areaIndex == 2:
        result[annotation == 1] = 1
    elif areaIndex == 3:
        result[annotation == 2] = 1
    elif areaIndex == 4:
        result[annotation == 3] = 1
    elif areaIndex == 5:
        result[annotation == 4] = 1
    elif areaIndex == 6:
        result[annotation == 5] = 1
    elif areaIndex == 7:
        result[annotation == 6] = 1
    return result

oriImg = Image.open(r'land_classification\data\14397_sat.jpg')
maskColor = Image.open(r'land_classification\result\testimg.png')
# mask_array = np.array(maskColor)
areaMask = getAreaMask(maskColor, 1)
# result = Image.fromarray(areaMask)
result = image_blend(oriImg, areaMask, 1, 0.5, 0)
result.show()