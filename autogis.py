# from PIL import Image
#
# aa = Image.open(r'F:\Dataset\tradition_villages\remote\AH1_001_CF.tif')
# print(aa.size)
import numpy as np

a = np.arange(12).reshape((1, 3, 4))
b = np.arange(6).reshape((1, 3, 2))
print(np.concatenate((a, b), axis=2))