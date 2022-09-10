import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from PIL import Image

def imageSets_mtvcd():
    total_path = r'F:\Dataset\traditional villages\villageSP_QDN\ImageSets\spatial_patterns.csv'
    saveBasePath = Path(r'F:\Dataset\traditional villages\villageSP_QDN\ImageSets')

    trainval_percent=1.0      # 训练验证集/测试集比例
    train_percent=0.8      # 训练集/验证集比例

    total = np.loadtxt(total_path, dtype=str, delimiter=',')
    if trainval_percent == 1.0:
        trainval = total
        test = []
    else:
        trainval, test = train_test_split(total, train_size=trainval_percent)
    train, valid = train_test_split(trainval, train_size=train_percent)
    np.savetxt(saveBasePath / 'trainval.txt', trainval, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'train.txt', train, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'test.txt', test, fmt='%s', delimiter=',')
    np.savetxt(saveBasePath / 'val.txt', valid, fmt='%s', delimiter=',')
    print("train and val size:", len(trainval))
    print("train size:", len(train))


if __name__ == "__main__":
    # imageSets_mtvcd()
    def f(x):
        return x**2
    ss = [y for x in range(10) if (y := f(x)) < 10]
    print(ss)
