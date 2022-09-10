#Autogis

This is a pytorch deep learning program for tradition Chinese settlements (TCSs) study.


### Installation

---
Please refer to [requirement.txt](./requirements.txt) for all required packages. Assuming [Anaconda](https://www.anaconda.com/products/distribution) with python 3.8, a step-by-step example for installing this project is as follows

```commandline
conda install pytorch torchvision cudatoolkit=11.1 -c pytorch
conda install -c tensorboard pillow opencv-python pyqt
conda install -c anaconda seaborn
```

Then, clone this repo. Make sure git is installed and run:

```commandline
git clone https://github.com/lanhas/autogis
cd autogis
```

##Data

---

Prepare datasets of interest as described in [dataset.md](./dataset.md).

##Training

---
Read the [training tutorial](./train.md) for details.

##Usage

---
Read the [usage tutorial](./usage.md) for details.

Acknowledgement for reference repos

---

+ [Deepglobal]()
+ [DeepLab V3+]()
+ [Meta-Baseline]()

##Citation

---
```commandline
@misc{xue2022antra,
      title{An Environmental Patterns Recognition Method of Traditional Chinese Settlements via Meta-Learning},
      author={Peng Xue}

}
```

