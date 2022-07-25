# from PIL import Image
#
# aa = Image.open(r'F:\Dataset\traditional villages_QDN\contourLine_g\GZ1_024_ZC_demi.png')
# aa = aa.convert('L')
# aa.save('test.png')
# aa.show()
import models.village_clss as models

model = models.resneta.resnet50()
print(model)