from easydict import EasyDict


cfg = EasyDict()

cfg.InputSize_w = 112
cfg.InputSize_h = 112
# cfg.InputSize_w = 64
# cfg.InputSize_h = 64

cfg.Imgdir = '/Users/yanyan/data/FireData'
# cfg.Imgdir = '/wdc/changshuang/data/FireData'
# cfg.Imgdir = '/wdc/changshuang/data/CropFireData'
# training set
cfg.EPOCHS = 60
