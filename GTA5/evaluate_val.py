import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch.utils.data as torch_data

import os

from PIL import Image
from torch.autograd import Variable

from util.metrics import runningScore
from model.model_noaux import SegModel
from util.utils import load_models
from util.loader.CityLoader import CityLoader

num_classes = 4
CITY_DATA_PATH = './data/Infrared'
DATA_LIST_PATH_VAL_IMG = './util/loader/cityscapes_list/val.txt'
DATA_LIST_PATH_VAL_LBL  = './util/loader/cityscapes_list/val_label.txt'
WEIGHT_DIR = './work_dir/weights'
CUDA_DIVICE_ID = '0'

IMG_MEAN = np.array((94.095, 102.306, 108.12), dtype=np.float32)

parser = argparse.ArgumentParser(description='DiGA \
	for unsupervised domain adaptation for semantic segmentation')
parser.add_argument('--weight_dir', type=str, default=WEIGHT_DIR)
parser.add_argument('--city_data_path', type=str, default=CITY_DATA_PATH, help='the path to cityscapes.')
parser.add_argument('--data_list_path_val_img', type=str, default=DATA_LIST_PATH_VAL_IMG)
parser.add_argument('--cuda_device_id', nargs='+', type=str, default=CUDA_DIVICE_ID)
parser.add_argument('--data_list_path_val_lbl', type=str, default=DATA_LIST_PATH_VAL_LBL)

args = parser.parse_args()

print ('cuda_device_id:', ','.join(args.cuda_device_id))
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(args.cuda_device_id)

palette = [0,0,0,255,0,0,0,255,0,255,255,0]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask


args = parser.parse_args()

val_set   = CityLoader(args.city_data_path, args.data_list_path_val_img, args.data_list_path_val_lbl, max_iters=None, crop_size=[256, 256], mean=IMG_MEAN, set='val')
val_loader= torch_data.DataLoader(val_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

upsample_1024 = nn.Upsample(size=[256, 256], mode='bilinear', align_corners=True)

model_dict = {}

student = SegModel().cuda()
model_dict['student'] = student

load_models(model_dict, args.weight_dir)

student.eval()

cty_running_metrics = runningScore(num_classes)
print('evaluating models ...')
for i_val, (images_val, labels_val) in enumerate(val_loader):
    print(i_val)
    #multi-scale testing
    images_val = Variable(images_val.cuda(), requires_grad=False)
    labels_val = Variable(labels_val, requires_grad=False)

    images_ds_val = nn.functional.interpolate(images_val, (256, 256), mode='bilinear', align_corners=True)
    with torch.no_grad():
        _, _, pred, _ = student(images_val)
        _, _, pred_ds, _= student(images_ds_val)
    pred = upsample_1024(pred)
    pred_ds = upsample_1024(pred_ds)
    pred = torch.max(pred, pred_ds)
    #pred = torch.add(pred, pred0)/2
    pred = pred.data.max(1)[1].cpu().numpy()
    gt = labels_val.data.cpu().numpy()
    cty_running_metrics.update(gt, pred)
cty_score, cty_class_iou = cty_running_metrics.get_scores()

for k, v in cty_score.items():
    print(k, v)

