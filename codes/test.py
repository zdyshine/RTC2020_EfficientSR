import argparse
import torch
import os
import numpy as np
from utils import utils
import skimage.color as sc
import skimage.io as sio
from model import model
import cv2
# Testing settings
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
parser = argparse.ArgumentParser(description='agora_sr_2020')
parser.add_argument("--test_hr_folder", type=os.path.abspath, default='../datasets/test_data/hr/',
                    help='the folder of the target images')
parser.add_argument("--test_lr_folder", type=os.path.abspath, default='../datasets/test_data/lr/x2/',
                    help='the folder of the input images')
parser.add_argument("--output_folder", type=os.path.abspath, default='../results/test_data/x2')
parser.add_argument("--checkpoint", type=os.path.abspath, default='../training/epoch_91.pth',
                    help='checkpoint folder to use')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use cuda')
parser.add_argument("--upscale_factor", type=int, default=2,
                    help='upscaling factor')
parser.add_argument("--is_y", action='store_true', default=False,
                    help='evaluate on y channel, if False evaluate on RGB channels')
opt = parser.parse_args()

print(opt)

cuda = opt.cuda
device = torch.device('cuda' if cuda else 'cpu')

filepath = opt.test_hr_folder

filelist = utils.get_list(filepath, ext='.png') + utils.get_list(filepath, ext='.JPG')
psnr_list = np.zeros(len(filelist))
ssim_list = np.zeros(len(filelist))
time_list = np.zeros(len(filelist))

model = model.model_rtc(upscale=opt.upscale_factor)
model_dict = utils.load_state_dict(opt.checkpoint)
model.load_state_dict(model_dict, strict=True)

i = 0
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

for imname in filelist:
    # im_gt = cv2.imread(imname, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_gt = sio.imread(imname) # RGB
    im_gt = utils.modcrop(im_gt, opt.upscale_factor)
    # im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    # im_l = cv2.imread(opt.test_lr_folder + imname.split('/')[-1].split('.')[0] + ext, cv2.IMREAD_COLOR)[:, :, [2, 1, 0]]  # BGR to RGB
    im_l = sio.imread(opt.test_lr_folder + '/' + imname.split('/')[-1])  # RGB
    if len(im_gt.shape) < 3:
        im_gt = im_gt[..., np.newaxis]
        im_gt = np.concatenate([im_gt] * 3, 2)
        im_l = im_l[..., np.newaxis]
        im_l = np.concatenate([im_l] * 3, 2)
    im_input = im_l / 255.0
    im_input = np.transpose(im_input, (2, 0, 1))
    im_input = im_input[np.newaxis, ...]
    im_input = torch.from_numpy(im_input).float()

    if cuda:
        model = model.to(device)
        im_input = im_input.to(device)

    with torch.no_grad():
        start.record()
        out = model(im_input)
        end.record()
        torch.cuda.synchronize()
        time_list[i] = start.elapsed_time(end)  # milliseconds

    out_img = utils.tensor2np(out.detach()[0])
    crop_size = opt.upscale_factor
    cropped_sr_img = utils.shave(out_img, crop_size)
    cropped_gt_img = utils.shave(im_gt, crop_size)
    if opt.is_y is True:
        im_label = utils.quantize(sc.rgb2ycbcr(cropped_gt_img)[:, :, 0])
        im_pre = utils.quantize(sc.rgb2ycbcr(cropped_sr_img)[:, :, 0])
    else:
        im_label = cropped_gt_img
        im_pre = cropped_sr_img
    psnr_list[i] = utils.compute_psnr(im_pre, im_label)
    ssim_list[i] = utils.compute_ssim(im_pre, im_label)


    output_folder = os.path.join(opt.output_folder,
                                 imname.split('/')[-1].split('.')[0] + 'x' + str(opt.upscale_factor) + '.png')

    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)

    cv2.imwrite(output_folder, out_img[:, :, [2, 1, 0]])
    i += 1


print("Mean PSNR: {}, SSIM: {}, TIME: {} ms".format(np.mean(psnr_list), np.mean(ssim_list), np.mean(time_list)))
