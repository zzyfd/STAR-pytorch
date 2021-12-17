import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import time
from dataloaders.dataloader_hdrplus_enhance_portrait_corrected import EnhanceDataset
from dataloaders.dataloader_FiveK import EnhanceDataset_FiveK
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
from util import calculate_ssim, calculate_psnr, tensor2img
import models


def weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d):
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def soft_dice_loss(y_true, y_pred, epsilon=1e-6):
    axes = tuple(range(1, len(y_pred.shape) - 1))
    numerator = 2. * torch.sum(y_pred * y_true, axes)
    denominator = torch.sum((y_pred) ** 2 + (y_true) ** 2, axes)

    return 1 - torch.mean(numerator / (denominator + epsilon))


def eval(config):
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)

    DCE_net = models.build_model(config)

    print(DCE_net)

    is_yuv = False

    DCE_net.apply(weights_init)

    timestr = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(config.snapshots_folder + '/runs/' + timestr)  # default

    if config.parallel:
        print('Using DataParallel')
        DCE_net = nn.DataParallel(DCE_net)
    if config.pretrain_dir is not None:
        print('Loading {}'.format(config.pretrain_dir))
        DCE_net.load_state_dict(torch.load(config.pretrain_dir), strict=True)
    image_size = config.image_h
    image_size_w = config.image_w
    eval_dataset = EnhanceDataset(
        config.lowlight_images_path, image_size, image_size_w,
        is_yuv=is_yuv, resize=True) if config.dataset == 'portrait' else EnhanceDataset_FiveK(
        config.lowlight_images_path,
        image_size, image_size_w, is_yuv=is_yuv, resize=True)
    eval_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=config.val_batch_size, shuffle=False, num_workers=1,
        pin_memory=False, drop_last=True)

    L_L1 = nn.L1Loss()

    DCE_net.eval()
    psnr_sum = 0
    ssim_sum = 0
    count = 0
    img_idx = 0
    if config.save_img:
        try:
            os.makedirs(config.snapshots_folder + '/results')
        except:
            pass
    for epoch in range(1):
        batch_time_sum = 0
        for iteration, img_lowlight in enumerate(eval_loader):

            img_input, img_ref = img_lowlight
            img_input = img_input.cuda()
            img_ref = img_ref.cuda()
            # matte = matte.cuda()
            img_resize = F.interpolate(img_input, (config.image_ds, config.image_ds), mode='area')
            torch.cuda.synchronize()
            end = time.time()
            enhanced_image, x_r = DCE_net(img_resize, img_in=img_input)


            torch.cuda.synchronize()
            batch_time = time.time() - end
            batch_time_sum += batch_time

            loss = L_L1(enhanced_image, img_ref)

            for i in range(img_input.shape[0]):
                img_output = tensor2img(enhanced_image[i], bit=8)
                img_gt = tensor2img(img_ref[i], bit=8)
                psnr = calculate_psnr(img_output, img_gt)
                ssim = calculate_ssim(img_output, img_gt)
                count += 1
                psnr_sum += psnr
                ssim_sum += ssim
                if config.save_img:
                    torchvision.utils.save_image(enhanced_image[i],
                                                 '{}/results/{}.jpg'.format(config.snapshots_folder, img_idx))
                    img_idx += 1

            if iteration == 0:
                A = x_r
                n, _, h, w = A.shape
                A = A.sub(A.view(n, _, -1).min(dim=-1)[0].view(n, _, 1, 1)).div(
                    A.view(n, _, -1).max(dim=-1)[0].view(n, _, 1, 1) - A.view(n, _, -1).min(dim=-1)[0].view(n,
                                                                                                            _,
                                                                                                            1,
                                                                                                            1))
                writer.add_image('input_enhanced_ref_residual',
                                 torch.cat([img_input[0], enhanced_image[0], img_ref[0],
                                            torch.abs(
                                                enhanced_image[0] - img_ref[0])] + [torch.stack(
                                     (A[0, i], A[0, i], A[0, i]), 0) for i in range(A.shape[1])],
                                           2
                                           ), epoch)

                print("------------------Epoch %d--------------------" % epoch)
                # import pdb; pdb.set_trace()epoch)

            if (iteration % config.display_iter) == 0:
                istep = epoch * len(eval_loader) + iteration
                print("Loss at iteration", iteration, ":", loss.item(),
                      "Batch time: ", batch_time, "PSNR: ", psnr_sum / count, " SSIM: ", ssim_sum / count,
                      "Batch Time AVG: ", batch_time_sum / (iteration + 1))
                writer.add_scalar('loss', loss, istep)

                # import pdb; pdb.set_trace()

            if ((iteration) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder +
                           "/Epoch_latest_blendparam_mean_corrected461_test.pth")
        print("Final Loss:", loss.item(),
              "Batch time: ", batch_time, "PSNR: ", psnr_sum / count, " SSIM: ", ssim_sum / count, "Batch Time AVG: ",
              batch_time_sum / (iteration + 1))
        writer.add_scalar('loss', loss, istep)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()


    parser.add_argument('--lowlight_images_path', type=str)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_type', type=str, default='fix')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--zerodce', type=bool, default=False)
    parser.add_argument('--dataset', type=str, default='fivek')
    parser.add_argument('--pretrain_dir', type=str,
                        default=None)
    parser.add_argument('--image_h', type=int, default=1200)
    parser.add_argument('--image_w', type=int, default=900)
    parser.add_argument('--image_ds', type=int, default=256)
    parser.add_argument('--save_img', action='store_true')
    parser.add_argument('--model', type=str, default='STAR-DCE-Base')

    config = parser.parse_args()
    models.MODEL_REGISTRY[config.model].add_args(parser)
    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)
    with torch.no_grad():
        eval(config)
