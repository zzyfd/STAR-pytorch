fb_pathmgr_registerd = False

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import argparse
import time
import torch.nn.functional as F
from dataloaders.dataloader_FiveK import EnhanceDataset_FiveK

from models import model
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import random
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


def train(config):
    torch.backends.cudnn.deterministic = True
    random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    np.random.seed(0)
    best_loss = 100

    DCE_net = models.build_model(config).cuda()
    print(DCE_net)

    is_yuv = False
    DCE_net.apply(weights_init)
    timestr = datetime.now().strftime('%Y%m%d-%H%M%S')
    writer = SummaryWriter(config.snapshots_folder + '/runs/' + timestr)  # default
    if config.load_pretrain == True:
        DCE_net.load_state_dict(torch.load(config.pretrain_dir), strict=True)
    image_size = config.image_h
    image_size_w = config.image_w
    train_dataset = EnhanceDataset_FiveK(config.lowlight_images_path,
                                         image_size, is_yuv=is_yuv, image_size_w=image_size_w)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.train_batch_size, shuffle=False, num_workers=config.num_workers,
        pin_memory=True, drop_last=True)

    L_L1 = nn.L1Loss() if not config.l2_loss else nn.MSELoss()

    optimizer = torch.optim.Adam(DCE_net.parameters(
    ), lr=config.lr, weight_decay=config.weight_decay)

    if config.lr_type == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                  T_max=config.num_epochs * len(train_loader))
    elif config.lr_type == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1,
                                                       step_size=config.num_epochs // 3 * len(train_loader))
    else:
        lr_scheduler = None

    DCE_net.train()
    if config.parallel:
        print('Using DataParallel')
        DCE_net = nn.DataParallel(DCE_net)

    for epoch in range(config.num_epochs):
        batch_time_sum = 0
        for iteration, img_lowlight in enumerate(train_loader):

            img_input, img_ref = img_lowlight
            img_input = img_input.cuda()
            img_input_ds = F.interpolate(img_input, (config.image_ds, config.image_ds), mode='area')
            img_ref = img_ref.cuda()
            torch.cuda.synchronize()
            end = time.time()
            enhanced_image, x_r = DCE_net(img_input_ds, img_in=img_input)
            torch.cuda.synchronize()
            batch_time = time.time() - end
            batch_time_sum += batch_time

            import losses
            L_color = losses.L_color()
            loss_color = torch.mean(L_color(enhanced_image)) if config.color_loss else torch.zeros([]).cuda()
            loss_l1 = L_L1(enhanced_image, img_ref) if not config.no_l1 else torch.zeros([]).cuda()
            loss_cos = 1 - nn.functional.cosine_similarity(enhanced_image, img_ref,
                                                           dim=1).mean() if config.cos_loss else torch.zeros([]).cuda()

            if config.mul_loss:
                loss = loss_l1 * loss_cos
            else:
                loss = loss_l1 + loss_color + 100 * loss_cos
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                DCE_net.parameters(), config.grad_clip_norm)
            optimizer.step()
            if config.lr_type != 'fix':
                lr_scheduler.step()

            if iteration == 0:
                A = x_r
                n, _, h, w = A.shape
                A = A.sub(A.view(n, _, -1).min(dim=-1)[0].view(n, _, 1, 1)).div(
                    A.view(n, _, -1).max(dim=-1)[0].view(n, _, 1, 1) - A.view(n, _, -1).min(dim=-1)[0].view(n,
                                                                                                            _,
                                                                                                            1,
                                                                                                            1)).squeeze()
                writer.add_image('input_enhanced_ref_residual',
                                 torch.cat([img_input[0], enhanced_image[0], img_ref[0],
                                            torch.abs(
                                                enhanced_image[0] - img_ref[0])] + [torch.stack(
                                     (A[0, i], A[0, i], A[0, i]), 0) for i in range(A.shape[1])],
                                           2
                                           ), epoch)
                print("------------------Epoch %d--------------------" % epoch)

            if (iteration % config.display_iter) == 0:
                istep = epoch * len(train_loader) + iteration
                print("Loss at iteration", iteration, ":", loss_l1.item(), " | LR : ", optimizer.param_groups[0]['lr'],
                      "Batch time: ", batch_time, "Batch Time AVG: ", batch_time_sum / (iteration + 1), 'Cos loss: ',
                      loss_cos.item())
                writer.add_scalar('loss', loss_l1, istep)
                writer.add_scalar('Color_loss', loss_color, istep)
                writer.add_scalar('Cos_loss', loss_cos, istep)

                writer.add_scalar('lr', optimizer.param_groups[0]['lr'], istep)

            if ((iteration) % config.snapshot_iter) == 0:
                torch.save(DCE_net.state_dict(), config.snapshots_folder +
                           "/Epoch_latest.pth")

            if loss < best_loss:
                best_loss = loss
                print('Save best loss {}'.format(loss))
                torch.save(DCE_net.state_dict(), config.snapshots_folder +
                           "/Epoch_best.pth")


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--lowlight_images_path', type=str, default=path)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--lr_type', type=str, default='fix')
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=1)
    parser.add_argument('--num_epochs', type=int, default=300)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--display_iter', type=int, default=10)
    parser.add_argument('--snapshot_iter', type=int, default=10)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/relpos")
    parser.add_argument('--load_pretrain', type=bool, default=False)
    parser.add_argument('--parallel', type=bool, default=False)
    parser.add_argument('--zerodce', type=bool, default=False)
    parser.add_argument('--tv_loss', type=bool, default=False)
    parser.add_argument('--color_loss', type=bool, default=False)
    parser.add_argument('--l2_loss', type=bool, default=False)
    parser.add_argument('--cos_loss', type=bool, default=False)
    parser.add_argument('--mul_loss', type=bool, default=False)
    parser.add_argument('--no_l1', type=bool, default=False)

    parser.add_argument('--dataset', type=str, default='fivek')
    parser.add_argument('--pretrain_dir', type=str,
                        default="snapshots/pretrained/Epoch99.pth")
    parser.add_argument('--image_h', type=int, default=256)
    parser.add_argument('--image_w', type=int, default=256)
    parser.add_argument('--image_ds', type=int, default=256)

    parser.add_argument('--model', type=str, default='STAR-DCE-Base')

    args, _ = parser.parse_known_args()
    models.MODEL_REGISTRY[args.model].add_args(parser)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    config = get_args()
    if not os.path.exists(config.snapshots_folder):
        os.makedirs(config.snapshots_folder)
    print(config)
    train(config)
