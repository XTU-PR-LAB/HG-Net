import time
import argparse
import datetime
import sys
import os

import torch
import torch.nn as nn
import torch.nn.utils as utils

import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter

from c2fLearner import *
from data_loader import *
from opt import Options
from path import Path

device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")

inv_normalize = transforms.Normalize(
    mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
    std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
)


def normalize_result(value, vmin=None, vmax=None):
    value = value.cpu().numpy()[0, :, :]

    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax

    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)
    else:
        value = value * 0.

    return np.expand_dims(value, 0)


def cal_eval_metrics(opt, depth_grd, depth_pred, depth_raw):
    gt_height, gt_width = opt.H, opt.W

    crop = np.array([0.40810811 * gt_height, 0.99189189 * gt_height,
                     0.03594771 * gt_width, 0.96405229 * gt_width]).astype(np.int32)
    crop_mask = np.zeros((gt_height, gt_width), dtype=np.float32)
    crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = 1
    crop_mask = crop_mask.reshape(-1)

    grd = depth_grd.cpu().numpy().reshape(-1)
    pred = depth_pred.cpu().numpy().reshape(-1)
    raw_info = depth_raw.cpu().numpy().reshape(-1)
    mask = np.logical_and(raw_info >= opt.min_depth, raw_info <= opt.max_depth)
    mask = np.logical_and(crop_mask, mask)
    grd = grd[mask]
    pred = pred[mask]

    pred[pred > opt.max_depth] = opt.max_depth
    pred[pred < opt.min_depth] = opt.min_depth

    return compute_errors(grd, pred)


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    d1 = (thresh < 1.25).mean()
    d2 = (thresh < 1.25 ** 2).mean()
    d3 = (thresh < 1.25 ** 3).mean()

    rms = (gt - pred) ** 2
    rms = np.sqrt(rms.mean())

    log_rms = (np.log(gt) - np.log(pred)) ** 2
    log_rms = np.sqrt(log_rms.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    err = np.abs(np.log10(pred) - np.log10(gt))
    log10 = np.mean(err)
    #print(d1, d2, d3, rms, log_rms, abs_rel, sq_rel, log10)
    return d1, d2, d3, rms, log_rms, abs_rel, sq_rel, log10


def eval(args):
    metrics = np.zeros([9], dtype=np.float64)

    coarse_model = disp_coarse()
    refine_model = disp_refine()

    num_params = sum([np.prod(p.size()) for p in coarse_model.parameters()])
    print("Total number of coarse parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in coarse_model.parameters() if p.requires_grad])
    print("Total number of coarse learning parameters: {}".format(num_params_update))

    num_params = sum([np.prod(p.size()) for p in refine_model.parameters()])
    print("Total number of refine parameters: {}".format(num_params))

    num_params_update = sum([np.prod(p.shape) for p in refine_model.parameters() if p.requires_grad])
    print("Total number of refine learning parameters: {}".format(num_params_update))

    if args.checkpoint_path != '':
        if os.path.isfile(args.checkpoint_path):
            print("Loading checkpoint '{}'".format(args.checkpoint_path))
            if args.gpu is None:
                checkpoint = torch.load(args.checkpoint_path)
            else:
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.checkpoint_path, map_location=loc)
            global_step_coarse = checkpoint['global_step_coarse']
            global_step_refine = checkpoint['global_step_refine']
            coarse_model.load_state_dict(checkpoint['coarse_disp'])
            refine_model.load_state_dict(checkpoint['refine_disp'])
            print("Model Initialized")
        else:
            print("No checkpoint found at '{}'".format(args.checkpoint_path))
            return

    # coarse_model = torch.nn.DataParallel(coarse_model)
    coarse_model.to(device)

    # refine_model = torch.nn.DataParallel(refine_model)
    refine_model.to(device)

    coarse_model.eval()
    refine_model.eval()
    cudnn.benchmark = True

    dataloader = c2fDataLoader(args)

    # writer = SummaryWriter(args.eval_summary_dir, flush_secs=30)
    l1_criterion = silog_loss(args.variance_focus).to(device)

    print('data_loader length: %d' % (len(dataloader.data)))
    print('evaluation access')
    time_acc = 0
    loss_acc = 0
    for step, sample_batched in enumerate(dataloader.data):
        before_op_time = time.time()
        with torch.no_grad():
            image = sample_batched['image'].to(device)
            depth_raw = sample_batched['depth_raw'].to(device)
            depth_blur = sample_batched['depth_blur'].to(device)
            sup_signal, _ = coarse_model(image)

            refine_disp, _ = refine_model(image, sup_signal)
            mask_raw = depth_raw > 1e-3
            refine_depth = refine_disp * args.max_depth
            loss = l1_criterion(refine_depth, depth_blur, mask_raw.to(torch.bool))
        after_op_time = time.time()
        loss_acc += loss.cpu().item()
        time_acc += (after_op_time - before_op_time)
        metrics[:-1] = metrics[:-1] + cal_eval_metrics(args, depth_blur.detach(), refine_depth.detach(),
                                                       depth_raw.detach())
        metrics[-1] = metrics[-1] + 1
        if step % args.log_freq == 0:
            print('step %d loss %.4f  %.4f sec/batch' % (step, loss_acc / args.log_freq, time_acc / args.log_freq))

            # depth_raw = torch.where(depth_raw < 1, depth_raw * 0 + 1, depth_raw)
            # for i in range(args.batch_size):
            #     writer.add_scalar('loss', loss, step)
            #     writer.add_image('depth_grd'.format(i),
            #                      normalize_result(1.0 / depth_raw[i, :, :, :].detach()),
            #                      step)
            #     writer.add_image('depth_pred'.format(i),
            #                      normalize_result(1.0 / refine_depth[i, :, :, :].detach()),
            #                      step)
            # writer.flush()
            time_acc = 0
            loss_acc = 0
    print('evaluation done!')
    metrics[:-1] = metrics[:-1] / metrics[-1]
    return metrics[:-1]


def main():
    args = Options().parse()
    a1, a2, a3, rmse, rmse_log, abs_rel, sq_rel, log10 = eval(args)
    print('a1:%.4f a2:%.4f a3:%.4f rmse:%.4f rmse_log:%.4f abs_rel:%.4f sq_rel:%.4f log10:%.4f\n' % (
        a1, a2, a3, rmse, rmse_log, abs_rel, sq_rel, log10))

if __name__ == '__main__':
    main()