import argparse

import time

from copy import deepcopy

from PIL import Image
import numpy as np
from collections import deque

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
import torchvision.models as models
# import matplotlib.pyplot as plt

from clip.custom_clip_device import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask
import matplotlib.pyplot as plt

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))


def plot_metrics(stu_ent_list, tea_ent_list, acc_list, args):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    x = list(range(len(acc_list)))

    # 第一个y轴：绘制loss
    ax1.set_xlabel("Iteration (Batch Index)")
    ax1.set_ylabel("Avg Entropy Loss", color='tab:blue')
    ax1.plot(x, stu_ent_list, label="Stu Avg Entropy", linestyle='--', color='tab:blue')
    ax1.plot(x, tea_ent_list, label="Tea Avg Entropy", linestyle='--', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylim(0, 5)  # <<< 调整loss曲线的Y轴范围

    # 添加第二个y轴：绘制accuracy
    ax2 = ax1.twinx()
    ax2.set_ylabel("Top-1 Accuracy (%)", color='tab:green')
    ax2.plot(x, acc_list, label="Top-1 Accuracy", linestyle='-.', color='tab:green')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.set_ylim(-100, 100)  # <<< 调整accuracy曲线的Y轴范围

    # 图例处理（来自两个轴，需要合并）
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    plt.title("TTA Loss and Accuracy per Iteration")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"logs/update_teacher_lr_{args.lr}_reset<={args.reset}.png")


def select_confident_samples(logits, top):
    batch_entropy = (-logits.softmax(1) * logits.log_softmax(1)).sum(1)
    idx = torch.argsort(batch_entropy, descending=False)[:int(batch_entropy.size()[0] * top)]
    return logits[idx], idx


def avg_entropy(outputs):
    logits = outputs - outputs.logsumexp(dim=-1, keepdim=True)  # logits = outputs.log_softmax(dim=1) [N, 1000]
    avg_logits = logits.logsumexp(dim=0) - np.log(logits.shape[0])  # avg_logits = logits.mean(0) [1, 1000]
    min_real = torch.finfo(avg_logits.dtype).min
    avg_logits = torch.clamp(avg_logits, min=min_real)
    return -(avg_logits * torch.exp(avg_logits)).sum(dim=-1)


# add
def get_param_by_name(model, param_name):
    for name, param in model.named_parameters():
        if param_name in name:
            return param
    raise ValueError(f"Parameter {param_name} not found in the model.")


def test_time_tuning(cloud_model, edge_model, rec, optimizer_cloud, optimizer_edge, cloud_scaler, edge_scaler, cloud_optim_state, edge_optim_state, args):
    global num_reset

    selected_idx = None
    # add
    criterion_div = nn.KLDivLoss(reduction='batchmean')
    for j in range(args.tta_steps):

        with torch.cuda.amp.autocast():
            # 1.calculate reset threshold according to student model's entropy
            with torch.no_grad():
                edge_feat = edge_model.get_image_features(rec)
                output_edge = edge_model(None, edge_feat)
                output_edge_select, selected_idx = select_confident_samples(output_edge, args.selection_p)
                loss_ent = avg_entropy(output_edge_select)

            reset = loss_ent.item() <= args.reset
            del edge_feat, output_edge, output_edge_select, loss_ent
            # 2.reset student and teacher model
            if reset:
                num_reset += 1
                # print(f"Avg Entropy: {loss_ent.item()}, Reset model!")
                with torch.no_grad():
                    edge_model.reset()
                    cloud_model.reset()
                optimizer_cloud.load_state_dict(cloud_optim_state)
                optimizer_edge.load_state_dict(edge_optim_state)
            # 3.update teacher model
            cloud_feat = cloud_model.get_image_features(rec)
            output_cloud = cloud_model(None, cloud_feat)
            output_cloud_select, selected_idx = select_confident_samples(output_cloud, args.selection_p)
            loss_cloud = avg_entropy(output_cloud_select)
            tea_ent_list.append(loss_cloud.item())

            optimizer_cloud.zero_grad()
            cloud_scaler.scale(loss_cloud).backward()
            cloud_scaler.step(optimizer_cloud)
            cloud_scaler.update()
            # 4.update student module
            with torch.no_grad():
                output_cloud = cloud_model(None, cloud_feat)
            del cloud_feat, output_cloud_select, loss_cloud

            edge_feat = edge_model.get_image_features(rec)
            output_edge = edge_model(None, edge_feat)
            output_edge_select, selected_idx = select_confident_samples(output_edge, args.selection_p)
            loss_ent = avg_entropy(output_edge_select)
            stu_ent_list.append(loss_ent.item())

            tau = args.tau
            loss_kl = args.beta * criterion_div(F.log_softmax(output_edge / tau, dim=1),
                                                F.softmax(output_cloud / tau, dim=1))

            optimizer_edge.zero_grad()
            edge_scaler.scale(loss_kl + loss_ent).backward()
            edge_scaler.step(optimizer_edge)
            edge_scaler.update()

    return


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    # 重要 一定要先指定GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    set_random_seed(args.seed)
    # print("Use GPU: {} for training".format(args.gpu))

    cloud_model = get_coop(args.cloud_arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
    # add
    edge_model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, type='Device')

    cloud_trainable = []
    edge_trainable = []
    for name, param in cloud_model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad_(True)
            cloud_trainable.append(param)
        else:
            param.requires_grad_(False)

    # add
    for name, param in edge_model.named_parameters():
        if 'prompt_learner' in name:  # prompt_learner
            param.requires_grad_(True)
            edge_trainable.append(param)
        else:
            param.requires_grad_(False)

    # print("=> Device Model created: visual backbone {}".format(args.arch))
    # print("=> Cloud Model created: visual backbone {}".format(args.cloud_arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        cloud_model = cloud_model.cuda(args.gpu)
        # add
        edge_model = edge_model.cuda(args.gpu)

    optimizer_cloud = optim.AdamW(cloud_trainable, args.lr)
    optimizer_edge = optim.AdamW(edge_trainable, args.lr)
    # add
    cloud_optim_state = deepcopy(optimizer_cloud.state_dict())
    edge_optim_state = deepcopy(optimizer_edge.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    # add
    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # print('=> Using native Torch AMP. Training in mixed precision.')

    cudnn.benchmark = True

    # norm stats from clip.load()
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])

    # iterating through eval datasets
    datasets = args.test_sets.split("/")
    results = {}
    for set_id in datasets:
        # if args.tpt:
        base_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution)])
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        data_transform = AugMixAugmenter(base_transform, preprocess, n_views=args.batch_size - 1,
                                         augmix=len(set_id) > 1)
        device_transform = transforms.Compose([
            transforms.Resize(args.resolution, interpolation=BICUBIC),
            transforms.CenterCrop(args.resolution),
            transforms.ToTensor(),
            normalize])

        batchsize = 1

        # print("evaluating: {}".format(set_id))
        # reset the model
        # Reset classnames of custom CLIP model
        if len(set_id) > 1:
            # fine-grained classification datasets
            classnames = eval("{}_classes".format(set_id.lower()))
        else:
            assert set_id in ['A', 'R', 'K', 'V', 'I']
            classnames_all = imagenet_classes
            classnames = []
            if set_id in ['A', 'R', 'V']:
                label_mask = eval("imagenet_{}_mask".format(set_id.lower()))
                if set_id == 'R':
                    for i, m in enumerate(label_mask):
                        if m:
                            classnames.append(classnames_all[i])
                else:
                    classnames = [classnames_all[i] for i in label_mask]
            else:
                classnames = classnames_all

        cloud_model.reset_classnames(classnames, args.cloud_arch)
        # add
        edge_model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, device_transform, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        results[set_id] = test_time_adapt_eval(val_loader, cloud_model, edge_model, optimizer_cloud, optimizer_edge,
                                               cloud_optim_state, edge_optim_state, cloud_scaler, edge_scaler, args)

        del val_dataset, val_loader
    #     try:
    #         print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
    #     except:
    #         print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))
    #
    # print("======== Result Summary ========")
    # print("params: nstep	lr	bs")
    # print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    # print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    # for id in results.keys():
    #     print("{}".format(id), end="	")
    # print("\n")
    # for id in results.keys():
    #     print("{:.2f}".format(results[id][0]), end="	")
    # print("\n")


def test_time_adapt_eval(val_loader, cloud_model, edge_model, optimizer_cloud, optimizer_edge, cloud_optim_state,
                         edge_optim_state, cloud_scaler, edge_scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # results = []
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    # print(f"Test with tau={args.tau}, beta={args.beta}, uplink percentage={args.uplink_percent}%:")
    # reset model and switch to evaluate mode
    cloud_model.eval()
    edge_model.eval()
    with torch.no_grad():
        cloud_model.reset()
        # add
        edge_model.reset()
    end = time.time()
    for i, (oris, recs, target) in enumerate(val_loader):

        assert args.gpu is not None
        # add
        if isinstance(oris, list):  # oris = 64 augmentations
            # for k in range(len(recs)):
            #     oris[k] = oris[k].cuda(args.gpu, non_blocking=True)
            ori = oris[0].cuda(args.gpu, non_blocking=True)
        else:
            if len(oris.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert oris.size()[0] == 1
                oris = oris.squeeze(0)
            oris = oris.cuda(args.gpu, non_blocking=True)
            ori = oris

        if isinstance(recs, list):
            for k in range(len(recs)):
                recs[k] = recs[k].cuda(args.gpu, non_blocking=True)
            # rec = recs[0]
        else:
            if len(recs.size()) > 4:
                # when using ImageNet Sampler as the dataset
                assert recs.size()[0] == 1
                recs = recs.squeeze(0)
            recs = recs.cuda(args.gpu, non_blocking=True)
        target = target.cuda(args.gpu, non_blocking=True)
        recs = torch.cat(recs, dim=0)


        test_time_tuning(cloud_model, edge_model, recs, optimizer_cloud, optimizer_edge, cloud_scaler,
                         edge_scaler, cloud_optim_state, edge_optim_state, args)

        # The actual inference goes here
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # output = model(image)
                ori_edge_feat = edge_model.get_image_features(ori)
                output = edge_model(None, ori_edge_feat)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        top1.update(acc1[0], ori.size(0))
        top5.update(acc5[0], ori.size(0))
        acc_list.append(top1.avg.item())

        if (i + 1) % args.print_freq == 0:
            progress.display(i)
    print(f"Test with thr={args.reset}, lr={args.lr}, total {num_reset} reset:")

    progress.display_summary()

    return [top1.avg, top5.avg]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test-time Prompt Tuning')
    parser.add_argument('data', metavar='DIR', help='path to dataset root')
    parser.add_argument('--test_sets', type=str, default='A/R/V/K/I',
                        help='test dataset (multiple datasets split by slash)')
    parser.add_argument('--dataset_mode', type=str, default='test', help='which split to use: train/val/test')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='RN50')
    parser.add_argument('-c', '--cloud_arch', metavar='CLOUD_ARCH', default='ViT-B/16')
    parser.add_argument('--resolution', default=224, type=int, help='CLIP image resolution')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-b', '--batch-size', default=64, type=int, metavar='N')
    parser.add_argument('--lr', '--learning-rate', default=5e-3, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('-p', '--print-freq', default=500, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    parser.add_argument('--tpt', action='store_true', default=False, help='run test-time prompt tuning')
    parser.add_argument('--selection_p', default=0.1, type=float, help='confidence selection percentile')
    parser.add_argument('--tta_steps', default=1, type=int, help='test-time-adapt steps')
    parser.add_argument('--n_ctx', default=4, type=int, help='number of tunable tokens')
    parser.add_argument('--ctx_init', default=None, type=str, help='init tunable prompts')
    parser.add_argument('--cocoop', action='store_true', default=False,
                        help="use cocoop's output as prompt initialization")
    parser.add_argument('--load', default=None, type=str, help='path to a pre-trained coop/cocoop')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--tau', type=float, default=0.4, help='temperature of KL loss')
    parser.add_argument('--beta', type=float, default=0.9, help='balance parameter of avg_ent and KL loss')
    parser.add_argument('--uplink_percent', type=float, default=100., help='percentage of upload samples')
    parser.add_argument('--reset', type=float, default=0.)

    stu_ent_list = []
    tea_ent_list = []
    acc_list = []
    num_reset = 0
    main()
    args = parser.parse_args()
    plot_metrics(stu_ent_list, tea_ent_list, acc_list, args)
