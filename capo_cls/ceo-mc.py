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
import matplotlib.pyplot as plt

from clip.custom_clip_device import get_coop
from clip.cocoop import get_cocoop
from data.imagnet_prompts import imagenet_classes
from data.datautils import AugMixAugmenter, build_dataset
from utils.tools import Summary, AverageMeter, ProgressMeter, accuracy, load_model_weight, set_random_seed
from data.cls_to_names import *
from data.fewshot_datasets import fewshot_datasets
from data.imagenet_variants import thousand_k_to_200, imagenet_a_mask, imagenet_r_mask, imagenet_v_mask

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

# 在文件开头或合适位置，定义开启 dropout 的函数
def enable_mc_dropout(model):
    """启用所有 Dropout 层的 train 模式，以便在 eval 时也能随机丢弃."""
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


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


# def test_time_tuning(model, device_model, rec, optimizer_cloud, optimizer_edge, optimizer_student, cloud_scaler, edge_scaler, student_scaler, args):
def test_time_tuning(model, device_model, rec, optimizer_cloud, optimizer_edge, cloud_scaler, edge_scaler, args):
    if args.cocoop:
        image_feature, pgen_ctx = inputs
        pgen_ctx.requires_grad = True
        optimizer = torch.optim.AdamW([pgen_ctx], args.lr)

    selected_idx = None
    # add
    # device_prop_ori = get_param_by_name(model, 'prompt_learner.ctx').clone()
    criterion_div = nn.KLDivLoss(reduction='batchmean')
    for j in range(args.tta_steps):
        # update prompt in cloud
        with torch.cuda.amp.autocast():
            cloud_feat = model.get_image_features(rec)
            if args.cocoop:
                output = model((image_feature, pgen_ctx))
            else:
                output = model(None, cloud_feat)

            if selected_idx is not None:
                output = output[selected_idx]
            else:
                output, selected_idx = select_confident_samples(output, args.selection_p)
            loss_cloud = avg_entropy(output)

        optimizer_cloud.zero_grad()
        cloud_scaler.scale(loss_cloud).backward()
        cloud_scaler.step(optimizer_cloud)
        cloud_scaler.update()

        # update align module in edge
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                output = model(None, cloud_feat)
            del cloud_feat
            edge_feat = device_model.get_image_features(rec)
            output_device = device_model(None, edge_feat)

            # ToDo: temperature T=2.0: output / T
            tau = args.tau
            loss_kl = args.beta * criterion_div(F.log_softmax(output_device / tau, dim=1), F.softmax(output / tau, dim=1))
            output_device, selected_idx = select_confident_samples(output_device, args.selection_p)
            loss_ent = avg_entropy(output_device)

        # optimizer_student.zero_grad()
        # student_scaler.scale(loss_kl+loss_ent).backward(retain_graph=True)
        # student_scaler.step(optimizer_student)
        # student_scaler.update()

        optimizer_edge.zero_grad()
        edge_scaler.scale(loss_kl+loss_ent).backward()
        edge_scaler.step(optimizer_edge)
        edge_scaler.update()

        # update prompt module in edge
        # with torch.cuda.amp.autocast():
        #     output_device = device_model(inputs)
        #     output_device, selected_idx = select_confident_samples(output_device, args.selection_p)
        #     loss_ent = avg_entropy(output_device)
        #
        # optimizer_edge.zero_grad()
        # edge_scaler.scale(loss_ent).backward()
        # edge_scaler.step(optimizer_edge)
        # edge_scaler.update()

    # add
    # param_diffs = get_param_by_name(model, 'prompt_learner.ctx') - device_prop_ori
    # param_diffs = get_param_by_name(model, 'prompt_learner.ctx').detach()

    # with torch.no_grad():
    #     for name, param in device_model.named_parameters():
    #         if name == 'prompt_learner.ctx':
    #             param.add_(param_diffs)

    return loss_ent.item(), loss_kl.item()

@torch.no_grad()
def momentum_update_model(self):
    update_w = self.update_w
    if self.momentum_update:
        self.update_counter += 1
        # reload momentum state_dict
        state_dict = self.clip_model.visual.state_dict()
        for k, v in state_dict.items():
            self.momentum_state_dict[k]= self.momentum * self.momentum_state_dict[k] + (1.0 - self.momentum) * v

        if self.update_counter >= self.update_freq:
            self.update_counter = 0
            for k, v in state_dict.items():
                # self.initial_state_dict[k] = self.momentum_state_dict[k]
                self.initial_state_dict[k] = (1 - update_w) * self.clip_state_dict[k] + update_w * self.momentum_state_dict[k]


def main():
    args = parser.parse_args()
    set_random_seed(args.seed)

    # This codebase has only been tested under the single GPU setting
    assert args.gpu is not None
    main_worker(args.gpu, args)


def main_worker(gpu, args):
    args.gpu = gpu
    set_random_seed(args.seed)
    print("Use GPU: {} for training".format(args.gpu))

    # create model (zero-shot clip model (ViT-L/14@px336) with promptruning)
    if args.test_sets in fewshot_datasets:
        classnames = eval("{}_classes".format(args.test_sets.lower()))
    else:
        classnames = imagenet_classes
    # add
    device_model = None

    model = get_coop(args.cloud_arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init)
    # add
    device_model = get_coop(args.arch, args.test_sets, args.gpu, args.n_ctx, args.ctx_init, type='Device')

    if args.load is not None:
        print("Use pre-trained soft prompt (CoOp) as initialization")
        pretrained_ctx = torch.load(args.load)['state_dict']['ctx']
        assert pretrained_ctx.size()[0] == args.n_ctx
        with torch.no_grad():
            model.prompt_learner[0].ctx.copy_(pretrained_ctx)
            model.prompt_learner[0].ctx_init_state = pretrained_ctx
    model_state = None
    cloud_trainable = []
    edge_trainable = []
    # student_trainable = []
    for name, param in model.named_parameters():
        if "prompt_learner" in name:
            param.requires_grad_(True)
            cloud_trainable.append(param)
        else:
            param.requires_grad_(False)

    # add
    for name, param in device_model.named_parameters():
        if 'prompt_learner' in name:  # prompt_learner
            param.requires_grad_(True)
            edge_trainable.append(param)
        # elif 'attnpool' in name:
        #     param.requires_grad_(True)
        #     student_trainable.append(param)
        # elif 'text_projection' in name:
        #     param.requires_grad_(True)
        #     student_trainable.append(param)
        else:
            param.requires_grad_(False)


    # device_model.text_encoder.transformer.eval()
    # device_model.text_encoder.ln_final.eval()
    # import pdb
    # pdb.set_trace()
    # for name, param in device_model.named_parameters():
    #     if param.requires_grad:
    #         print(name)

    print("=> Device Model created: visual backbone {}".format(args.arch))
    print("=> Cloud Model created: visual backbone {}".format(args.cloud_arch))

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    else:
        assert args.gpu is not None
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # add
        device_model = device_model.cuda(args.gpu)

    # define optimizer

    # trainable_param = model.prompt_learner.parameters()

    # optimizer = torch.optim.AdamW(trainable_param, args.lr)
    # add
    # edge_trainable = [param for param in model.parameters() if param.requires_grad]
    # student_trainable = [param for param in device_model.parameters() if param.requires_grad]
    optimizer_cloud = optim.AdamW(cloud_trainable, args.lr)
    optimizer_edge = optim.AdamW(edge_trainable, args.lr)
    # optimizer_student = optim.AdamW(student_trainable, args.lr)

    # optim_state = deepcopy(optimizer.state_dict())
    # add
    cloud_optim_state = deepcopy(optimizer_cloud.state_dict())
    edge_optim_state = deepcopy(optimizer_edge.state_dict())
    # student_optim_state = deepcopy(optimizer_student.state_dict())

    # setup automatic mixed-precision (Amp) loss scaling
    # scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    # add
    cloud_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    edge_scaler = torch.cuda.amp.GradScaler(init_scale=1000)
    # student_scaler = torch.cuda.amp.GradScaler(init_scale=1000)


    print('=> Using native Torch AMP. Training in mixed precision.')

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
        # else:
            # data_transform = transforms.Compose([
            #     transforms.Resize(args.resolution, interpolation=BICUBIC),
            #     transforms.CenterCrop(args.resolution),
            #     transforms.ToTensor(),
            #     normalize,
            # ])
            # batchsize = args.batch_size

        print("evaluating: {}".format(set_id))
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

        model.reset_classnames(classnames, args.cloud_arch)
        # add
        device_model.reset_classnames(classnames, args.arch)

        val_dataset = build_dataset(set_id, device_transform, data_transform, args.data, mode=args.dataset_mode)
        print("number of test samples: {}".format(len(val_dataset)))
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batchsize, shuffle=True,
            num_workers=args.workers, pin_memory=True)

        # results[set_id] = test_time_adapt_eval(val_loader, model, model_state, optimizer, optim_state, scaler, args)
        # results[set_id] = test_time_adapt_eval(val_loader, model, device_model, optimizer_cloud, optimizer_edge, optimizer_student, cloud_optim_state, edge_optim_state, student_optim_state, cloud_scaler, edge_scaler, student_scaler, args)
        results[set_id] = test_time_adapt_eval(val_loader, model, device_model, optimizer_cloud, optimizer_edge, cloud_optim_state, edge_optim_state, cloud_scaler, edge_scaler, args)

        del val_dataset, val_loader
        try:
            print("=> Acc. on testset [{}]: @1 {}/ @5 {}".format(set_id, results[set_id][0], results[set_id][1]))
        except:
            print("=> Acc. on testset [{}]: {}".format(set_id, results[set_id]))

    print("======== Result Summary ========")
    print("params: nstep	lr	bs")
    print("params: {}	{}	{}".format(args.tta_steps, args.lr, args.batch_size))
    print("\t\t [set_id] \t\t Top-1 acc. \t\t Top-5 acc.")
    for id in results.keys():
        print("{}".format(id), end="	")
    print("\n")
    for id in results.keys():
        print("{:.2f}".format(results[id][0]), end="	")
    print("\n")


def test_time_adapt_eval(val_loader, model, device_model, optimizer_cloud, optimizer_edge, cloud_optim_state, edge_optim_state, cloud_scaler, edge_scaler, args):
    batch_time = AverageMeter('Time', ':6.3f', Summary.NONE)
    top1 = AverageMeter('Acc@1', ':6.2f', Summary.AVERAGE)
    top5 = AverageMeter('Acc@5', ':6.2f', Summary.AVERAGE)
    # all_entropy = deque(maxlen=500)
    all_vars = []
    # results = []
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    print(f"Test with tau={args.tau}, beta={args.beta}, uplink percentage={args.uplink_percent}%:")
    # reset model and switch to evaluate mode
    model.eval()
    device_model.eval()
    with torch.no_grad():
        model.reset()
        # add
        device_model.reset()
    end = time.time()
    for i, (oris, recs, target) in enumerate(val_loader):
    # device_proj_ori = get_param_by_name(device_model, 'text_projection').clone()
    # for i, (images, target) in enumerate(val_loader):

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
            # rec = recs
        target = target.cuda(args.gpu, non_blocking=True)
        recs = torch.cat(recs, dim=0)
            # oris = torch.cat(oris, dim=0)
        # reset the tunable prompt to its initial state
        with torch.no_grad():
            ori_img_features = device_model.get_image_features(ori)
        # 在 test_time_adapt_eval 的循环体里，替换原有 entropy 计算
        # 假设我们选择 M 次前向
        M = args.mc_samples  # e.g. 10
        enable_mc_dropout(device_model)  # 在 eval 前调用一次

        # 取多次 output
        logits_list = []
        for _ in range(M):
            with torch.no_grad():
                out = device_model(None, ori_img_features)  # [batch_size, num_classes]
            logits_list.append(out.unsqueeze(0))           # 在第 0 维堆叠

        # logits_tensor: [M, batch_size, num_classes]
        logits_tensor = torch.cat(logits_list, dim=0)

        # 计算均值 logits_mean: [batch_size, num_classes]
        logits_mean = logits_tensor.mean(dim=0)

        # 计算方差 per-sample: sigma2 = (1/M) * sum_m (f_m - f_mean)^2
        # 得到 [batch_size, num_classes] 再沿类别维度求平均，得到 [batch_size]
        var_per_sample = ((logits_tensor - logits_mean.unsqueeze(0))**2).mean(dim=(0,2))

        # 如果 batch_size=1，就直接取 item()
        sample_var = var_per_sample.item()
        all_vars.append(sample_var)

        # 用方差来算阈值
        threshold = np.percentile(all_vars, (100. - args.uplink_percent))

        is_uplink = sample_var > threshold

        if is_uplink:
            with torch.no_grad():
                model.reset()
                # add
                device_model.reset()
                
            optimizer_cloud.load_state_dict(cloud_optim_state)
            optimizer_edge.load_state_dict(edge_optim_state)
            _, __ = test_time_tuning(model, device_model, recs, optimizer_cloud, optimizer_edge, cloud_scaler, edge_scaler, args)

            # device_model.eval()
            # if i % 100 == 0:
            #     iter_list.append(i)
            #     teacher_loss_list.append(loss_ent)
            #     student_loss_list.append(loss_kl)

        # The actual inference goes here
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                # output = model(image)
                device_model.eval()
                output = device_model(None, ori_img_features)
                # output = device_model(image)
                        # add
        # device_model.momentum_update_model()

        # for n, p in device_model.named_parameters():
        #     # if 'prompt_learner' in n:
        #     param = p.detach().cpu().numpy()
        #     np.save(f".utils/{n}", param)
        # import pdb
        # pdb.set_trace()
        #     elif 'prompt_learner' in n:
        #         prompt_param = p.detach().cpu().numpy()
        #         np.save("./prompt_device.npy", prompt_param)
        # for n, p in model.named_parameters():
        #     if 'prompt_learner' in n:
        #         prompt_param = p.detach().cpu().numpy()
        #         np.save("./prompt_cloud.npy", prompt_param)
        # import pdb
        # pdb.set_trace()
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        top1.update(acc1[0], ori.size(0))
        top5.update(acc5[0], ori.size(0))
        # if i % 100 == 0:
        #     acc_list.append(top1.avg.item())

        if (i + 1) % args.print_freq == 0:
            progress.display(i)

    progress.display_summary()
    # import pandas as pd
    # df = pd.DataFrame(results)
    # df.to_csv("entropy_confidence_analysis_{}.csv".format(args.uplink_percent), index=False)
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
    parser.add_argument('-p', '--print-freq', default=5000, type=int,
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
    parser.add_argument('--mc_samples', type=int, default=10, help='number of mc samples')

    # iter_list = []
    # teacher_loss_list = []
    # student_loss_list = []
    # acc_list = []
    main()

    # draw
    # args = parser.parse_args()
    #
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 10))
    #
    # ax1.plot(iter_list, teacher_loss_list, label='Teacher Loss', color='blue', marker='o')
    # ax1.set_xlabel('Iterations')
    # ax1.set_ylabel('Loss', color='black')
    # ax1.tick_params(axis='y', labelcolor='black')
    # ax1.set_title('Teacher Loss and Accuracy')
    # ax1.legend(loc='upper left')
    # ax1.grid()
    #
    # ax1_acc = ax1.twinx()
    # ax1_acc.plot(iter_list, acc_list, label='Student Accuracy', color='green', marker='o')
    # ax1_acc.set_ylabel('Student Accuracy', color='green')
    # ax1_acc.tick_params(axis='y', labelcolor='green')
    # ax1_acc.legend(loc='upper right')
    #
    # ax2.plot(iter_list, student_loss_list, label='Student Loss', color='blue', marker='o')
    # ax2.set_xlabel('Iterations')
    # ax2.set_ylabel('Loss', color='black')
    # ax2.tick_params(axis='y', labelcolor='black')
    # ax2.set_title('Student Loss and Accuracy')
    # ax2.legend(loc='upper left')
    # ax2.grid()
    #
    # ax2_acc = ax2.twinx()
    # ax2_acc.plot(iter_list, acc_list, label='Student Accuracy', color='green', marker='o')
    # ax2_acc.set_ylabel('Student Accuracy', color='green')
    # ax2_acc.tick_params(axis='y', labelcolor='green')
    # ax2_acc.legend(loc='upper right')
    #
    # # 保存图形
    # plt.tight_layout()
    # plt.savefig(f'logs/training_loss_acc_{args.test_sets}.png', dpi=300)
    # with open(f"logs/loss_kl_{args.test_sets}.txt", "w") as f:
    #     for item in student_loss_list:
    #         f.write(f'{item}\n')
    # with open(f"logs/loss_ent_{args.test_sets}.txt", "w") as f:
    #     for item in teacher_loss_list:
    #         f.write(f'{item}\n')
    # with open(f"logs/acc_student_{args.test_sets}.txt", "w") as f:
    #     for item in acc_list:
    #         f.write(f'{item}\n')
    # print("saved!")
