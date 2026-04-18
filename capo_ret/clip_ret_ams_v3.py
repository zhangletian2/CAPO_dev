# coding=utf-8
import os
import json
import time
import random
import logging
import datetime
from tqdm import tqdm
from copy import deepcopy

import torch
import numpy as np
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim.optimizer import Optimizer

import lavis.tasks as tasks
from lavis.common.config import Config
from lavis.common.dist_utils import get_rank, init_distributed_mode
from lavis.common.logger import setup_logger
from lavis.common.utils import now
from lavis.common.logger import MetricLogger

from lavis.datasets.builders import *
from lavis.models import *
from lavis.processors import *
from lavis.runners.runner_base import RunnerBase
from lavis.tasks import *

from params import parse_args
from clip_reward import get_reward_model
from lavis_evaluate import setup_seeds
from custom_models import CLIPRet_TTA

from clip_ret_policy import tokenize_all_text, get_all_text_embeds, get_all_image_embeds


def kd_distill_loss_v2(logits_student, logits_teacher, T_stu=1.0, T_tea=1.0):
    """
    vanilla KD, KLDiv between teacher and student, only the gradient related part
    """
    log_pred_student = F.log_softmax(logits_student / T_stu, dim=1)
    pred_teacher = F.softmax(logits_teacher / T_tea, dim=1)
    # kl_div = -p log q
    loss_kd = - torch.sum(pred_teacher * log_pred_student, dim=1).mean()
    loss_kd = loss_kd * T_stu * T_stu

    return loss_kd


def tune_image(image, model, reward_model, optimizer, scaler, args=None):
    """tune function for test time adaptation (image encoder)"""
    model.train()

    # knowledge distillation for single sample
    reward_model.set_image_features(images=image)
    r_logits_per_image, r_logits_per_text = reward_model.calulate_similarity()

    for step in range(args.tta_steps):    
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(image)
            loss = kd_distill_loss_v2(logits_per_image, r_logits_per_image)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def tune_text(text, model, reward_model, optimizer, scaler, args=None):
    """
    tune function for test time adaptation (text encoder)
    Args:
        text: raw text
    """
    model.train()

    # knowledge distillation for single sample
    reward_model.set_text_features(captions=text)
    r_logits_per_image, r_logits_per_text = reward_model.calulate_similarity()

    for step in range(args.tta_steps):   
        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text = model(images=None, text=text)
            loss = kd_distill_loss_v2(logits_per_text, r_logits_per_text)

        # compute gradient and do SGD step
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()


def test_time_tune(data_loader, device, model, reward_model=None, scaler=None, optimizer=None, optim_state=None, text_bs=128, args=None):
    """policy gradient when playing image to text policy"""
    model.eval()
    device = model.device

    # results matrix
    score_matrix_i2t = torch.full((len(data_loader.dataset.image), len(data_loader.dataset.text)), -100.0).to(device)
    score_matrix_t2i = torch.full((len(data_loader.dataset.text), len(data_loader.dataset.image)), -100.0).to(device)
    image_filenames = [os.path.join(data_loader.dataset.vis_root, filename) for filename in data_loader.dataset.image]

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            text_ids = tokenize_all_text(data_loader.dataset.text, model, text_bs)
            if model.only_visual:
                text_embeds = get_all_text_embeds(text_ids, model, text_bs)
                model.set_text_features(text_features=text_embeds)
                reward_model.set_many_text_features(data_loader.dataset.text, text_bs=text_bs)
            else:
                image_embeds = get_all_image_embeds(data_loader, model)
                model.set_image_features(image_features=image_embeds)
                reward_model.set_image_features_with_dataloder(data_loader)

    if model.only_visual:
        for i, samples in tqdm(enumerate(data_loader), ncols=150):
            image = samples["image1"].to(model.device)
            image_cloud = samples["image2"].to(model.device)
            tune_image(image_cloud, model, reward_model, optimizer, scaler, args=args)

            # after TTA, do evaluation
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    # logits_per_image contains multiplication of logit_scale, it should not affect the retrieva results
                    logits_per_image, _ = model(image)
                score_matrix_i2t[i] = logits_per_image[0]

            # reset/update parameters
            model.momentum_update_model()
            model.reset_initial()
            # reset the optimizer state
            optimizer.load_state_dict(optim_state)
    else:
        for i, text in tqdm(enumerate(data_loader.dataset.text), ncols=150):
            tune_text(text, model, reward_model, optimizer, scaler, args=args)

            # after TTA, do evaluation
            model.eval()
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    _, logits_per_text = model(images=None, text=text)
                score_matrix_t2i[i] = logits_per_text[0]

            model.momentum_update_model()
            model.reset_initial()
            optimizer.load_state_dict(optim_state)

    return score_matrix_i2t.detach().cpu().numpy(), score_matrix_t2i.detach().cpu().numpy()

class GradientGuidedAdam(Optimizer):
    def __init__(self, params, lr=5e-3, betas=(0.9, 0.999), eps=1e-8, gamma=0.1, T_horizon=10):
        """
        Args:
            params (iterable): Iterable of parameters to optimize or dicts defining parameter groups.
            lr (float): Learning rate.
            betas (Tuple[float, float]): Coefficients used for computing running averages of gradient and its square.
            eps (float): Term added to the denominator to improve numerical stability.
            gamma (float): Fraction of parameters to be updated based on gradient magnitudes.
            T_horizon (int): Horizon for selecting data (can be adapted for batch-wise simulations).
        """
        defaults = dict(lr=lr, betas=betas, eps=eps, gamma=gamma, T_horizon=T_horizon)
        super(GradientGuidedAdam, self).__init__(params, defaults)
        # self.total_updated_params = 0

    def step(self, closure=None):
        """
        Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()
        # cnt = 0
        for group in self.param_groups:
            # Hyperparameters
            lr = group['lr']
            betas = group['betas']
            eps = group['eps']
            gamma = group['gamma']

            for param in group['params']:
                if param.grad is None:
                    continue

                # State initialization
                state = self.state[param]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(param.data)  # First moment vector
                    state['v'] = torch.zeros_like(param.data)  # Second moment vector
                    state['u'] = torch.zeros_like(param.data)  # Adam update vector

                m, v, u = state['m'], state['v'], state['u']
                grad = param.grad.data

                # Update biased first moment estimate
                m.mul_(betas[0]).add_(grad, alpha=1 - betas[0])

                # Update biased second raw moment estimate
                v.mul_(betas[1]).addcmul_(grad, grad, value=1 - betas[1])

                # Compute bias-corrected first and second moment estimates
                state['step'] += 1
                bias_correction1 = 1 - betas[0] ** state['step']
                bias_correction2 = 1 - betas[1] ** state['step']

                # Compute the Adam update
                denom = (v.sqrt() / bias_correction2).add_(eps)
                u.copy_(m / bias_correction1 / denom)

                # Masking updates based on gamma fraction
                if 'mask' not in state or state['step'] % group['T_horizon'] == 0:
                    # Select top `gamma` fraction of parameters based on |u|
                    _, indices = torch.topk(u.abs().flatten(), int(gamma * u.numel()))
                    mask = torch.zeros_like(u, dtype=torch.bool).flatten()
                    mask[indices] = True
                    state['mask'] = mask.view_as(u)
                    # self.total_updated_params += mask.sum().item()

                # Apply the updates
                param.data.addcmul_(u, state['mask'].float(), value=-lr)

        return loss


def main():
    # allow auto-dl completes on main process without timeout when using NCCL backend.
    # os.environ["NCCL_BLOCKING_WAIT"] = "1"
    # set before init_distributed_mode() to ensure the same job_id shared across all ranks.
    job_id = now()
    args = parse_args()
    print('\n job_ID {}: \n'.format(job_id))

    cfg = Config(args)

    init_distributed_mode(cfg.run_cfg)
    setup_seeds(cfg)
    # set after init_distributed_mode() to only log on master.
    setup_logger()

    cfg.pretty_print()
    task = tasks.setup_task(cfg)
    datasets = task.build_datasets(cfg)
    model = task.build_model(cfg)

    runner = RunnerBase(cfg=cfg, job_id=job_id, task=task, model=model, datasets=datasets)
    dataloader = runner.dataloaders['test']
    device = runner.model.device

    # setup automatic mixed-precision (Amp) loss scaling
    scaler = torch.cuda.amp.GradScaler(init_scale=1000)

    # create model
    model = CLIPRet_TTA(device, arch=args.arch, only_visual=(args.retrieval_task == "image2text"),
                            momentum_update=args.momentum_update, update_freq=args.update_freq,
                            update_w=args.update_w, momentum=args.tta_momentum)
    model = model.to(device)

    # define the CLIPRewards
    reward_model = get_reward_model(device, args)

    # https://huggingface.co/docs/transformers/main_classes/optimizer_schedules#transformers.AdamW
    optimizer = GradientGuidedAdam(model.parameters(), lr=args.lr, eps=1e-06)
    optim_state = deepcopy(optimizer.state_dict())

    # score_matrix_i2t, score_matrix_t2i = image_to_text_policy(dataloader, runner.model, optimizer, task.cfg.k_test, 128, args=args)
    score_matrix_i2t, score_matrix_t2i = test_time_tune(dataloader, device, model, reward_model=reward_model,
                                                scaler=scaler, optimizer=optimizer, optim_state=optim_state, text_bs=128, args=args)

    eval_result = task._report_metrics(score_matrix_i2t, score_matrix_t2i,
                                        dataloader.dataset.txt2img, dataloader.dataset.img2txt)

    # print and output
    output_filename = os.path.join(args.output, "results_{}.json".format(args.retrieval_task))
    logging.info(output_filename)
    for k, v in eval_result.items():
        eval_result[k] = round(v, 3)
    logging.info(eval_result)
    # save output
    with open(output_filename, 'w') as fp:
        json.dump(eval_result, fp, indent=4)


if __name__ == "__main__":
    main()
