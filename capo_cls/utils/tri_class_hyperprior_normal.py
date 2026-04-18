import argparse
import torch
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision as tv
import visualpriors
import subprocess
import glob
import cv2
from torch import dtype, nn, optim
from torch.autograd import Function
import time
import numpy as np
import os
from visualpriors_code.tasko_util import BPEncoder, BPDecoder, TaskonomyEncoder
import skimage

class BypassRound(Function):
    @staticmethod
    def forward(ctx, inputs):
        return torch.round(inputs)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

bypass_round = BypassRound.apply

class MultiTaskoDataset_MultiRes(torch.utils.data.Dataset):
    def __init__(self, data_glob, root_path, label_globs, loaders, transforms, batchsize=16):
        super(MultiTaskoDataset_MultiRes, self).__init__()
        self.transforms = transforms
        self.data_imgs = sorted(glob.glob(data_glob))
        self.label_ids = []
        for i in label_globs:
            self.label_ids.append(i.split('/')[-2].rstrip('traines')[:-1])
        self.label_imgs_list = []
        for i in label_globs:
            self.label_imgs_list.append(sorted(glob.glob(f'{root_path}/{i}')))
        self.loaders = loaders
        self.black_list = {}
        self.bsz = batchsize
        
    def __getitem__(self, idx):
        rsz_level = (idx // self.bsz) * 13 % 8
        sz = rsz_level * 32 + 128
        for j in range(100):
          try:
            data_p = self.data_imgs[idx]
            data_img = TF.resize(Image.open(data_p), sz)
            data_img = self.transforms[0](data_img)
            
            ret = {}
            ret['data'] = data_img

            data_img_256 = Image.open(data_p)
            data_img_256 = self.transforms[-1](data_img_256)
            ret['data_256'] = data_img_256
            
            for i in range(len(self.label_ids)):
                label_p = self.label_imgs_list[i][idx]
                label_img = self.loaders[i+1](label_p)
                label_img = self.transforms[i+1](label_img)
                ret[self.label_ids[i]] = label_img
            return ret
          except:
              idx += 1
              self.black_list[data_p] = 1
              continue

    def __len__(self):
        return len(self.data_imgs)

class MultiTaskoDataset(torch.utils.data.Dataset):
    def __init__(self, data_glob, root_path, label_globs, loaders, transforms):
        super(MultiTaskoDataset, self).__init__()
        self.transforms = transforms
        self.data_imgs = sorted(glob.glob(data_glob))
        self.label_ids = []
        for i in label_globs:
            self.label_ids.append(i.split('/')[-2].rstrip('traines')[:-1])
        self.label_imgs_list = []
        for i in label_globs:
            self.label_imgs_list.append(sorted(glob.glob(f'{root_path}/{i}')))
        self.loaders = loaders
        self.black_list = {}
        
    def __getitem__(self, idx):
        for j in range(100):
          try:
            data_p = self.data_imgs[idx]
            data_img = Image.open(data_p)
            data_img = self.transforms[0](data_img)
            ret = {}
            ret['data'] = data_img
            data_img_256 = Image.open(data_p)
            data_img_256 = self.transforms[-1](data_img_256)
            ret['data_256'] = data_img_256

            for i in range(len(self.label_ids)):
                label_p = self.label_imgs_list[i][idx]
                label_img = self.loaders[i+1](label_p)
                label_img = self.transforms[i+1](label_img)
                ret[self.label_ids[i]] = label_img
            return ret
          except:
              idx += 1
              self.black_list[data_p] = 1
            #   print(self.data_imgs[idx])
            #   for i in range(len(self.label_ids)):
            #     print(self.label_imgs_list[i][idx])
              continue

    def __len__(self):
        return len(self.data_imgs)

class TransformMerge(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TransformMerge, self).__init__()
        self.transform = nn.Sequential(
          nn.Conv2d(in_dim, in_dim*2, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(in_dim*2, out_dim, 3, 1, 1),
        )
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class TransformSplit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TransformSplit, self).__init__()
        self.transform = nn.Sequential(
          nn.Conv2d(in_dim, 512, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(512, 256, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(256, 128, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(128, out_dim, 3, 1, 1)
        )
    def forward(self, inputs):
        x = self.transform(inputs)
        return x
    
class TransformEnc(nn.Module):
    def __init__(self, in_dim):
        super(TransformEnc, self).__init__()
        self.transform = nn.Sequential(
          nn.Conv2d(in_dim, 1024, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(1024, 1024, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(1024, 512, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(512, 256, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(256, 128, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(128, 128, 3, 1, 1)
        )
        # we go rounding afterwards
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class TransformDec(nn.Module):
    def __init__(self, out_dim):
        super(TransformDec, self).__init__()
        self.transform = nn.Sequential(
          nn.Conv2d(128, 1024, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(1024, 1024, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(1024, 512, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(512, 256, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(256, 128, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(128, out_dim, 3, 1, 1)
        )
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class TransformSplit(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TransformSplit, self).__init__()
        self.transform = nn.Sequential(
          nn.Conv2d(in_dim, 512, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(512, 256, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(256, 128, 3, 1, 1),
          nn.ReLU(),
          nn.Conv2d(128, out_dim, 3, 1, 1)
        )
    def forward(self, inputs):
        x = self.transform(inputs)
        return x
              
class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()
        self.m_normal_dist = torch.distributions.normal.Normal(0., 1.)

    def _cumulative(self, inputs, stds, mu):
        half = 0.5
        eps = 1e-6
        upper = (inputs - mu + half) / (stds)
        lower = (inputs - mu - half) / (stds)
        cdf_upper = self.m_normal_dist.cdf(upper)
        cdf_lower = self.m_normal_dist.cdf(lower)
        res = cdf_upper - cdf_lower
        return res

    def forward(self, inputs, hyper_sigma, hyper_mu):
        likelihood = self._cumulative(inputs, hyper_sigma, hyper_mu)
        likelihood_bound = 1e-8
        likelihood = torch.clamp(likelihood, min=likelihood_bound)
        return likelihood

class h_analysisTransformModel(nn.Module):
    def __init__(self, in_dim):
        super(h_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class h_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim):
        super(h_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class h2_analysisTransformModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(h2_analysisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 512, 3, 1, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, inputs):
        x = self.transform(inputs)
        return x

class h2_synthesisTransformModel(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(h2_synthesisTransformModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, out_dim)
        )
    def forward(self, inputs):
        x = self.transform(inputs)
        return x
    

class PredictionModel(nn.Module):
    def __init__(self, in_dim, out_dim=128):
        super(PredictionModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, out_dim*2, 3, 1, 1)
        )
        self.out_dim = out_dim

    def forward(self, inputs):
        x = self.transform(inputs)
        c = self.out_dim
        hyper_mu = x[:, :c, :, :]
        hyper_sigma = x[:, c:, :, :]
        hyper_sigma = torch.exp(hyper_sigma)
        return hyper_mu, hyper_sigma

class VectorPredictionModel(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(VectorPredictionModel, self).__init__()
        self.transform = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(mid_dim, mid_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(mid_dim, out_dim*2)
        )
    def forward(self, inputs, out_shape):
        x = self.transform(inputs)
        x_shape = x.shape
        x = torch.reshape(x, (x_shape[0], x_shape[1], 1, 1))
        x = x.repeat((1,1,out_shape[-2], out_shape[-1]))
        c = x_shape[1]//2
        hyper_mu = x[:, :c, :, :]
        hyper_sigma = x[:, c:, :, :]
        hyper_sigma = torch.exp(hyper_sigma)
        return hyper_mu, hyper_sigma

class CodeBook(nn.Module):
    def __init__(self, size=512):
        super(CodeBook, self).__init__()
        codebook = torch.nn.Parameter(
                torch.randn((size, 16, 16), dtype=torch.float32, requires_grad=True)
            )
        self.register_parameter('codebook', codebook)
    
    def forward(self, indices):
        # assume indices to have shape N, c, size
        s = indices.shape
        x = torch.reshape(indices, (s[0]*s[1], s[2]))
        x = torch.einsum('nk,kab->nab', x, self.codebook)
        x = torch.reshape(x, (s[0], s[1], 16, 16))
        return x

class CodeBookPrediction(nn.Module):
    def __init__(self, in_dim, mid_dim, out_dim):
        super(CodeBookPrediction, self).__init__()
        self.codebook_module = CodeBook(size=512)
        self.transform = nn.Sequential(
            nn.Conv2d(in_dim, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, out_dim*2, 3, 1, 1)
        )
        self.out_dim = out_dim
    def forward(self, inputs, out_shape):
        x = self.codebook_module(inputs)
        x = nn.functional.interpolate(x, out_shape[-2:])
        x = self.transform(x)
        c = self.out_dim
        hyper_mu = x[:, :c, :, :]
        hyper_sigma = x[:, c:, :, :]
        hyper_sigma = torch.exp(hyper_sigma)
        return hyper_mu, hyper_sigma

class Net(nn.Module):
    def __init__(self, in_dim):
        super(Net, self).__init__()

        self.tasko_enc1 = BPEncoder(task_name='normal', eval_only=True)
        self.tasko_enc2 = BPEncoder(task_name='reshading', eval_only=True)
        self.tasko_enc3 = BPEncoder(task_name='curvature', eval_only=True)

        self.tasko_dec1 = BPDecoder(task_name='normal', is_decoder_mlp=False, out_channels=3, eval_only=True)
        self.tasko_dec2 = BPDecoder(task_name='reshading', is_decoder_mlp=False, out_channels=1, eval_only=True)
        self.tasko_dec3 = BPDecoder(task_name='curvature', is_decoder_mlp=False, out_channels=2, eval_only=True)

        self.merge_conv1 = TransformMerge(in_dim, in_dim)
        self.merge_conv2 = TransformMerge(in_dim, in_dim)
        self.merge_conv3 = TransformMerge(in_dim, in_dim)
        
        self.encoder = TransformEnc(in_dim*3) # output dim=128
        self.entropy_bottleneck_z2 = GaussianModel()
        self.entropy_bottleneck_z1 = GaussianModel()
        
        self.ha_model = h_analysisTransformModel(128)
        self.hs_model = h_synthesisTransformModel(32)
        self.pred_h2 = PredictionModel(64)

        self.z1_sigma = torch.nn.Parameter(torch.ones(
            (1, 32, 1, 1), dtype=torch.float32, requires_grad=True))
        self.register_parameter('z1_sigma', self.z1_sigma)
        
        self.decoder = TransformDec(in_dim*3)
        self.split_conv1 = TransformSplit(in_dim*3, in_dim)
        self.split_conv2 = TransformSplit(in_dim*3, in_dim)
        self.split_conv3 = TransformSplit(in_dim*3, in_dim)
    
    def forward(self, inputs, mode='train', sanity=False, as_mod=False):
        feat1 = self.tasko_enc1(inputs)
        feat2 = self.tasko_enc2(inputs)
        feat3 = self.tasko_enc3(inputs)
        
        feat1x = self.merge_conv1(feat1)
        feat2x = self.merge_conv2(feat2)
        feat3x = self.merge_conv3(feat3)
        
        feat = torch.cat([feat1x, feat2x, feat3x], 1)
        
        z2 = self.encoder(feat)
        z2_rounded = bypass_round(z2)
        noise = torch.rand_like(z2_rounded) - 0.5
        z2_noisy = z2 + noise

        z1 = self.ha_model(z2_rounded)
        z1_rounded = bypass_round(z1)
        noise = torch.rand_like(z1_rounded) - 0.5
        z1_noisy = z1 + noise

        h1 = self.hs_model(z1_rounded)

        z1_sigma = torch.abs(self.z1_sigma)
        z1_sigma = torch.clamp(z1_sigma, min = 1e-6)
        
        z1_mu = torch.zeros_like(z1_sigma)
        z2_mu, z2_sigma = self.pred_h2(h1) 
        z2_sigma = torch.clamp(torch.abs(z2_sigma), min = 1e-6)
        
        if mode == 'train':
            z1_likelihood = self.entropy_bottleneck_z1(z1_noisy, z1_sigma, z1_mu)
            z2_likelihood = self.entropy_bottleneck_z2(z2_noisy, z2_sigma, z2_mu)
        elif mode == 'test':
            z1_likelihood = self.entropy_bottleneck_z1(z1_rounded, z1_sigma, z1_mu)
            z2_likelihood = self.entropy_bottleneck_z2(z2_rounded, z2_sigma, z2_mu)
            
        num_pixels = 256**2 * inputs.shape[0]

        bpp_list = [
            torch.log(l).sum() / (-np.log(2) * num_pixels) for l in [z1_likelihood, z2_likelihood]
        ]
        
        train_bpp = bpp_list[0] + bpp_list[1]

        if as_mod:
            return z3_rounded, train_bpp

        h2 = self.decoder(z2_rounded)
        f1 = self.split_conv1(h2)
        f2 = self.split_conv2(h2)
        f3 = self.split_conv3(h2)

        pred1 = self.tasko_dec1(f1)
        pred2 = self.tasko_dec2(f2)
        pred3 = self.tasko_dec3(f3)

        if sanity:
            gt_pred1 = self.tasko_dec1(feat1)
            gt_pred2 = self.tasko_dec2(feat2)
            gt_pred3 = self.tasko_dec3(feat3)

            return pred1, pred2, pred3, train_bpp, bpp_list[0], bpp_list[1], gt_pred1, gt_pred2, gt_pred3
        else:
            return pred1, pred2, pred3, train_bpp, bpp_list[0], bpp_list[1]


class Preprocess(object):
    def __init__(self, rg=255):
        self.rg = rg

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        img /= (self.rg/2)
        img -= 1.0
        return img.transpose((2, 0, 1))

def rescale_image(im, new_scale=[-1.,1.], current_scale=None, no_clip=False):
    im = skimage.img_as_float(im).astype(np.float32)
    if current_scale is not None:
        min_val, max_val = current_scale
        if not no_clip:
            im = np.clip(im, min_val, max_val)
        im = im - min_val
        im /= (max_val - min_val) 
    min_val, max_val = new_scale
    im *= (max_val - min_val)
    im += min_val
    return im 

def curvature_preprocess(img):
    img = img[:,:,:2]
    img = img - [123.572, 120.1]
    img = img / [31.922, 21.658]
    return img

def curvature_reconstruct(img):
    im = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.float32)
    im[:,:,0:2] = img[:,:,:]
    im[:,:,0:2] *= [31.922, 21.658]
    im[:,:,0:2] += [123.572, 120.1]
    im = np.clip(im, 0, 255)
    im = im.astype(np.uint8)
    return im

def normal_reconstruct(img):
    im = (img + 1) * 127.5
    im = np.clip(im, 0, 255)
    im = im.astype(np.uint8)
    return im


class KeyPreprocess(object):
    def __init__(self, current_scale=(0,0.005)):
        self.current_scale = current_scale
    
    def __call__(self, PIL_img):
        d = np.asarray(PIL_img, dtype=np.uint16)
        d = rescale_image(d, current_scale=self.current_scale)
        d = np.expand_dims(d, 0)
        return d

class TransformCurvature(object):
    def __init__(self):
        self.a = np.array([123.572, 120.1], dtype=np.float32)
        self.b = np.array([31.922, 21.658], dtype=np.float32)

    def __call__(self, PIL_img):
        d = np.asarray(PIL_img, dtype=np.float32)[:,:,:2]
        d -= self.a
        d /= self.b
        # d = np.expand_dims(d, 0)
        return d.transpose((2, 0, 1))

class TransformReshading(object):
    def __init__(self, rg=255):
        self.rg = rg

    def __call__(self, PIL_img):
        img = np.asarray(PIL_img, dtype=np.float32)[:,:,0:1]
        if len(img.shape) == 2:
            img = np.expand_dims(img, -1)
        img /= (self.rg/2)
        img -= 1.0
        return img.transpose((2, 0, 1))

class TransformVector(object):
    def __init__(self):
        pass
    def __call__(self, vec):
        val = np.argmax(vec)
        return val

class TransformIdentity(object):
    def __init__(self):
        pass
    def __call__(self, vec):
        return vec

def train():
    lmbda = args.lmbda
    lmbda_fid = args.lmbda_fid
    
    train_label_globs = [
        'normal_train/*.png', 'reshading_train/*.png', 'principal_curvature_train/*.png'
    ]
    test_label_globs = [
        'normal_test/*.png', 'reshading_test/*.png', 'principal_curvature_test/*.png'
    ]
    loaders = [
        Image.open, Image.open, Image.open, Image.open 
    ]

    train_dataset = MultiTaskoDataset(
        data_glob=args.train_data,
        root_path=args.root_path,
        label_globs=train_label_globs,
        loaders=loaders,
        transforms=[
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ), # for rgb
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        TransformReshading()
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        TransformCurvature(),
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ), # for rgb 256
            ]
        )
    training_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batchsize, shuffle=True, num_workers=8
    )

    test_dataset = MultiTaskoDataset(
        data_glob=args.val_data,
        root_path=args.root_path,
        label_globs=test_label_globs,
        loaders=loaders,
        transforms=[
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ), # for rgb
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        TransformReshading()
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        TransformCurvature(),
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ), # for rgb 256
            ]
        )
    testing_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=8, drop_last=True
    )

    device = torch.device('cuda')
    net = nn.DataParallel(Net(8).cuda())
    # net = Net(8).cuda()
    if not args.load_weights == "":
        net.load_state_dict(torch.load(args.load_weights))
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    CELoss = nn.CrossEntropyLoss()

    opt = optim.Adam(net.parameters(), lr=args.lr)
    sch = optim.lr_scheduler.MultiStepLR(opt, [100,200,250,275], 0.5)

    def cross_entropy(p, q):
        return -1 * p * torch.log(q)

    # sanity check
    with torch.no_grad():
        list_distort1 = 0.
        cnt = 0
        list_acc = 0.
        list_distort2 = 0.
        list_distort3 = 0.
        list_distort4 = 0.
        for d in testing_loader:
            cnt += 1
            data = d['data']
            soft_l1 = d['normal']
            soft_l2 = d['reshading']
            soft_l3 = d['principal_curvature']
            

            data = data.cuda()
            soft_l1 = soft_l1.cuda()
            soft_l2 = soft_l2.cuda()
            soft_l3 = soft_l3.cuda()

            pred1, pred2, pred3, train_bpp, b1, b2, gt_pred1, gt_pred2, gt_pred3 = net(data, 'test', sanity=True)
            gt_pred1 = visualpriors.feature_readout(data, 'normal', device=device)
            gt_pred2 = visualpriors.feature_readout(data, 'reshading', device=device)
            gt_pred3 = visualpriors.feature_readout(data, 'curvature', device=device)

            label1 = soft_l1
            label2 = soft_l2
            label3 = soft_l3
            
            loss_value1 = L1Loss(gt_pred1, soft_l1).sum()
            loss_value2 = L1Loss(gt_pred2, soft_l2).sum()
            loss_value3 = L1Loss(gt_pred3, soft_l3).sum()

            list_distort1 += loss_value1.item()
            list_distort2 += loss_value2.item()
            list_distort3 += loss_value3.item()

        print('[SANITY CHECK] Distort-1: %.4e, Distort-2: %.4e, Distort-3: %.4e' % (
                list_distort1 / cnt,
                list_distort2 / cnt,
                list_distort3 / cnt
            )
        )
        '''
        for i in range(4):
            d = gt_pred1[i].cpu().numpy().transpose((1,2,0))
            im = normal_reconstruct(d)
            img = Image.fromarray(im)
            img.save('vis_normal_%d.png' % i)
            
            d = gt_pred2[i].cpu().numpy()[0]
            im = normal_reconstruct(d)
            img = Image.fromarray(im)
            img.save('vis_reshading_%d.png' % i)
            
            d = gt_pred3[i].cpu().numpy().transpose((1,2,0))
            im = curvature_reconstruct(d)
            img = Image.fromarray(im)
            img.save('vis_curvature_%d.png' % i)

            
            d = label1[i].cpu().numpy().transpose((1,2,0))
            im = normal_reconstruct(d)
            img = Image.fromarray(im)
            img.save('gt_normal_%d.png' % i)
            
            d = label2[i].cpu().numpy()[0]
            im = normal_reconstruct(d)
            img = Image.fromarray(im)
            img.save('gt_reshading_%d.png' % i)
            
            d = label3[i].cpu().numpy().transpose((1,2,0))
            im = curvature_reconstruct(d)
            img = Image.fromarray(im)
            img.save('gt_curvature_%d.png' % i)
        '''


    losss = (L1Loss,L1Loss,L1Loss)
    for epoch in range(300):
        start_time = time.time()
        list_distort_values = []
        list_rate = 0.
        list_loss = 0.
        list_distort1 = 0.
        list_distort2 = 0.
        list_distort3 = 0.
        
        list_ref_acc = 0.
        list_ref_d1 = 0.
        list_ref_d2 = 0.
        list_ref_d3 = 0.
        cnt = 0
        list_acc = 0.
        list_b1 = 0.
        list_b2 = 0.
        list_b3 = 0.
        for i,d in enumerate(training_loader):
            opt.zero_grad()
            data = d['data']
            data_256 = d['data_256']
            soft_l1 = d['normal']
            soft_l2 = d['reshading']
            soft_l3 = d['principal_curvature']

            data = data.cuda()
            data_256 = data_256.cuda()
            soft_l1 = soft_l1.cuda()
            soft_l2 = soft_l2.cuda()
            soft_l3 = soft_l3.cuda()

            pred1, pred2, pred3, rate, b1, b3 = net(data, 'train', sanity=False)
            rate = rate.mean()
            b1 = b1.mean()
            b3 = b3.mean()

            label1 = soft_l1
            label2 = soft_l2
            label3 = soft_l3

            loss_value1 = L1Loss(soft_l1, pred1).sum()
            loss_value2 = L1Loss(soft_l2, pred2).sum()
            loss_value3 = L1Loss(soft_l3, pred3).sum()

            w = (5, 1, 8)
            distort_loss = w[0] * loss_value1 + w[1] * loss_value2 + w[2] * loss_value3
            
            list_distort1 += loss_value1
            list_distort2 += loss_value2
            list_distort3 += loss_value3

            train_loss = rate + lmbda * distort_loss
            list_loss += train_loss.item()
            list_rate += rate.item()
            list_b1 += b1.item()
            list_b3 += b3.item()

            train_loss.backward()
            opt.step()
            cnt += 1
        sch.step()
        
        timestamp = time.time()
        print('[Epoch %04d TRAIN %.1f seconds] Loss: %.4e distort1: %.4e, distort2: %.4e, distort3: %.4e, bpp: %.4e, b1: %.4e, b3: %.4e' % (
            epoch,
            timestamp - start_time,
            list_loss / cnt,
            list_distort1 / cnt,
            list_distort2 / cnt,
            list_distort3 / cnt,
            list_rate / cnt,
            list_b1 / cnt,
            list_b3 / cnt
        )
        )

        if True:
            with torch.no_grad():
                list_distort_values = []
                list_rate = 0.
                list_loss = 0.
                list_distort1 = 0.
                list_distort2 = 0.
                list_distort3 = 0.
                list_ref_acc = 0.
                list_ref_d1 = 0.
                list_ref_d2 = 0.
                list_ref_d3 = 0.
                cnt = 0
                list_acc = 0.
                list_b1 = 0.
                list_b2 = 0.
                list_b3 = 0.
                for d in testing_loader:
                    data = d['data']
                    soft_l1 = d['normal']
                    soft_l2 = d['reshading']
                    soft_l3 = d['principal_curvature']

                    data = data.cuda()
                    soft_l1 = soft_l1.cuda()
                    soft_l2 = soft_l2.cuda()
                    soft_l3 = soft_l3.cuda()
                   
                    pred1, pred2, pred3, rate, b1, b3 = net(data, 'test', sanity=False)
                    rate = rate.mean()
                    b1 = b1.mean()
                    b3 = b3.mean()

                    label1 = soft_l1
                    label2 = soft_l2
                    label3 = soft_l3

                    loss_value1 = L1Loss(soft_l1, pred1).sum()
                    loss_value2 = L1Loss(soft_l2, pred2).sum()
                    loss_value3 = L1Loss(soft_l3, pred3).sum()
                    
                    w = (5, 1, 8)
                    distort_loss = w[0] * loss_value1 + w[1] * loss_value2 + w[2] * loss_value3

                    list_distort1 += loss_value1
                    list_distort2 += loss_value2
                    list_distort3 += loss_value3

                    train_loss = rate + lmbda * distort_loss
                    list_loss += train_loss.item()
                    list_rate += rate.item()
                    list_b1 += b1.item()
                    list_b3 += b3.item()

                    cnt += 1
                
                timestamp = time.time()
                
                print('[Epoch %04d Test %.1f seconds] Loss: %.4e distort1: %.4e, distort2: %.4e, distort3: %.4e, bpp: %.4e, b1: %.4e, b3: %.4e' % (
                    epoch,
                    timestamp - start_time,
                    list_loss / cnt,
                    list_distort1 / cnt,
                    list_distort2 / cnt,
                    list_distort3 / cnt,
                    list_rate / cnt,
                    list_b1 / cnt,
                    list_b3 / cnt
                )
                )
                
                if epoch % 10 == 0:
                    print('[INFO] Saving')
                    checkpoint_dir = args.checkpoint_dir
                    if not os.path.isdir(checkpoint_dir):
                        os.mkdir(checkpoint_dir)
                    torch.save(net.state_dict(), './%s/%04d.ckpt' %
                            (checkpoint_dir, epoch))

def evaluate():
    test_label_globs = [
        'normal_test/*.png', 'reshading_test/*.png', 'principal_curvature_test/*.png'
    ]
    loaders = [
        Image.open, Image.open, Image.open, Image.open 
    ]
    
    test_dataset = MultiTaskoDataset(
        data_glob=args.val_data,
        root_path=args.root_path,
        label_globs=test_label_globs,
        loaders=loaders,
        transforms=[
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ), # for rgb
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        TransformReshading()
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        TransformCurvature(),
                    ]
                ),
                tv.transforms.Compose(
                    [
                        tv.transforms.Resize(256),
                        Preprocess(),
                    ]
                ), # for rgb 256
            ]
        )
    testing_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batchsize, shuffle=False, num_workers=8, drop_last=True
    )

    device = torch.device('cuda')
    net = Net(8).cuda()
    
    def cross_entropy(p, q):
        return -1 * p * torch.log(q)

    L1Loss = nn.L1Loss()
    d = torch.load(args.load_weights)
    if 'module' in list(d.keys())[0]:
        nd = {}
        for k in d.keys():
            nd[k.lstrip('module')[1:]] = d[k]
        net.load_state_dict(nd)
    else:
        net.load_state_dict(d)
    fs = sorted(glob.glob(args.val_data))
    fids = [f.split('/')[-1][:-8] for f in fs]
    pool_z2 = []
    pool_like = []
    pool_mu = []
    pool_sigma = []
    pool_label = []
    pool_pred = []
    pool_z1 = []
    pool_like1 = []

    pool_feat = []
    pool_z3_rounded = []
    lmbda = 0.05
    list_distort1 = 0.
    list_distort2 = 0.
    list_distort3 = 0.
    list_distort4 = 0.
    with torch.no_grad():
        list_distort_values = []
        list_rate = 0.
        list_loss = 0.
        list_distort = 0.
        list_acc_ref = 0.
        cnt = 0
        list_acc = 0.
        for d in testing_loader:
            data = d['data']
            soft_l1 = d['normal']
            soft_l2 = d['reshading']
            soft_l3 = d['principal_curvature']
            
            data = data.cuda()
            soft_l1 = soft_l1.cuda()
            soft_l2 = soft_l2.cuda()
            soft_l3 = soft_l3.cuda()
            label1 = soft_l1
            label2 = soft_l2
            label3 = soft_l3

            pred1, pred2, pred3, rate, b1, b2, gt_pred1, gt_pred2, gt_pred3  = net(data, 'test', sanity=True)
            
            loss_value1 = L1Loss(soft_l1, pred1).sum()
            loss_value2 = L1Loss(soft_l2, pred2).sum()
            loss_value3 = L1Loss(soft_l3, pred3).sum()
            
            w = (1, 1, 1)
            distort_loss = w[0] * loss_value1 + w[1] * loss_value2 + w[2] * loss_value3

            list_distort1 += loss_value1
            list_distort2 += loss_value2
            list_distort3 += loss_value3

            train_loss = rate + lmbda * distort_loss
            list_loss += train_loss.item()
            list_rate += rate.item()
            # list_b1 += b1.item()
            # list_b3 += b3.item()

            cnt += 1
        
        timestamp = time.time()
        
        print('[Test] Loss: %.4e Normal: %.4e, Reshading: %.4e, Curvature: %.4e, bpp: %.4e' % (
            list_loss / cnt,
            list_distort1 / cnt,
            list_distort2 / cnt,
            list_distort3 / cnt,
            list_rate / cnt
        )
        )

        
    # num = len(pool_feat)
    # for i in range(num):
    #     np.save('./normal_npy_test/feat_%04d.npy' % (i), pool_feat[i])
    #     np.save('./normal_npy_test/z3rounded_%04d.npy' % (i), pool_z3_rounded[i])
    import IPython
    IPython.embed()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "command", choices=["train", "eval"],
        help="What to do.")
    parser.add_argument(
        "--train_data", default="images/*.png",
        help="Glob pattern identifying training data. This pattern must expand to a list of RGB images in PNG format.")
    parser.add_argument(
        "--train_label", default="images/*.png",
        help="Glob pattern identifying training label. This pattern must expand to a list of RGB images in PNG format.")
    parser.add_argument(
        "--val_data", default="images/*.png",
        help="Glob pattern identifying validation data. This pattern must expand to a list of RGB images in PNG format.")
    parser.add_argument(
        "--val_label", default="images/*.png",
        help="Glob pattern identifying validation data. This pattern must expand to a list of RGB images in PNG format.")
    parser.add_argument(
        "--root_path", default="./",
        help="The root path of the dataset")
    parser.add_argument(
        "--batchsize", type=int, default=16,
        help="Batch size for training.")
    parser.add_argument(
        "--lambda", type=float, default=0.01, dest="lmbda",
        help="Lambda for rate-distortion tradeoff.")
    parser.add_argument(
        "--lambda_fid", type=float, default=0.1, dest="lmbda_fid",
        help="Lambda for rate-distortion tradeoff.") 

    parser.add_argument(
        "--load_weights", default="",
        help="Loaded weights")

    parser.add_argument(
        "--input", default='./test.png',
        help="input image"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-5,
        help="learning rate"
    )

    parser.add_argument(
        "--checkpoint_dir", default="train",
        help="Directory where to save/load model checkpoints.")

        
    parser.add_argument(
        "--load_enc_path", default="",
        help="Load encoder weights path")
        
    parser.add_argument(
        "--load_dec_path", default="",
        help="Load decoder weights path")
    

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == 'eval':
        evaluate()


        
