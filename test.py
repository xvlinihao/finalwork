from __future__ import print_function
import os
import gc
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from model import DCP
from util import transform_point_cloud,npmat2euler
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import plotly.graph_objects as go
from data_input import RegisterSet

def register(net,register_loader,args):
    net.eval()
    new_src=[]
    new_target=[]
    #s=[]
    #t=[]
    name=args.src.split('/')[1]
    name=name.split('.')[0]



    for src,target in tqdm(register_loader):
        #s.append(src)
        #t.append(target)
        src=src.cuda()
        target=target.cuda()
        print('src.shape',src.shape,'target.shape',target.shape)
        

        rotation_ab_pred, translation_ab_pred, rotation_ba_pred, translation_ba_pred = net(src, target)

        rotation_ab_pred.detach().cpu().numpy()
        translation_ab_pred.detach().cpu().numpy()
        rotation_ba_pred.detach().cpu().numpy()
        translation_ba_pred.detach().cpu().numpy()

        transformed_src = transform_point_cloud(src, rotation_ab_pred, translation_ab_pred)
        new_src.append(transformed_src.detach().cpu().numpy())
        transformed_target = transform_point_cloud(target, rotation_ba_pred, translation_ba_pred)
        new_target.append(transformed_target.detach().cpu().numpy())
    new_src=np.concatenate(new_src,axis=0)
    new_target=np.concatenate(new_target,axis=0)
    
    #s=np.concatenate(s,axis=1).T
    #t=np.concatenate(t,axis=1).T
    #new_src=np.concatenate(new_src,axis=1).T
    new_target=np.concatenate(new_target,axis=1).T

    #np.save('new_src.npy',new_src)
    np.save('new_{}.npy'.format(name),new_target)
    #np.save('src.npy',s)
    #np.save('target.npy',t)

def main():
    parser = argparse.ArgumentParser(description='Point Cloud Registration')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dcp', metavar='N',
                        choices=['dcp'],
                        help='Model to use, [dcp]')
    parser.add_argument('--emb_nn', type=str, default='pointnet', metavar='N',
                        choices=['pointnet', 'dgcnn'],
                        help='Embedding nn to use, [pointnet, dgcnn]')
    parser.add_argument('--pointer', type=str, default='transformer', metavar='N',
                        choices=['identity', 'transformer'],
                        help='Attention-based pointer generator to use, [identity, transformer]')
    parser.add_argument('--head', type=str, default='svd', metavar='N',
                        choices=['mlp', 'svd', ],
                        help='Head to use, [mlp, svd]')
    parser.add_argument('--emb_dims', type=int, default=512, metavar='N',
                        help='Dimension of embeddings')
    parser.add_argument('--n_blocks', type=int, default=1, metavar='N',
                        help='Num of blocks of encoder&decoder')
    parser.add_argument('--n_heads', type=int, default=4, metavar='N',
                        help='Num of heads in multiheadedattention')
    parser.add_argument('--ff_dims', type=int, default=1024, metavar='N',
                        help='Num of dimensions of fc in transformer')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='N',
                        help='Dropout ratio in transformer')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=10, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action='store_true', default=False,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1234, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='evaluate the model')
    parser.add_argument('--cycle', type=bool, default=False, metavar='N',
                        help='Whether to use cycle consistency')
    parser.add_argument('--gaussian_noise', type=bool, default=False, metavar='N',
                        help='Wheter to add gaussian noise')
    parser.add_argument('--unseen', type=bool, default=False, metavar='N',
                        help='Wheter to test on unseen category')
    parser.add_argument('--num_points', type=int, default=1024, metavar='N',
                        help='Num of points to use')
    parser.add_argument('--dataset', type=str, default='modelnet40', choices=['modelnet40'], metavar='N',
                        help='dataset to use')
    parser.add_argument('--factor', type=float, default=4, metavar='N',
                        help='Divided factor for rotations')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--src',type=str,default='',metavar='N',
                        help='source ply file')
    parser.add_argument('--target',type=str,default='',metavar='N',
                        help='target ply file')
    args = parser.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False 
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.model == 'dcp':
        net = DCP(args).cuda()
        net.load_state_dict(torch.load(args.model_path), strict=False)
        if args.eval:
            register_loader = DataLoader(
            RegisterSet(args.src,args.target),batch_size=args.batch_size,shuffle=False,drop_last=False)
            register(net,register_loader,args)
        else:
            return

if __name__ == "__main__":
    main()