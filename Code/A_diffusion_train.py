import torch
import numpy as np
import A_train
import argparse

parser = argparse.ArgumentParser(description='ADMI')

parser.add_argument('--device', type=str, default="cuda:0", help='input sequence length')
parser.add_argument('--batch', type=int, default=16, help='input batch size')
parser.add_argument('--dataset', type=str, default="beijing18", help='data set name')
parser.add_argument('--missing_rate', type=float, default=0, help='missing percent for experiment')
parser.add_argument('--seed', type=int, default=3407, help='random seed')

# input data enc_in c_out setting: beijing18:99 urbantraffic:214 physionet12:37
parser.add_argument('--seq_len', type=int, default=48, help='input sequence length')
parser.add_argument('--enc_in', type=int, default=99, help='encoder input size')
parser.add_argument('--c_out', type=int, default=99, help='decoder output size')

# encoder model setting 
parser.add_argument('--d_model', type=int, default=128, help='dimension of model')
parser.add_argument('--e_layers', type=int, default=4, help='num of encoder layers')

# Self-supervised learning setting
parser.add_argument('--mask_ratio_ssl', type=float, default=0.15, help='additional mask rate for ssl')
parser.add_argument('--mask_strategy_ssl', type=str, default="speed", help='mask strategy for ssl [speed, tst, random, randblock]')
parser.add_argument('--avg_mask_len_ssl', type=int, default=3, help='averge continous missing lenth')
parser.add_argument('--epoch_ssl', type=int, default=200, help='training epoch for self-supervised training')
parser.add_argument('--learning_rate_ssl', type=float, default=1e-3, help='learning rate of self-supervised training')

#Diffusion learning setting
parser.add_argument('--encoder', type=str, default='transformer', help='condition encoder model')
parser.add_argument('--diffusion_step_num', type=int, default=50, help='total number of diffusion step')
parser.add_argument('--timeemb', type=int, default=128, help='side information timeemb dimension')
parser.add_argument('--featureemb', type=int, default=16, help='side information dimension')
parser.add_argument('--nheads', type=int, default=8, help='number of head for attention')
parser.add_argument('--channel', type=int, default=128, help='channel dimension of diffusion')
parser.add_argument('--proj_t', type=int, default=128, help='proj_t for feature self-attention')
parser.add_argument('--residual_layers', type=int, default=4, help='number of residual layers in diffusion model')
parser.add_argument('--target_strategy', type=str, default='random', help='mask conditon generation strategy')
parser.add_argument('--schedule', type=str, default='quad', help='beta increase schedule')
parser.add_argument('--beta_start', type=float, default=0.0001)
parser.add_argument('--beta_end', type=float, default=0.2)
parser.add_argument('--epoch_diff', type=int, default=1, help='training epoch for diffusion training')
parser.add_argument('--learning_rate_diff', type=float, default=1e-3, help='learning rate of diffusion training')
parser.add_argument('--pretrain_encoder', type=int, default=1, help='use pretrained encoder')
parser.add_argument('--finetune_encoder', type=int, default=1, help='fine tune pretrained encoder') 

if __name__ == '__main__':
    configs = parser.parse_args()
    print(configs)
    f = open("Z_result.txt","a")
    print(configs, file=f)

    np.random.seed(configs.seed)
    torch.manual_seed(configs.seed)
    torch.cuda.manual_seed(configs.seed)

    if configs.pretrain_encoder:
        pretrain_model = A_train.SSL_train(configs)
    else:
        pretrain_model = None
    model = A_train.diffusion_train(configs, pretrain_model)
    print("TEST")
    A_train.diffusion_test(configs, model)
