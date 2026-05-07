# -*- coding: utf-8 -*-
"""
@Version: 0.1
@Author: Shang
@Date: 2025/5/17
@Description: Bigfirst
@Improvement:
"""

import logging
from utils import *
import loaddata
import engine_model
import torch
from torch import nn
from torch.nn import functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import standard_scale
from torch.nn import MSELoss, BCELoss
from sklearn.model_selection import KFold, StratifiedKFold
import os
from pytorch_grad_cam import GradCAM
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, confusion_matrix
from SCCA import *
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from tabulate import tabulate
import evaluation
import result_analysis
import time
import os
import argparse
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from myDataloader import myDataset
from torch.utils.data import Dataset, DataLoader
from config_mamba import MambaConfig


def sim_mat_construction(data, dist_type='Euc'):
    sample_num = data.shape[0]
    if dist_type == 'Euc':
        if type(data) == np.ndarray:
            sim_mat = np.zeros((sample_num, sample_num))
            for x in range(sample_num):
                sim_mat[:, x] = np.sum(np.power((data[x, :] - data), 2), 1)
        elif type(data) == torch.Tensor:
            sim_mat = torch.zeros((sample_num, sample_num))
            for x in range(sample_num):
                sim_mat[:, x] = torch.sum(torch.pow((data[x, :] - data), 2), 1)
    elif dist_type == 'Cos':
        if type(data) == np.ndarray:
            sim_mat = np.dot(data, data.T)
        elif type(data) == torch.Tensor:
            # data_l2 = data/torch.norm(data, dim=1).unsqueeze(1).repeat(1,data.shape[1])
            sim_mat = torch.matmul(data, data.T)
    return sim_mat


def cluster_method(data, labels=None, class_num=2):
    if labels is None:
        k_means_i = KMeans(init='k-means++', n_clusters=class_num, n_init=10, max_iter=300)
        k_means_i.fit(data.cpu().detach().numpy())
        label_km = k_means_i.labels_

        sc = SpectralClustering(n_clusters=class_num)
        sc.fit(data.cpu().detach().numpy())
        label_sc = sc.labels_

        gmm = GaussianMixture(n_components=class_num)
        label_gmm = gmm.fit_predict(data.cpu().detach().numpy())

        return [label_km, label_sc, label_gmm]
    else:
        k_means_i = KMeans(init='k-means++', n_clusters=class_num, n_init=10, max_iter=300)
        k_means_i.fit(data.cpu().detach().numpy())
        label_km = k_means_i.labels_
        # inertia_km = k_means_i.inertia_
        nmi_km, ari_km, f_km, acc_km = evaluation.evaluate(labels.cpu().detach().numpy(), label_km)

        sc = SpectralClustering(n_clusters=class_num)
        sc.fit(data.cpu().detach().numpy())
        label_sc = sc.labels_
        nmi_sc, ari_sc, f_sc, acc_sc = evaluation.evaluate(labels.cpu().detach().numpy(), label_sc)

        gmm = GaussianMixture(n_components=class_num)
        label_gmm = gmm.fit_predict(data.cpu().detach().numpy())
        nmi_gmm, ari_gmm, f_gmm, acc_gmm = evaluation.evaluate(labels.cpu().detach().numpy(), label_gmm)

        outputInfo = {'methods': ['kMeans', 'Spectral', 'GMM'], 'f1-score': [nmi_km, nmi_sc, nmi_gmm],
                      'recall': [ari_km, ari_sc, ari_gmm], 'precision': [f_km, f_sc, f_gmm],
                      'acc': [acc_km, acc_sc, acc_gmm]}
        # print(tabulate(outputInfo, headers='keys', tablefmt='fancy_grid'))
        out_df = pd.DataFrame(outputInfo)
        logging.info(out_df)
        max_acc = np.max([acc_km, acc_sc, acc_gmm])
        return max_acc, nmi_km, nmi_sc, nmi_gmm, ari_km, ari_sc, ari_gmm, f_km, f_sc, f_gmm, acc_km, acc_sc, acc_gmm


class ConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, denominator_pos=True):
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.denominator_pos = denominator_pos
        self.cca_loss = nn.MSELoss(reduction="sum")

    def forward(self, x_map, labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            x_map: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if x_map.is_cuda
                  else torch.device(
            'cpu')) 

        batch_size = x_map.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  

        # compute logits
        sim_mat = sim_mat_construction(x_map, dist_type='Cos')
        # sim_mat = torch.matmul(x_map, x_map.T)
        anchor_dot_contrast = torch.div(
            sim_mat,
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)  
        logits = (anchor_dot_contrast - logits_max.detach()).to(device)
        # logits = (anchor_dot_contrast).to(device)

        # tile mask
        mask_positive = abs(mask - torch.ones_like(mask))
        # print(mask + mask_positive == torch.ones_like(mask)  )
        mask = mask.repeat(2, 2)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )  

        mask = mask * logits_mask  
        margin = mask * 0  

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        if self.denominator_pos:
            denominator = exp_logits
        else:
            denominator = exp_logits * mask_positive

        all_sum = torch.log(denominator.sum(1, keepdim=True))
        log_prob = (logits * mask - margin) - all_sum

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = log_prob.sum(1) / mask.sum(1) 

        # loss_contrastive
        loss_con = - (self.temperature / self.base_temperature) * mean_log_prob_pos 
        loss = loss_con.mean() / x_map.shape[0]
        return loss


class SigmoidFocalCrossEntropyLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalCrossEntropyLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        sigmoid_p = torch.sigmoid(y_pred)
        ce_loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
        p_t = y_true * sigmoid_p + (1 - y_true) * (1 - sigmoid_p)
        focal_loss = ce_loss * ((1 - p_t) ** self.gamma)
        alpha_weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        loss = alpha_weight * focal_loss
        return loss.mean()


class exp_common:
    def __init__(self, data_HC_AD, ROI_SNP_IDX, train_idx, test_idx, fold_idx, path, args):
        self.AV45 = data_HC_AD['AV45']
        self.FDG = data_HC_AD['FDG']
        self.VBM = data_HC_AD['VBM']
        self.SNP = data_HC_AD['SNP']
        # self.COVS = data_HC_AD['COVS']
        # self.MMSE = data_HC_AD['MMSE']
        self.Y = data_HC_AD['DX']

        self.train_idx = train_idx
        self.test_idx = test_idx
        self.fold_idx = fold_idx
        self.ROI_SNP_IDX = ROI_SNP_IDX
        # self.Y_OH = data_HC_AD['DX']
        # for i in range(np.unique(self.Y).shape[0]):
        #     self.Y_OH[self.Y == np.unique(self.Y)[i]] = i
        # self.Y_OH = np.eye(np.unique(self.Y_OH).shape[0])[self.Y_OH]

        self.supConLoss = ConLoss(temperature=args.temperature, base_temperature=args.temperature)
        self.mse = MSELoss()
        # self.MCI_list = MCI_list
        self.path = path
        self.args = args
        self.pretrain_epoch = self.args.pretrain_epoch
        self.device = self.args.device

        self.dataSize = [self.AV45.shape[1], self.FDG.shape[1], self.VBM.shape[1]]

        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # device configuration
        my_seed = 1
        np.random.seed(my_seed)
        torch.manual_seed(my_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Learning schedules
        self.num_epochs = self.args.num_epochs  # 200
        self.num_batches = len(self.train_idx)
        self.initial_learning_rate = self.args.lr
        self.decay_steps = self.args.decay_steps
        self.decay_rate = self.args.decay_rate

        # Loss control hyperparameter
        self.alpha_rec = self.args.alpha_rec  # reconstruction
        self.alpha_gen = self.args.alpha_gen  # generation
        self.alpha_dis = self.args.alpha_dis  # discrimination
        self.alpha_clf = self.args.alpha_clf  # classification

        # Define learning rate decay function

    def lr_decay_func(self, epoch):
        return self.initial_learning_rate * (self.decay_rate ** (epoch // self.decay_steps))

    def training(self):
        print(f'Start Training, Fold {self.fold_idx}')

        # split training dataset and testing dataset

        X_AV45_train, X_AV45_test = self.AV45[self.train_idx, :], self.AV45[self.test_idx, :]
        X_FDG_train, X_FDG_test = self.FDG[self.train_idx, :], self.FDG[self.test_idx, :]
        X_VBM_train, X_VBM_test = self.VBM[self.train_idx, :], self.VBM[self.test_idx, :]
        E_SNP_train, E_SNP_test = self.SNP[self.train_idx, :], self.SNP[self.test_idx, :]

        Y_train, Y_test = self.Y[self.train_idx], self.Y[self.test_idx]
        # Y_train_OH, Y_test_OH = self.Y_OH[self.train_idx], self.Y_OH[self.test_idx]

        X_AV45_train = torch.FloatTensor(X_AV45_train)
        X_FDG_train = torch.FloatTensor(X_FDG_train)
        X_VBM_train = torch.FloatTensor(X_VBM_train)
        E_SNP_train = torch.LongTensor(E_SNP_train.numpy())
        E_SNP_train = torch.nn.functional.one_hot(E_SNP_train).float()

        Y_train = torch.FloatTensor(Y_train)
        # Y_train_OH = torch.FloatTensor(Y_train_OH).to(self.device)

        X_train = []
        X_train.append(X_AV45_train)
        X_train.append(X_FDG_train)
        X_train.append(X_VBM_train)

        X_AV45_test = torch.FloatTensor(X_AV45_test)
        X_FDG_test = torch.FloatTensor(X_FDG_test)
        X_VBM_test = torch.FloatTensor(X_VBM_test)
        E_SNP_test = torch.LongTensor(E_SNP_test.numpy())
        E_SNP_test = torch.nn.functional.one_hot(E_SNP_test).float()

        Y_test = torch.FloatTensor(Y_test)
        # Y_test_OH = torch.FloatTensor(Y_test_OH).to(self.device)

        X_test = []
        X_test.append(X_AV45_test)
        X_test.append(X_FDG_test)
        X_test.append(X_VBM_test)

        train_data = myDataset(X_AV45_train, X_FDG_train, X_VBM_train, E_SNP_train, Y_train)
        test_data = myDataset(X_AV45_test, X_FDG_test, X_VBM_test, E_SNP_test, Y_test)

        train_dataloader = DataLoader(train_data, batch_size=self.args.batch_size, shuffle=True)
        test_dataloader = DataLoader(test_data, batch_size=32, shuffle=True)

        SNP_num = 1
        model_gen = engine_model.generate_mvphenotype(snp_dim=SNP_num, args=self.args, dataSize=self.dataSize,
                                                      ROI_SNP_IDX=self.ROI_SNP_IDX, device=self.device).to(self.device)
        model_con = engine_model.supervised_contrastive(x_size=self.dataSize, feature_dim=self.args.contrastive_dim
                                                        , device=self.device, snp_num=self.args.snp_num)


        # Apply gradients
        # var = model.parameters()
        theta_G = []
        theta_D = []
        # theta_Q = list()
        theta_Q = list(model_gen.mamba_model.parameters())
        for i in range(len(self.dataSize)):
            # theta_Q = theta_Q + list(model_gen.gene_mamba[i].parameters()) # .state_dict()
            theta_G.append(model_gen.generator[i][0].weight)
            theta_G.append(model_gen.generator[i][0].bias)
            theta_G.append(model_gen.generator[i][2].weight)
            theta_G.append(model_gen.generator[i][2].bias)

            theta_D.append(model_gen.discriminator[i][0].weight)
            theta_D.append(model_gen.discriminator[i][0].bias)

        # Call optimizers
        # opt_mamba = torch.optim.Adam(theta_Q, lr=self.initial_learning_rate)
        opt_gen = torch.optim.Adam(theta_Q + theta_G, lr=self.initial_learning_rate)
        opt_dis = torch.optim.Adam(theta_D, lr=self.initial_learning_rate)
        opt_con = torch.optim.Adam(model_con.parameters(), lr=self.initial_learning_rate)

        logging.basicConfig(
            format='[%(asctime)s] %(message)s',
            level=logging.INFO,
            handlers=[
                logging.FileHandler("{}/log.txt".format(self.path), mode='a', encoding='UTF-8'),
                logging.StreamHandler()
            ]
        )
        logging.info({f'feature dim: {self.args.contrastive_dim}'})

        for epoch in range(self.num_epochs):
            model_gen.train()
            model_con.train()
            loss_con = []
            loss_gen = []
            loss_dis = []
            for X_train, E_SNP_train, Y_train in train_dataloader:

                # Phenotype-Genotype association module
                _, X_train_fake, ab = model_gen.generate(SNP=E_SNP_train.to(self.device))
                real_output = model_gen.discriminate(x_MRI_real_or_fake=X_train)
                fake_output = model_gen.discriminate(x_MRI_real_or_fake=X_train_fake)

                X_train_mask = []
                X_train_mask.append(torch.mul(X_train[0].to(self.device), ab[0]))
                X_train_mask.append(torch.mul(X_train[1].to(self.device), ab[1]))
                X_train_mask.append(torch.mul(X_train[2].to(self.device), ab[2]))

                L_con_all = []
                zs, zs_cat = model_con(X_train_mask)
                for i in range(len(X_train_mask)):
                    for j in range(i + 1, len(X_train_mask)):
                        L_con_all.append(self.supConLoss(torch.cat((zs[i], zs[j]), 0), Y_train))
                L_con = sum(L_con_all)

                # Least-Square GAN loss
                L_gen = 0
                L_dis = 0
                for i in range(len(fake_output)):
                    L_gen += F.mse_loss(fake_output[i], torch.ones_like(fake_output[i]))
                    L_dis += F.mse_loss(torch.ones_like(real_output[i]), real_output[i]) \
                            + F.mse_loss(torch.zeros_like(fake_output[i]), fake_output[i])
                L_gen *= self.alpha_gen
                L_dis *= self.alpha_dis

                # loss = L_con + L_gen + L_dis
                loss_con.append(L_con)
                loss_gen.append(L_gen)
                loss_dis.append(L_dis)
                # zero gradients
                # opt_mamba.zero_grad()
                opt_gen.zero_grad()
                opt_dis.zero_grad()
                opt_con.zero_grad()

                # loss backward
                L_gen.backward(retain_graph=True)
                L_dis.backward(retain_graph=True)
                L_con.backward()

                # L_reg.backward()

                # step
                opt_con.step()
                # opt_mamba.step()
                opt_gen.step()
                opt_dis.step()

                # if self.args.Neuron_regularization:
                #     for i in range(len(model_con.encoders_x)):
                #         for j in [0, 2, 4, 6, 8]:
                #             model_con.encoders_x[i].encoder[j].weight.data = Neuron_Regularizer(
                #                 model_con.encoders_x[i].encoder[j].weight.data)

            # Loss report
            logging.info(
                f'Epoch: {epoch + 1}, Lgen: {sum(loss_gen) / len(loss_gen):>.4f}, Ldis: {sum(loss_dis) / len(loss_dis):>.4f}, Lcon: {sum(loss_con) / len(loss_con):>.4f}')


            # Results
            model_gen.eval()
            model_con.eval()
            with torch.no_grad():
                logging.info(f'Start Testing, Fold {self.fold_idx}')

                _, _, A_test = model_gen.generate(E_SNP_test.to(self.device))

                # testing the multimodal phenotype data fusion representation
                X_test_mask = []
                X_test_mask.append(torch.mul(X_AV45_test.to(self.device), A_test[0]))
                X_test_mask.append(torch.mul(X_FDG_test.to(self.device), A_test[1]))
                X_test_mask.append(torch.mul(X_VBM_test.to(self.device), A_test[2]))
                _, zs_val = model_con(X_test_mask)
                max_acc, nmi_km, nmi_sc, nmi_gmm, ari_km, ari_sc, ari_gmm, f_km, f_sc, f_gmm, acc_km, acc_sc, acc_gmm = cluster_method(
                    zs_val, Y_test)

        return (
            model_con, model_gen, nmi_km, nmi_sc, nmi_gmm, ari_km, ari_sc, ari_gmm, f_km, f_sc, f_gmm
            , acc_km, acc_sc, acc_gmm)

    def get_device(self, memory_rate, my_seed):
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(my_seed)
            torch.cuda.set_device(0)
            torch.cuda.empty_cache()
            total_memory = torch.cuda.get_device_properties(0).total_memory
            torch.empty(int(total_memory * memory_rate), dtype=torch.int8, device='cuda')
            return 'cuda'
        else:
            return 'cpu'


def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0001, help='learning rate', type=float)
    parser.add_argument("--num_epochs", default=200, help='total number of epochs', type=int)
    parser.add_argument("--temperature", default=0.07, help='temperature', type=float)
    parser.add_argument("--decay_rate", default=0.96, help='decay rate', type=float)
    parser.add_argument("--decay_steps", default=1000, help='decay steps', type=int)
    parser.add_argument("--alpha_rec", default=.7, help='reconstruction', type=float)
    parser.add_argument("--alpha_gen", default=.5, help='generation', type=float)
    parser.add_argument("--alpha_dis", default=1, help='discrimination', type=float)
    parser.add_argument("--alpha_clf", default=1, help='classification', type=float)
    parser.add_argument("--batch_size", default=64, help='batch_size', type=int)
    parser.add_argument("--contrastive_dim", default=128, help='latent contrastive dimension', type=int)
    parser.add_argument("--layer", default=None, help='encoder layer')
    parser.add_argument("--snp_num", default=50, help='left SNP numbers', type=int)
    parser.add_argument("--latent_snp_dim", default=50, help='SNP dimension of sampling', type=int)
    parser.add_argument("--l1_param", default=0.003, help='l1 regularization parameter', type=float)
    parser.add_argument("--pretrain_epoch", default=95, help='pretrain epoch', type=int)
    parser.add_argument("--device", default=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                        , help='device')
    parser.add_argument("--info", default='No information')
    parser.add_argument("--dataset", default='ADNI2')
    parser.add_argument("--ROI", default=False)
    parser.add_argument("--d_model", default=3, help='input_data.shape[-1]', type=int)
    parser.add_argument("--d_state", default=1, help='N', type=int)
    # parser.add_argument("--d_intermediate", default=0, type=int)
    # parser.add_argument("--n_layer", default=1, type=int)
    parser.add_argument("--dt_rank", default=1, help='dimension of delta before broadcast', type=int)
    parser.add_argument("--Neuron_regularization", default=False)

    return parser


def main():
    args = add_args().parse_args()

    if args.dataset == 'ADNI2':
        if not args.ROI:
            data_HC_AD, data_MCI_all, data_EMCI, data_LMCI, data_SMC, feature_ID, ROI_SNP_IDX = loaddata.load_common_data(
                scale=True)
        else:
            data_HC_AD, data_MCI_all, data_EMCI, data_LMCI, data_SMC, feature_ID, ROI_SNP_IDX = loaddata.load_common_data_by_brainregion(
                scale=True)
    else:
        data_HC_AD, ROI_SNP_IDX = loaddata.load_ADNI1_data()

    skfolds = StratifiedKFold(n_splits=5)
    # batch_size_list = [16, 32, 64]
    N_list = [1, 2, 4]
    layer_list = [[128, 256], [128, 256], [128, 256, 512]]
    contrastive_dim_list = [32, 64, 128]
    # for bs in batch_size_list:
    # for N in N_list:
    #     for layer in layer_list:
    #         for contrastive_dim in contrastive_dim_list:

    args.d_state, args.layer, args.contrastive_dim = 2, [128, 256], 64

    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    path = '/home/shang/.virtualenvs/mamba/results/' + time_now
    if not os.path.exists(path):
        os.makedirs(path)
        print('fold create complete ' + path)

    logging.basicConfig(
        format='[%(asctime)s] %(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler("{}/log.txt".format(path), mode='a', encoding='UTF-8'),
            logging.StreamHandler()
        ]
    )
    logging.info(f'training dataset is: {args.dataset}')
    logging.info(f'whether train SNP by ROI is: {args.ROI}')
    logging.info(f'batch_size: {args.batch_size}')
    logging.info(f'layer: {args.layer}')
    logging.info(f'the number of N is: {args.d_state}')
    logging.info(f'the contrastive_dim is: {args.contrastive_dim}')

    nmi_km_list = []
    nmi_sc_list = []
    nmi_gmm_list = []
    ari_km_list = []
    ari_sc_list = []
    ari_gmm_list = []
    f_km_list = []
    f_sc_list = []
    f_gmm_list = []
    acc_km_list = []
    acc_sc_list = []
    acc_gmm_list = []

    for fold_idx, (train_index, test_index) in enumerate(skfolds.split(data_HC_AD['AV45'], data_HC_AD['DX'])):
        logging.info(f'Start Testing, Fold {fold_idx}')
        exp = exp_common(data_HC_AD, ROI_SNP_IDX, train_index, test_index, fold_idx, path, args)
        model_con, model_gen, nmi_km, nmi_sc, nmi_gmm, ari_km, ari_sc, ari_gmm, f_km, f_sc, f_gmm, acc_km, acc_sc, acc_gmm = exp.training()

        torch.save(model_con.state_dict(), os.path.join(path, f'model_contrast_{fold_idx}.pt'))
        torch.save(model_gen.state_dict(), os.path.join(path, f'model_engine_{fold_idx}.pt'))
        nmi_km_list.append(nmi_km)
        nmi_sc_list.append(nmi_sc)
        nmi_gmm_list.append(nmi_gmm)
        ari_km_list.append(ari_km)
        ari_sc_list.append(ari_sc)
        ari_gmm_list.append(ari_gmm)
        f_km_list.append(f_km)
        f_sc_list.append(f_sc)
        f_gmm_list.append(f_gmm)
        acc_km_list.append(acc_km)
        acc_sc_list.append(acc_sc)
        acc_gmm_list.append(acc_gmm)

    nmi_km_m, nmi_km_s = np.mean(nmi_km_list), np.std(nmi_km_list)
    nmi_sc_m, nmi_sc_s = np.mean(nmi_sc_list), np.std(nmi_sc_list)
    nmi_gmm_m, nmi_gmm_s = np.mean(nmi_gmm_list), np.std(nmi_gmm_list)
    ari_km_m, ari_km_s = np.mean(ari_km_list), np.std(ari_km_list)
    ari_sc_m, ari_sc_s = np.mean(ari_sc_list), np.std(ari_sc_list)
    ari_gmm_m, ari_gmm_s = np.mean(ari_gmm_list), np.std(ari_gmm_list)
    f_km_m, f_km_s = np.mean(f_km_list), np.std(f_km_list)
    f_sc_m, f_sc_s = np.mean(f_sc_list), np.std(f_sc_list)
    f_gmm_m, f_gmm_s = np.mean(f_gmm_list), np.std(f_gmm_list)
    acc_km_m, acc_km_s = np.mean(acc_km_list), np.std(acc_km_list)
    acc_sc_m, acc_sc_s = np.mean(acc_sc_list), np.std(acc_sc_list)
    acc_gmm_m, acc_gmm_s = np.mean(acc_gmm_list), np.std(acc_gmm_list)

    outputInfo = {'methods': ['kMeans', 'Spectral', 'GMM'],
                  'f1-score': [f'{nmi_km_m}±{nmi_km_s}', f'{nmi_sc_m}±{nmi_sc_s}', f'{nmi_gmm_m}±{nmi_gmm_s}'],
                  'recall': [f'{ari_km_m}±{ari_km_s}', f'{ari_sc_m}±{ari_sc_s}', f'{ari_gmm_m}±{ari_gmm_s}'],
                  'precision': [f'{f_km_m}±{f_km_s}', f'{f_sc_m}±{f_sc_s}', f'{f_gmm_m}±{f_gmm_s}'],
                  'acc': [f'{acc_km_m}±{acc_km_s}', f'{acc_sc_m}±{acc_sc_s}', f'{acc_gmm_m}±{acc_gmm_s}']}
    # print(tabulate(outputInfo, headers='keys', tablefmt='fancy_grid'))
    out_df = pd.DataFrame(outputInfo)
    logging.info(out_df)


if __name__ == '__main__':
    main()
