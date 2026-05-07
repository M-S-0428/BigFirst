# -*- coding: utf-8 -*-

import torch
from torch import nn
import numpy as np
from torch.nn import functional as F
from sklearn.preprocessing import OneHotEncoder
from sklearn import mixture
from mamba_ssm import MambaLMHeadModel, Mamba
from mamba_ssm import Mamba2


class engine(nn.Module):
    def __init__(self, N_in, N_o, device):
        """
        Initialize network parameters
        """
        super(engine, self).__init__()
        self.N_in = N_in
        self.N_o = N_o
        self.device = device

        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = nn.Sequential(nn.Linear(N_in, 500),  # F_SNP = 2000
                                     nn.ELU(),
                                     nn.Linear(500, 100), )  # 2 * dim(z_SNP) = 100

        # Decoder network, P
        self.decoder = nn.Sequential(nn.Linear(50, 500),  # dim(z_SNP) = 50
                                     nn.ELU(),
                                     nn.Linear(500, 2000))  # F_SNP = 2000

        """Attentive Vector Generation Module"""
        # Generator network, G
        self.generator = nn.Sequential(nn.Linear(56, 100),  # # dim(z) = dim(z_SNP) + dim(c) = 54 Now 56
                                       nn.ELU(),
                                       nn.Linear(100, 180),
                                       nn.Sigmoid())  # 2 * F_MRI = 90*2 = 180

        # Discriminator network, D
        self.discriminator = nn.Sequential(nn.Linear(90, 1),  # F_MRI = 90
                                           nn.Sigmoid())  # real or fake

        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = nn.Sequential(nn.Linear(90, 25),  # dim(Concat(a, x_MRI)) = 90
                                                 nn.ELU())

        self.diagnostician_clf = nn.Sequential(nn.Linear(25, self.N_o))
        self.diagnostician_reg = nn.Sequential(nn.Linear(25, 1))

    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(10, 50).to(self.device)
        return self.decode(eps, apply_sigmoid=True)

    # Represent mu and sigma from the input SNP
    def encode(self, x_SNP):
        mean, logvar = torch.chunk(self.encoder(x_SNP), 2, dim=1)
        return mean, logvar

    # Construct latent distribution
    def reparameterize(self, mean, logvar):
        eps = torch.randn_like(mean).to(self.device)
        return eps * torch.exp(logvar * .5) + mean

    # Reconstruct the input SNP
    def decode(self, z_SNP, apply_sigmoid=False):
        logits = self.decoder(z_SNP)
        if apply_sigmoid:
            probs = F.sigmoid(logits).to(self.device)
            return probs
        return logits

    # Attentive vector and fake neuroimaging generation
    def generate(self, z_SNP, c_demo):
        z = torch.cat((c_demo, z_SNP), dim=-1)
        a, x_MRI_fake = torch.chunk(self.generator(z), 2, dim=-1)
        return x_MRI_fake, a

    # Classify the real and the fake neuroimaging
    def discriminate(self, x_MRI_real_or_fake):
        return self.discriminator(x_MRI_real_or_fake)

    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_MRI, a, apply_logistic_activation=False):
        feature = self.diagnostician_share(
            x_MRI * a)  # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            # y_hat = logit_clf.argmax(dim=1)               
            y_hat = torch.softmax(logit_clf, -1)
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def predict(self, x_MRI, a, apply_logistic_activation=False):
        feature = self.diagnostician_share(torch.mul(x_MRI,
                                                     a))  # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = logit_clf.argmax(dim=1)
            encoder = OneHotEncoder(sparse=False)
            y_hat = encoder.fit_transform(y_hat)  # 0 0 1
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def cluster(self, x_MRI, a, cluster_num_list):
        lowest_bic = np.infty
        bic = []
        feature = self.diagnostician_share(x_MRI * a)  # Hadamard production of the attentive vector
        for cluster_num in cluster_num_list:
            gmm = mixture.GaussianMixture(n_components=cluster_num)
            gmm.fit(feature)
            bic.append(gmm.bic(feature))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        subtype_label = best_gmm.predict(feature)
        return subtype_label


class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim, layer=None):
        super(Encoder, self).__init__()
        if layer is None:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 2048),
                nn.ReLU(),
                nn.Linear(2048, feature_dim),
            )
        else:
            layer.insert(0, input_dim)
            layer.append(feature_dim)
            self.encoder = nn.Sequential()
            for i in range(len(layer) - 1):
                self.encoder.add_module(f'lr_{i}', nn.Linear(layer[i], layer[i + 1]))
                if i != len(layer) - 2:
                    self.encoder.add_module(f'relu_{i}', nn.ReLU())

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim)

        )

    def forward(self, x):
        return self.decoder(x)


class supervised_contrastive(nn.Module):
    def __init__(self, x_size: list, feature_dim, device, snp_num):
        super(supervised_contrastive, self).__init__()
        self.device = device
        self.x_size = x_size
        self.x_view = len(x_size)
        self.encoders_x = []

        for i in range(self.x_view):
            self.encoders_x.append(Encoder(x_size[i], feature_dim).to(device))
        self.encoders_x = nn.ModuleList(self.encoders_x)

        self.u = torch.from_numpy(np.full((snp_num, 1), 1e-3)).float().to(device)
        self.v = torch.from_numpy(np.full((feature_dim * self.x_view, 1), 1e-3)).float().to(device)

    def forward(self, xs):
        zs = []

        for v in range(self.x_view):
            x = xs[v].to(self.device)
            z = self.encoders_x[v](x)
            # z = F.normalize(z, p=2, dim=-1, eps=1e-12)
            zs.append(z)
        zs_cat = torch.cat([zs[i] for i in range(self.x_view)], dim=1)

        return zs, zs_cat


class BAG_MLP(nn.Module):
    def __init__(self, x_size: list, device):
        super(BAG_MLP, self).__init__()
        self.device = device
        self.x_size = x_size
        self.x_view = len(x_size)
        self.encoders_x = []

        for i in range(self.x_view):
            self.encoders_x.append(Encoder(x_size[i], 1).to(self.device))
            # self.MLP.append(nn.Linear(feature_dim, 2).to(self.device))
        self.encoders_x = nn.ModuleList(self.encoders_x)


    def forward(self, xs):
        zs_list = []
        for v in range(self.x_view):
            x = xs[v].to(self.device)
            z = self.encoders_x[v](x)
            # z = F.normalize(z, p=2, dim=-1, eps=1e-12)
            # prob = self.MLP[v](z)
            zs_list.append(z)
            # probs.append(prob)
        # zs = torch.cat(zs_list, dim=1)
        # logits = self.logits_fuse(zs)

        return zs_list

class engine_mv_regcog(nn.Module):
    def __init__(self, SNP_num, N_o, dataSize, z_SNP, CovSize, device, dim):
        """
        Initialize network parameters
        """
        super(engine_mv_regcog, self).__init__()
        self.SNP_num = SNP_num
        self.N_o = N_o
        self.device = device
        self.dataSize = dataSize
        self.z_SNP = z_SNP
        self.CovSize = CovSize
        self.dim = dim

        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = []
        self.decoder = []
        self.generator = []
        self.discriminator = []
        for i in range(len(self.dataSize)):
            self.encoder.append(nn.Sequential(nn.Linear(self.SNP_num, 500),  # F_SNP = 2000
                                              nn.ELU(),
                                              nn.Linear(500, self.z_SNP * 2)).to(self.device))  # 2 * dim(z_SNP) = 100

            # Decoder network, P
            self.decoder.append(nn.Sequential(nn.Linear(self.z_SNP, 500),  # dim(z_SNP) = 50
                                              nn.ELU(),
                                              nn.Linear(500, SNP_num)).to(self.device))  # F_SNP = 2000

            """Attentive Vector Generation Module"""
            # Generator network, G
            self.generator.append(
                nn.Sequential(nn.Linear(self.z_SNP + self.CovSize, 100),  # # dim(z) = dim(z_SNP) + dim(c) = 54 Now 56
                              nn.ELU(),
                              nn.Linear(100, self.dataSize[i] * 2),
                              nn.Sigmoid()).to(self.device))  # 2 * F_MRI = 90*2 = 180

            # Discriminator network, D
            self.discriminator.append(nn.Sequential(nn.Linear(self.dataSize[i], 1),  # F_MRI = 90
                                                    nn.Sigmoid()).to(self.device))  # real or fake

        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = nn.Sequential(nn.Linear(len(dataSize) * self.dim, 25),  # dim(Concat(a, x_MRI)) = 90
                                                 nn.ELU()).to(self.device)

        self.diagnostician_clf = nn.Sequential(nn.Linear(25, self.N_o)).to(self.device)
        self.diagnostician_reg = nn.Sequential(nn.Linear(25, 1))

        # self.diagnostician_reg = nn.Sequential(nn.Linear(25, 1))

    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(10, 50).to(self.device)
        return self.decode(eps, apply_sigmoid=True)

    # Represent mu and sigma from the input SNP
    def encode(self, x_SNP):
        mean_list = []
        logvar_list = []
        for enc in self.encoder:
            mean, logvar = torch.chunk(enc(x_SNP), 2, dim=1)
            mean_list.append(mean)
            logvar_list.append(logvar)
        return mean_list, logvar_list

    # Construct latent distribution
    def reparameterize(self, mean: list, logvar: list):
        result = []
        for i in range(len(mean)):
            eps = torch.randn_like(mean[i]).to(self.device)
            result.append(eps * torch.exp(logvar[i] * .5) + mean[i])
        return result

    # Reconstruct the input SNP
    def decode(self, z_SNP: list, apply_sigmoid=False):
        logits = []
        probs = []
        for i in range(len(z_SNP)):
            logits.append(self.decoder[i](z_SNP[i]))
            if apply_sigmoid:
                probs.append(F.sigmoid(logits[-1]).to(self.device))
        if apply_sigmoid:
            return probs
        return logits

    # Attentive vector and fake neuroimaging generation
    def generate(self, z_SNP: list, c_demo):
        phenotype_list = []
        phenotype_mask = []
        for i in range(len(z_SNP)):
            z = torch.cat((c_demo, z_SNP[i]), dim=-1)
            a, x_phen_fake = torch.chunk(self.generator[i](z), 2, dim=-1)
            phenotype_list.append(x_phen_fake)
            phenotype_mask.append(a)
        return phenotype_list, phenotype_mask

    # Classify the real and the fake neuroimaging
    def discriminate(self, x_MRI_real_or_fake):
        discriminate_list = []
        for i in range(len(x_MRI_real_or_fake)):
            discriminate_list.append(self.discriminator[i](x_MRI_real_or_fake[i]))
        return discriminate_list

    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_phenotype, a, apply_logistic_activation=False):
        x_phenotype_concat = torch.cat(x_phenotype, dim=1)
        a_cat = torch.cat(a, dim=1)
        feature = self.diagnostician_share(x_phenotype_concat * a_cat)  # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            # y_hat = logit_clf.argmax(dim=1)
            y_hat = torch.softmax(logit_clf, -1)
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def regression(self, z, apply_logistic_activation=False):
        feature = self.diagnostician_share(z)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            # y_hat = logit_clf.argmax(dim=1)
            s_hat = F.sigmoid(logit_reg)
            return s_hat
        return logit_reg

    def predict(self, x_phenotype, a, apply_logistic_activation=False):
        x_phenotype_concat = torch.cat(x_phenotype, dim=1)
        a_cat = torch.cat(a, dim=1)
        feature = self.diagnostician_share(torch.mul(x_phenotype_concat, a_cat))
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = logit_clf.argmax(dim=1)
            encoder = OneHotEncoder(sparse=False)
            y_hat = encoder.fit_transform(y_hat)  # 0 0 1
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def cluster(self, x_phenotype, a, cluster_num_list):
        x_phenotype_concat = torch.cat(x_phenotype, dim=1)
        a_cat = torch.cat(a, dim=1)
        lowest_bic = np.infty
        bic = []
        feature = self.diagnostician_share(x_phenotype_concat * a_cat)  # Hadamard production of the attentive vector
        for cluster_num in cluster_num_list:
            gmm = mixture.GaussianMixture(n_components=cluster_num)
            gmm.fit(feature)
            bic.append(gmm.bic(feature))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        subtype_label = best_gmm.predict(feature)
        return subtype_label


class generate_mvphenotype(nn.Module):
    def __init__(self, snp_dim, args, device, ROI_SNP_IDX=None, dataSize=None):
        """
        Initialize network parameters
        """
        super(generate_mvphenotype, self).__init__()
        if dataSize is None:
            dataSize = [116, 116, 116]
        self.snp_dim = snp_dim
        self.ROI_SNP_IDX = ROI_SNP_IDX
        self.device = device
        self.dataSize = dataSize

        """SNP Representation Module"""

        self.generator = []
        self.discriminator = []
        self.gene_mamba = []

        if self.ROI_SNP_IDX is None:
            self.ROI_num = 1
        else:
            self.ROI_num = len(self.ROI_SNP_IDX)

        # project linear layer
        self.lr = nn.ModuleList([nn.Linear(3, 1).to(self.device) for _ in range(len(self.dataSize))])

        # mamba_model = MambaLMHeadModel(config=MambaConfig(), device=device)
        # self.Embedding = nn.Embedding(3, 3).to(self.device)
        self.mamba_model = Mamba(d_model=args.d_model, d_state=args.d_state, dt_rank=args.dt_rank).to(self.device)
        self.maxpool = nn.MaxPool1d(args.d_model)

        # self.gene_mamba.append(mamba_model)
        for i in range(len(self.dataSize)):
            # gene mamba network
            """Attentive Vector Generation Module"""
            # Generator network, G
            self.generator.append(
                nn.Sequential(nn.Linear(self.snp_dim * self.ROI_num, 100),
                              # # dim(z) = dim(z_SNP) + dim(c) = 54 Now 56
                              nn.ELU(),
                              nn.Linear(100, self.dataSize[i] * 2),
                              nn.Sigmoid()).to(self.device))  # 2 * F_MRI = 90*2 = 180

            # Discriminator network, D
            self.discriminator.append(nn.Sequential(nn.Linear(self.dataSize[i], 1),  # F_MRI = 90
                                                    nn.Sigmoid()).to(self.device))  # real or fake

    # Attentive vector and fake neuroimaging generation
    def generate(self, SNP):
        phenotype_list = []
        phenotype_mask = []

        # SNP = self.Embedding(SNP)
        if self.ROI_SNP_IDX is not None:
            zs = []
            for _, value in self.ROI_SNP_IDX.items():
                SNP_ROI = SNP[:, value.squeeze(1) - 1]
                y = self.mamba_model(SNP_ROI)
                y = self.maxpool(y)
                zs.append(torch.squeeze(y, 2)[:, -self.snp_dim:])
            z = torch.cat(zs, 1)
        else:
            y = self.mamba_model(SNP)
            y = self.maxpool(y)
            z = torch.squeeze(y, 2)[:, -self.snp_dim:]

        for i in range(len(self.dataSize)):
            # z = self.mamba_model(input_ids=SNP)
            a, x_phen_fake = torch.chunk(self.generator[i](z), 2, dim=-1)
            phenotype_list.append(x_phen_fake)
            phenotype_mask.append(a)
        return z, phenotype_list, phenotype_mask

    # Classify the real and the fake neuroimaging
    def discriminate(self, x_MRI_real_or_fake):
        discriminate_list = []
        for i in range(len(x_MRI_real_or_fake)):
            discriminate_list.append(self.discriminator[i](x_MRI_real_or_fake[i].to(self.device)))
        return discriminate_list


class engine_mv_clf(nn.Module):
    def __init__(self, SNP_num, N_o, dataSize, z_SNP, CovSize, device):
        """
        Initialize network parameters
        """
        super(engine_mv_clf, self).__init__()
        self.SNP_num = SNP_num
        self.N_o = N_o
        self.device = device
        self.dataSize = dataSize
        self.z_SNP = z_SNP
        self.CovSize = CovSize

        """SNP Representation Module"""
        # Encoder network, Q
        self.encoder = []
        self.decoder = []
        self.generator = []
        self.discriminator = []
        for i in range(len(self.dataSize)):
            self.encoder.append(nn.Sequential(nn.Linear(self.SNP_num, 500),  # F_SNP = 2000
                                              nn.ELU(),
                                              nn.Linear(500, self.z_SNP * 2),
                                              nn.Sigmoid()).to(self.device))  # 2 * dim(z_SNP) = 100

            # Decoder network, P
            self.decoder.append(nn.Sequential(nn.Linear(self.z_SNP, 500),  # dim(z_SNP) = 50
                                              nn.ELU(),
                                              nn.Linear(500, SNP_num)).to(self.device))  # F_SNP = 2000

            """Attentive Vector Generation Module"""
            # Generator network, G
            self.generator.append(
                nn.Sequential(nn.Linear(self.z_SNP, 100),  # # dim(z) = dim(z_SNP) + dim(c) = 54 Now 56
                              nn.ELU(),
                              nn.Linear(100, self.dataSize[i] * 2),
                              nn.Sigmoid()).to(self.device))  # 2 * F_MRI = 90*2 = 180

            # Discriminator network, D
            self.discriminator.append(nn.Sequential(nn.Linear(self.dataSize[i], 1),  # F_MRI = 90
                                                    nn.Sigmoid()).to(self.device))  # real or fake

        """Diagnostician Module"""
        # Diagnostician network, C
        self.diagnostician_share = nn.Sequential(nn.Linear(sum(self.dataSize), 25),  # dim(Concat(a, x_MRI)) = 90
                                                 nn.ELU()).to(self.device)

        self.diagnostician_clf = nn.Sequential(nn.Linear(25, self.N_o)).to(self.device)

        self.diagnostician_reg = nn.Sequential(nn.Linear(25, 1))

    # Reconstructed SNPs sampling
    def sample(self, eps=None):
        if eps is None:
            eps = torch.randn(10, 50).to(self.device)
        return self.decode(eps, apply_sigmoid=True)

    # Represent mu and sigma from the input SNP
    def encode(self, x_SNP):
        mean_list = []
        logvar_list = []
        for enc in self.encoder:
            mean, logvar = torch.chunk(enc(x_SNP), 2, dim=1)
            mean_list.append(mean)
            logvar_list.append(logvar)
        return mean_list, logvar_list

    # Construct latent distribution
    def reparameterize(self, mean: list, logvar: list):
        result = []
        for i in range(len(mean)):
            eps = torch.randn_like(mean[i]).to(self.device)
            result.append(eps * torch.exp(logvar[i] * .5) + mean[i])
        return result

    # Reconstruct the input SNP
    def decode(self, z_SNP: list, apply_sigmoid=False):
        logits = []
        probs = []
        for i in range(len(z_SNP)):
            logits.append(self.decoder[i](z_SNP[i]))
            if apply_sigmoid:
                probs.append(F.sigmoid(logits[-1]).to(self.device))
        if apply_sigmoid:
            return probs
        return logits

    # Attentive vector and fake neuroimaging generation
    def generate(self, z_SNP: list, c_demo):
        phenotype_list = []
        phenotype_mask = []
        for i in range(len(z_SNP)):
            # z = torch.cat((c_demo, z_SNP[i]), dim=-1)
            a, x_phen_fake = torch.chunk(torch.sigmoid(self.generator[i](z_SNP[i])), 2, dim=-1)
            phenotype_list.append(x_phen_fake)
            phenotype_mask.append(a)
        return phenotype_list, phenotype_mask

    # Classify the real and the fake neuroimaging
    def discriminate(self, x_MRI_real_or_fake):
        discriminate_list = []
        for i in range(len(x_MRI_real_or_fake)):
            discriminate_list.append(self.discriminator[i](x_MRI_real_or_fake[i]))
        return discriminate_list

    # Downstream tasks; brain disease diagnosis and cognitive score prediction
    def diagnose(self, x_phenotype, a, apply_logistic_activation=False):
        x_phenotype_concat = torch.cat(x_phenotype, dim=1)
        a_cat = torch.cat(a, dim=1)
        feature = self.diagnostician_share(x_phenotype_concat * a_cat)  # Hadamard production of the attentive vector
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            # y_hat = logit_clf.argmax(dim=1)
            y_hat = torch.softmax(logit_clf, -1)
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def regression(self, z, apply_logistic_activation=False):
        feature = self.diagnostician_share(z)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            # y_hat = logit_clf.argmax(dim=1)
            s_hat = F.sigmoid(logit_reg)
            return s_hat
        return logit_reg

    def predict(self, x_phenotype, a, apply_logistic_activation=False):
        x_phenotype_concat = torch.cat(x_phenotype, dim=1)
        a_cat = torch.cat(a, dim=1)
        feature = self.diagnostician_share(torch.mul(x_phenotype_concat, a_cat))
        logit_clf = self.diagnostician_clf(feature)
        logit_reg = self.diagnostician_reg(feature)
        if apply_logistic_activation:
            y_hat = logit_clf.argmax(dim=1)
            encoder = OneHotEncoder(sparse=False)
            y_hat = encoder.fit_transform(y_hat)  # 0 0 1
            s_hat = F.sigmoid(logit_reg)
            return y_hat, s_hat
        return logit_clf, logit_reg

    def cluster(self, x_phenotype, a, cluster_num_list):
        x_phenotype_concat = torch.cat(x_phenotype, dim=1)
        a_cat = torch.cat(a, dim=1)
        lowest_bic = np.infty
        bic = []
        feature = self.diagnostician_share(x_phenotype_concat * a_cat)  # Hadamard production of the attentive vector
        for cluster_num in cluster_num_list:
            gmm = mixture.GaussianMixture(n_components=cluster_num)
            gmm.fit(feature)
            bic.append(gmm.bic(feature))
            if bic[-1] < lowest_bic:
                lowest_bic = bic[-1]
                best_gmm = gmm

        subtype_label = best_gmm.predict(feature)
        return subtype_label


class mamba_clf(nn.Module):
    def __init__(self, config, num_last_tokens, SNP_num, device):
        super(mamba_clf, self).__init__()
        self.device = device
        self.num_last_tokens = num_last_tokens
        self.SNP_num = SNP_num
        # self.mamba = MambaLMHeadModel(config=config, device=self.device)
        self.Embedding = nn.Embedding(config.vocab_size, config.d_model).to(self.device)

        self.mamba = Mamba(d_model=config.d_model, d_state=config.d_state).to(self.device)
        self.mlplayer = nn.Sequential(
            nn.Linear(self.num_last_tokens, 256),
            nn.Dropout(p=0.9),
            nn.ReLU(),
            nn.Linear(256, 2)
        ).to(self.device)

        self.mlplayer1 = nn.Linear(SNP_num, 1).to(self.device)

        self.maxpool = nn.MaxPool1d(config.d_model)

        # self.mamba = Mamba2(d_model=config.d_model, device=self.device)
        # self.embedding = nn.Embedding(config.vocab_size, config.d_model).to(self.device)
        #
        # self.fuselayer = nn.Sequential(
        #     nn.Linear(self.num_last_tokens, 256),
        #     nn.ReLU(),
        #     nn.Linear(256, 1)
        # ).to(self.device)
        # self.fuselayer = nn.Linear(self.num_last_tokens, 1).to(self.device)

        # self.mlplayer = nn.Sequential(
        #     nn.Linear(self.num_last_tokens, 256),
        #     nn.Dropout(p=0.9),
        #     nn.ReLU(),
        #     nn.Linear(256, 2)
        # ).to(self.device)

    def forward(self, input_ids):
        # y = self.mamba(input_ids, num_last_tokens=self.num_last_tokens).logits
        input_ids = self.Embedding(input_ids)
        y = self.mamba(input_ids)
        y = self.maxpool(y)
        logits = self.mlplayer1(torch.squeeze(y, 2))

        return logits
