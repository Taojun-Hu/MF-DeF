# -*- coding: utf-8 -*-
import scipy.sparse as sps
import numpy as np
import torch
torch.manual_seed(2020)
from torch import nn
import torch.nn.functional as F
from math import sqrt
import pdb
import time

from utils import ndcg_func,  recall_func, precision_func, generate_total_sample
acc_func = lambda x,y: np.sum(x == y) / len(x)
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from collections import defaultdict

mse_func = lambda x,y: np.mean((x-y)**2)
acc_func = lambda x,y: np.sum(x == y) / len(x)

from baselines import MF_BaseModel, Embedding_Sharing

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLP without DF
class MLP(nn.Module):
    def __init__(self, input_size, batch_size, *args, **kwargs):
        super(MLP, self).__init__()
        # self.linear_1 = torch.nn.Linear(input_size, hidden_size)
        # self.linear_2 = torch.nn.Linear(hidden_size, 1)
        self.linear_1 = torch.nn.Linear(input_size, 1, bias=False)
        self.num_feature = input_size
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        x = torch.Tensor(x).to(device)
        out = self.sigmoid(self.linear_1(x))
        # out = self.sigmoid(self.linear_2(out))
        return torch.squeeze(out)
           
    def fit(self, x, y, feature_train,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(device)

                feature = feature_train[selected_idx].to(device)

                pred = self.forward(feature, True)

                xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MLP] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.forward(x)
        return pred.detach().cpu().numpy()

# MLP_exp: MLP with delay feedback (but not considering the MLP without discount)
class MLP_exp(nn.Module):
    def __init__(self, input_size, batch_size, *args, **kwargs):
        super().__init__()
        self.linear_1 = torch.nn.Linear(input_size, 1, bias=False)
        self.linear_2 = torch.nn.Linear(input_size, 1, bias=False)
        self.num_feature = input_size
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        x = torch.Tensor(x).to(device)
        out1 = self.sigmoid(self.linear_1(x))
        out2 = torch.exp(self.linear_2(x))
        out2 = torch.clamp(out2, 1e-5, 3)

        # out2 = self.relu(self.linear_2(x)) + 1e-2

        return torch.squeeze(out1), torch.squeeze(out2)
           
    def fit(self, x, y, e, d, feature_train,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(device)

                feature = feature_train[selected_idx].to(device)

                sub_e = e[selected_idx]
                sub_d = d[selected_idx]
                sub_e = torch.Tensor(sub_e).to(device)
                sub_d = torch.Tensor(sub_d).to(device)
                
                out1, out2 = self.forward(feature, True)

                xent_loss = torch.nanmean( -(sub_y*(torch.log( out2*out1 ) -out2*sub_d ) + (1-sub_y)*torch.log( 1 - out1 + out1*torch.exp(-out2*sub_e) )) )
                # xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP_exp] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP_exp] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MLP_exp] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred, _ = self.forward(x)
        return pred.detach().cpu().numpy()

# MLP_weibull
class MLP_weibull(nn.Module):
    def __init__(self, input_size, batch_size, *args, **kwargs):
        super(MLP_weibull, self).__init__()
        self.linear_1 = torch.nn.Linear(input_size, 1, bias=True)
        self.linear_scale = torch.nn.Linear(input_size, 1, bias=False)
        self.linear_conce = torch.nn.Linear(input_size, 1, bias=False)
        self.num_feature = input_size
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        x = torch.Tensor(x).to(device)
        out1 = self.sigmoid(self.linear_1(x))

        pscale, pconce = torch.squeeze(torch.exp(self.linear_scale(x))), torch.squeeze(torch.exp(self.linear_conce(x)))

        pscale = torch.clamp(pscale, 1e-5, 3)
        pconce = torch.clamp(pscale, 1e-5, 3)

        # pscale, pconce = torch.squeeze(self.relu(self.linear_scale(x)) + 1e-4  ), torch.squeeze(self.relu(self.linear_conce(x)) + 1e-4 )

        return torch.squeeze(out1), pscale, pconce
           
    def fit(self, x, y, e, d, feature_train,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(device)

                # feature = np.concatenate((feature_user[sub_x[:, 0]], feature_item[sub_x[:, 1]]), axis =1)

                feature = feature_train[selected_idx].to(device)

                sub_e = e[selected_idx]
                sub_d = d[selected_idx]
                sub_e = torch.Tensor(sub_e).to(device)
                sub_d = torch.Tensor(sub_d).to(device)
                
                out1, pscale, pconce = self.forward(feature, True)

                weibull_def = torch.distributions.weibull.Weibull(pscale, pconce)
                density_def = weibull_def.log_prob(sub_d)
                survive_def = 1 - weibull_def.cdf(sub_e)

                xent_loss = torch.nanmean( -(sub_y*(torch.log(out1 ) + density_def ) + (1-sub_y)*torch.log( 1 - out1 + out1*survive_def )) )
                # xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP_weibull] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP_weibull] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MLP_weibull] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred, _, _ = self.forward(x)
        return pred.detach().cpu().numpy()

# MLP_gamma
class MLP_gamma(nn.Module):
    def __init__(self, input_size, batch_size, *args, **kwargs):
        super(MLP_gamma, self).__init__()
        self.linear_1 = torch.nn.Linear(input_size, 1, bias=False)
        self.linear_scale = torch.nn.Linear(input_size, 1, bias=False)
        self.linear_conce = torch.nn.Linear(input_size, 1, bias=False)
        self.num_feature = input_size
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        x = torch.Tensor(x).to(device)
        out1 = self.sigmoid(self.linear_1(x))

        # pscale, pconce = torch.squeeze(torch.exp(self.linear_scale(x))), torch.squeeze(torch.exp(self.linear_conce(x)))
        pscale, pconce = torch.squeeze(torch.exp(self.linear_scale(x))), torch.squeeze(torch.exp(self.linear_conce(x)))

        pscale = torch.clamp(pscale, 1e-5, 3)
        pconce = torch.clamp(pscale, 1e-5, 3)

        # pscale, pconce = torch.squeeze(self.relu(self.linear_scale(x)) + 1e-2  ), torch.squeeze(self.relu(self.linear_conce(x)) + 1e-2 )

        return torch.squeeze(out1), pscale, pconce
           
    def fit(self, x, y, e, d, feature_train,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(device)

                feature = feature_train[selected_idx].to(device)

                sub_e = e[selected_idx]
                sub_d = d[selected_idx]
                sub_e = torch.Tensor(sub_e).to(device)
                sub_d = torch.Tensor(sub_d).to(device)
                
                out1, pscale, pconce = self.forward(feature, True)

                gamma_def = torch.distributions.gamma.Gamma(pscale, pconce)
                density_def = gamma_def.log_prob(sub_d)
                survive_def = 1 - gamma_def.cdf(sub_e)

                xent_loss = torch.nanmean( -(sub_y*(torch.log(out1 ) + density_def ) + (1-sub_y)*torch.log( 1 - out1 + out1*survive_def )) )
                # xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP_gamma] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP_gamma] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MLP_gamma] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred, _, _ = self.forward(x)
        return pred.detach().cpu().numpy()


# MLP_lognormal
class MLP_lognormal(nn.Module):
    def __init__(self, input_size, batch_size, *args, **kwargs):
        super(MLP_lognormal, self).__init__()
        self.linear_1 = torch.nn.Linear(input_size, 1, bias=False)
        self.linear_scale = torch.nn.Linear(input_size, 1, bias=False)
        self.linear_conce = torch.nn.Linear(input_size, 1, bias=False)
        self.num_feature = input_size
        self.batch_size = batch_size
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.xent_func = torch.nn.BCELoss()

    def forward(self, x, is_training=False):
        x = torch.Tensor(x).to(device)
        out1 = self.sigmoid(self.linear_1(x))

        pmu, psigma = torch.squeeze(self.linear_scale(x)), torch.squeeze(torch.exp(torch.exp(self.linear_conce(x) )))

        psigma = torch.clamp(psigma, 1e-5, 3)

        # pmu, psigma = torch.squeeze(self.linear_scale(x)), torch.squeeze( self.relu(self.linear_conce(x)) + 1e-3 )

        return torch.squeeze(out1), pmu, psigma
           
    def fit(self, x, y, e, d, feature_train,
        num_epoch=1000, lr=0.05, lamb=0, 
        tol=1e-4, verbose=False):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // self.batch_size

        early_stop = 0
        
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[self.batch_size*idx:(idx+1)*self.batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]
                sub_y = torch.Tensor(sub_y).to(device)

                feature = feature_train[selected_idx].to(device)

                sub_e = e[selected_idx]
                sub_d = d[selected_idx]
                sub_e = torch.Tensor(sub_e).to(device)
                sub_d = torch.Tensor(sub_d).to(device)
                
                out1, pmu, psigma = self.forward(feature, True)

                lognormal_def = torch.distributions.log_normal.LogNormal(pmu, psigma)
                density_def = lognormal_def.log_prob(sub_d)
                survive_def = 1 - lognormal_def.cdf(sub_e)

                xent_loss = torch.nanmean( -(sub_y*(torch.log(out1 ) + density_def ) + (1-sub_y)*torch.log( 1 - out1 + out1*survive_def )) )
                # xent_loss = self.xent_func(pred,sub_y)

                optimizer.zero_grad()
                xent_loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MLP_lognormal] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MLP_lognormal] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MLP_lognormal] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred, _, _ = self.forward(x)
        return pred.detach().cpu().numpy()


class MF_IPS_DF(nn.Module):
    def __init__(self, num_users, num_items, num_feature, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.num_feature = num_feature
        self.W = torch.nn.Embedding(self.num_users, self.embedding_k)
        self.H = torch.nn.Embedding(self.num_items, self.embedding_k)
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()
        self.delaymodel = torch.nn.Linear(num_feature, 1, bias=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x, feature, is_training=False):
        user_idx = torch.LongTensor(x[:,0]).to(device)
        item_idx = torch.LongTensor(x[:,1]).to(device)
        U_emb = self.W(user_idx)
        V_emb = self.H(item_idx)

        out = torch.sum(U_emb.mul(V_emb), 1)

        out2 = torch.exp(self.delaymodel(feature))
        out2 = torch.clamp(out2, 1e-5, 3)

        # out2 = self.relu(self.delaymodel(feature)) + 1e-2

        if is_training:
            return torch.squeeze(out), torch.squeeze(out2), U_emb, V_emb
        else:
            return torch.squeeze(out), torch.squeeze(out2)

    def fit(self, x, y, e, d, feature_train, y_ips=None,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, lamb1=0, 
        tol=1e-4, verbose = False):

        optimizer = torch.optim.Adam([{'params': self.W.parameters(), 'lr':lr, 'weight_decay':lamb }, {'params': self.H.parameters(), 'lr':lr, 'weight_decay':lamb }, {'params': self.delaymodel.parameters(), 'lr':lr, 'weight_decay':lamb1 }], lr=lr, weight_decay=lamb)
        last_loss = 1e9

        num_sample = len(x)
        total_batch = num_sample // batch_size

        early_stop = 0
        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample)
            np.random.shuffle(all_idx)
            epoch_loss = 0

            for idx in range(total_batch):
                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].to(device)
                sub_y = torch.Tensor(sub_y).to(device)

                feature = feature_train[selected_idx].to(device)

                sub_e = e[selected_idx]
                sub_d = d[selected_idx]
                sub_e = torch.Tensor(sub_e).to(device)
                sub_d = torch.Tensor(sub_d).to(device)

                prob1, lamb1, u_emb, v_emb = self.forward(sub_x, feature, True)
                prob1 = self.sigmoid(prob1)

                xent_loss = torch.nanmean(-inv_prop*(sub_y*(torch.log( lamb1*prob1 )-lamb1*sub_d) + (1-sub_y)*torch.log( torch.exp(-lamb1*sub_e)*prob1 + 1 - prob1 )))

                # print(x)
                
                # xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop)

                loss = xent_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                epoch_loss += xent_loss.detach().cpu().numpy()

            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-IPS-DF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-IPS-DF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-IPS-DF] Reach preset epochs, it seems does not converge.")

    def predict(self, x, feature):
        feature = torch.Tensor(feature).to(device)
        pred, _ = self.forward(x, feature, False)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity
            
        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl


class MF_DR_JL_DF(nn.Module):
    def __init__(self, num_users, num_items, num_feature, embedding_k=4, *args, **kwargs):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_k = embedding_k
        self.num_feature = num_feature
        self.prediction_model = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.imputation = MF_BaseModel(
            num_users=self.num_users, num_items=self.num_items, embedding_k=self.embedding_k)
        self.delaymodel = torch.nn.Linear(num_feature, 1, bias = True)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.xent_func = torch.nn.BCELoss()

    # with x, y, e, d, feature_user, feature_item
    def fit(self, x, y, e, d, feature_train, feature_user, feature_item, y_ips=None,
        num_epoch=1000, batch_size=128, lr=0.05, lamb=0, lambv=0, 
        tol=1e-4, G=1, verbose = False): 

        optimizer_both = torch.optim.Adam([{'params': self.prediction_model.parameters(), 'lr':lr, 'weight_decay':lamb }, {'params': self.delaymodel.parameters(), 'lr':lr, 'weight_decay':lambv }], lr=lr, weight_decay=lamb)

        # optimizer_prediction = torch.optim.Adam(
        #     self.prediction_model.parameters(), lr=lr, weight_decay=lamb)
        optimizer_imputation = torch.optim.Adam(
            self.imputation.parameters(), lr=lr, weight_decay=lamb)
        # optimizer_delaymodel = torch.optim.Adam(self.delaymodel.parameters(), lr = lr, weight_decay=lamb)
        
        last_loss = 1e9
            
        # generate all counterfactuals and factuals
        x_all = generate_total_sample(self.num_users, self.num_items)

        num_sample = len(x) #6960 
        total_batch = num_sample // batch_size

        if y_ips is None:
            one_over_zl = self._compute_IPS(x, y)
        else:
            one_over_zl = self._compute_IPS(x, y, y_ips)

        early_stop = 0
        for epoch in range(num_epoch):
            all_idx = np.arange(num_sample) # observation
            np.random.shuffle(all_idx)

            # sampling counterfactuals
            ul_idxs = np.arange(x_all.shape[0]) # all
            np.random.shuffle(ul_idxs)

            epoch_loss = 0

            for idx in range(total_batch):

                # mini-batch training
                selected_idx = all_idx[batch_size*idx:(idx+1)*batch_size]
                sub_x = x[selected_idx]
                sub_y = y[selected_idx]

                # propensity score
                inv_prop = one_over_zl[selected_idx].to(device)       

                sub_y = torch.Tensor(sub_y).to(device)

                feature = feature_train[selected_idx].to(device)

                sub_e = e[selected_idx]
                sub_d = d[selected_idx]
                sub_e = torch.Tensor(sub_e).to(device)
                sub_d = torch.Tensor(sub_d).to(device)

                # lamb1 = torch.exp(self.delaymodel(feature))
                # lamb1 = torch.clamp(lamb1, 1e-5, 3)

                lamb1 = self.relu(self.delaymodel(feature))
                lamb1 = torch.clamp(lamb1, 1e-5, 3)
                        
                pred = self.prediction_model.forward(sub_x)
                pred = torch.clamp(pred, 1e-10, 1-1e-10)

                imputation_y = self.imputation.predict(sub_x).to(device)
                # imputation_y = torch.clamp(imputation_y, 1e-10, 1-1e-10)
                
                x_sampled = x_all[ul_idxs[G*idx* batch_size : G*(idx+1)*batch_size]]

                feature_sampled = torch.cat([ feature_user[x_sampled[:, 0]], feature_item[x_sampled[:, 1]] ], axis = 1 ).to(device)
                lamb2 = torch.exp(self.delaymodel(feature_sampled))
                lamb2 = torch.clamp(lamb2, 1e-5, 3)
                                       
                pred_u = self.prediction_model.forward(x_sampled) 
                imputation_y1 = self.imputation.predict(x_sampled).to(device)

                xent_loss = torch.nansum(- inv_prop*(sub_y*(torch.log( lamb1*pred )-lamb1*sub_d) + (1-sub_y)*torch.log( torch.exp(-lamb1*sub_e)*pred + 1 - pred )))

                imputation_loss = torch.nansum(- (imputation_y*(torch.log( lamb1*pred )-lamb1*sub_d) + (1-imputation_y)*torch.log( torch.exp(-lamb1*sub_e)*pred + 1 - pred )))
                
                # xent_loss = F.binary_cross_entropy(pred, sub_y, weight=inv_prop, reduction="sum") # o*eui/pui
                # imputation_loss = F.binary_cross_entropy(pred, imputation_y, reduction="sum")                 
                
                ips_loss = (xent_loss - imputation_loss)/selected_idx.shape[0]
                
                # direct loss

                # direct_loss = F.binary_cross_entropy(pred_u, imputation_y1, reduction="sum")
                direct_loss = torch.nansum(- (imputation_y1*(torch.log( lamb2*pred_u )-lamb2*sub_d) + (1-imputation_y1)*torch.log( torch.exp(-lamb2*sub_e)*pred_u + 1 - pred_u )))

                direct_loss = (direct_loss)/(x_sampled.shape[0])

                loss = ips_loss + direct_loss 

                optimizer_both.zero_grad()
                loss.backward()
                optimizer_both.step()

                epoch_loss += xent_loss.detach().cpu().numpy()                

                pred = self.prediction_model.predict(sub_x).to(device)
                imputation_y = self.imputation.forward(sub_x)                
                
                e_loss = F.binary_cross_entropy(pred, sub_y, reduction="none")
                e_hat_loss = F.binary_cross_entropy(imputation_y, pred, reduction="none")
                # e_hat_loss = F.binary_cross_entropy(pred, imputation_y, reduction="none")
                imp_loss = (((e_loss - e_hat_loss) ** 2) * inv_prop).sum()
                
                optimizer_imputation.zero_grad()
                imp_loss.backward()
                optimizer_imputation.step()                
             
                
            relative_loss_div = (last_loss-epoch_loss)/(last_loss+1e-10)
            if  relative_loss_div < tol:
                if early_stop > 5:
                    print("[MF-DR-JL-DF] epoch:{}, xent:{}".format(epoch, epoch_loss))
                    break
                early_stop += 1
                
            last_loss = epoch_loss

            if epoch % 10 == 0 and verbose:
                print("[MF-DR-JL-DF] epoch:{}, xent:{}".format(epoch, epoch_loss))

            if epoch == num_epoch - 1:
                print("[MF-DR-JL-DF] Reach preset epochs, it seems does not converge.")

    def predict(self, x):
        pred = self.prediction_model.predict(x)
        pred = self.sigmoid(pred)
        return pred.detach().cpu().numpy()

    def _compute_IPS(self,x,y,y_ips=None):
        if y_ips is None:
            one_over_zl = np.ones(len(y))
        else:
            py1 = y_ips.sum() / len(y_ips)
            py0 = 1 - py1
            po1 = len(x) / (x[:,0].max() * x[:,1].max())
            py1o1 = y.sum() / len(y)
            py0o1 = 1 - py1o1
            propensity = np.zeros(len(y))
            propensity[y == 0] = (py0o1 * po1) / py0
            propensity[y == 1] = (py1o1 * po1) / py1
            one_over_zl = 1 / propensity

        one_over_zl = torch.Tensor(one_over_zl)
        return one_over_zl  