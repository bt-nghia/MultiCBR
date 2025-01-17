#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.sparse as sp 
from models.AsymModule import *

def cal_bpr_loss(pred):
    # pred: [bs, 1+neg_num]
    if pred.shape[1] > 2:
        negs = pred[:, 1:]
        pos = pred[:, 0].unsqueeze(1).expand_as(negs)
    else:
        negs = pred[:, 1].unsqueeze(1)
        pos = pred[:, 0].unsqueeze(1)

    loss = - torch.log(torch.sigmoid(pos - negs)) # [bs]
    loss = torch.mean(loss)

    return loss


def laplace_transform(graph):
    rowsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=1).A.ravel()) + 1e-8))
    colsum_sqrt = sp.diags(1/(np.sqrt(graph.sum(axis=0).A.ravel()) + 1e-8))
    graph = rowsum_sqrt @ graph @ colsum_sqrt

    return graph


def to_tensor(graph):
    graph = graph.tocoo()
    values = graph.data
    indices = np.vstack((graph.row, graph.col))
    graph = torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(graph.shape))

    return graph


def np_edge_dropout(values, dropout_ratio):
    mask = np.random.choice([0, 1], size=(len(values),), p=[dropout_ratio, 1-dropout_ratio])
    values = mask * values
    return values


class MultiCBR(nn.Module):
    def __init__(self, conf, raw_graph):
        super().__init__()
        self.conf = conf
        device = self.conf["device"]
        self.device = device

        self.embedding_size = conf["embedding_size"]
        self.embed_L2_norm = conf["l2_reg"]
        self.num_users = conf["num_users"]
        self.num_bundles = conf["num_bundles"]
        self.num_items = conf["num_items"]
        self.num_layers = self.conf["num_layers"]
        self.c_temp = self.conf["c_temp"]

        self.fusion_weights = conf['fusion_weights']

        self.init_emb()
        self.init_fusion_weights()

        assert isinstance(raw_graph, list)
        self.ub_graph, self.ui_graph, self.bi_graph = raw_graph

        #item co-ocurence graph
        self.ubi_graph = self.ub_graph @ self.bi_graph
        
        # generate the graph without any dropouts for testing
        self.UB_propagation_graph_ori = self.get_propagation_graph(self.ub_graph)
        self.UI_propagation_graph_ori = self.get_propagation_graph(self.ui_graph)
        self.UI_aggregation_graph_ori = self.get_aggregation_graph(self.ui_graph)
        self.BI_propagation_graph_ori = self.get_propagation_graph(self.bi_graph)
        self.BI_aggregation_graph_ori = self.get_aggregation_graph(self.bi_graph)

        # generate the graph with the configured dropouts for training, if aug_type is OP or MD, the following graphs with be identical with the aboves
        self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])
        self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])
        self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
        self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        if self.conf['aug_type'] == 'MD':
            self.init_md_dropouts()
        elif self.conf['aug_type'] == "Noise":
            self.init_noise_eps()

        self.n_head = 2
        self.a_self_loop = False
        self.extra_layer = True
        self.ibi_edge_index = torch.tensor(np.load("datasets/{}/n_neigh_ibi.npy".format(conf["dataset"]), allow_pickle=True)).to(self.device)
        self.iui_edge_index = torch.tensor(np.load("datasets/{}/n_neigh_iui.npy".format(conf["dataset"]), allow_pickle=True)).to(self.device)
        self.iui_asym = Amatrix(in_dim=64, out_dim=64, n_layer=1, dropout=0.1, heads=self.n_head, concat=False, self_loop=self.a_self_loop, extra_layer=self.extra_layer)
        self.ibi_asym = Amatrix(in_dim=64, out_dim=64, n_layer=1, dropout=0.1, heads=self.n_head, concat=False, self_loop=self.a_self_loop, extra_layer=self.extra_layer)
        self.bs = 2048

    def init_md_dropouts(self):
        self.UB_dropout = nn.Dropout(self.conf["UB_ratio"], True)
        self.UI_dropout = nn.Dropout(self.conf["UI_ratio"], True)
        self.BI_dropout = nn.Dropout(self.conf["BI_ratio"], True)
        self.mess_dropout_dict = {
            "UB": self.UB_dropout,
            "UI": self.UI_dropout,
            "BI": self.BI_dropout,
        }


    def init_noise_eps(self):
        self.UB_eps = self.conf["UB_ratio"]
        self.UI_eps = self.conf["UI_ratio"]
        self.BI_eps = self.conf["BI_ratio"]
        self.eps_dict = {
            "UB": self.UB_eps,
            "UI": self.UI_eps,
            "BI": self.BI_eps,
        }


    def init_emb(self):
        '''
        Try init normal
        '''
        self.users_feature = nn.Parameter(torch.FloatTensor(self.num_users, self.embedding_size))
        nn.init.xavier_normal_(self.users_feature)
        self.bundles_feature = nn.Parameter(torch.FloatTensor(self.num_bundles, self.embedding_size))
        nn.init.xavier_normal_(self.bundles_feature)
        self.items_feature = nn.Parameter(torch.FloatTensor(self.num_items, self.embedding_size))
        nn.init.xavier_normal_(self.items_feature)


    def init_fusion_weights(self):
        assert (len(self.fusion_weights['modal_weight']) == 3), \
            "The number of modal fusion weights does not correspond to the number of graphs"

        assert  (len(self.fusion_weights['UB_layer']) == self.num_layers + 1) and\
                (len(self.fusion_weights['UI_layer']) == self.num_layers + 1) and\
                (len(self.fusion_weights['BI_layer']) == self.num_layers + 1),\
            "The number of layer fusion weights does not correspond to number of layers"

        modal_coefs = torch.FloatTensor(self.fusion_weights['modal_weight'])
        UB_layer_coefs = torch.FloatTensor(self.fusion_weights['UB_layer'])
        UI_layer_coefs = torch.FloatTensor(self.fusion_weights['UI_layer'])
        BI_layer_coefs = torch.FloatTensor(self.fusion_weights['BI_layer'])

        self.modal_coefs = modal_coefs.unsqueeze(-1).unsqueeze(-1).to(self.device)

        self.UB_layer_coefs = UB_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.UI_layer_coefs = UI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)
        self.BI_layer_coefs = BI_layer_coefs.unsqueeze(0).unsqueeze(-1).to(self.device)


    def get_propagation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device
        propagation_graph = sp.bmat([[sp.csr_matrix((bipartite_graph.shape[0], bipartite_graph.shape[0])), bipartite_graph], [bipartite_graph.T, sp.csr_matrix((bipartite_graph.shape[1], bipartite_graph.shape[1]))]])

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = propagation_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(device)
    
    def get_self_propagation_graph(self, co_graph, modification_ratio=0, threshold=20):
        propagation_graph = co_graph * (co_graph >= threshold)

        if modification_ratio != 0:
            if self.conf['aug_type'] == 'ED':
                graph = propagation_graph.tocoo()
                vals = np_edge_dropout(graph.data, modification_ratio)
                propagation_graph = sp.coo_matrix((vals, (graph.row, graph.col)), shape=graph.shape).tocsr()

        return to_tensor(laplace_transform(propagation_graph)).to(self.device)

    def get_aggregation_graph(self, bipartite_graph, modification_ratio=0):
        device = self.device

        if modification_ratio != 0:
            if self.conf["aug_type"] == "ED":
                graph = bipartite_graph.tocoo()
                values = np_edge_dropout(graph.data, modification_ratio)
                bipartite_graph = sp.coo_matrix((values, (graph.row, graph.col)), shape=graph.shape).tocsr()

        bundle_size = bipartite_graph.sum(axis=1) + 1e-8
        bipartite_graph = sp.diags(1/bundle_size.A.ravel()) @ bipartite_graph
        return to_tensor(bipartite_graph).to(device)


    def propagate(self, graph, A_feature, B_feature, graph_type, layer_coef, test):
        features = torch.cat((A_feature, B_feature), 0)
        all_features = [features]

        for i in range(self.num_layers):
            features = torch.spmm(graph, features)
            if self.conf["aug_type"] == "MD" and not test:
                mess_dropout = self.mess_dropout_dict[graph_type]
                features = mess_dropout(features)
            elif self.conf["aug_type"] == "Noise" and not test:
                random_noise = torch.rand_like(features).to(self.device)
                eps = self.eps_dict[graph_type]
                features += torch.sign(features) * F.normalize(random_noise, dim=-1) * eps

            all_features.append(F.normalize(features, p=2, dim=1))

        all_features = torch.stack(all_features, 1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        A_feature, B_feature = torch.split(all_features, (A_feature.shape[0], B_feature.shape[0]), 0)

        return A_feature, B_feature

    def ii_propagate(self, ii_graph, i_feat, graph_type, layer_coef, test):
        all_features = [i_feat]
        for i in range(self.num_layers):
            i_feat = torch.spmm(ii_graph, i_feat)
            all_features.append(F.normalize(i_feat, p=2, dim=1))
        all_features = torch.stack(all_features, dim=1) * layer_coef
        all_features = torch.sum(all_features, dim=1)
        return all_features / self.num_layers

    def aggregate(self, agg_graph, node_feature, graph_type, test):
        aggregated_feature = torch.matmul(agg_graph, node_feature)

        # simple embedding dropout on bundle embeddings
        if self.conf["aug_type"] == "MD" and not test:
            mess_dropout = self.mess_dropout_dict[graph_type]
            aggregated_feature = mess_dropout(aggregated_feature)
        elif self.conf["aug_type"] == "Noise" and not test:
            random_noise = torch.rand_like(aggregated_feature).to(self.device)
            eps = self.eps_dict[graph_type]
            aggregated_feature += torch.sign(aggregated_feature) * F.normalize(random_noise, dim=-1) * eps

        return aggregated_feature


    def fuse_users_bundles_feature(self, users_feature, bundles_feature):
        users_feature = torch.stack(users_feature, dim=0)
        bundles_feature = torch.stack(bundles_feature, dim=0)

        # Modal aggregation
        users_rep = torch.sum(users_feature * self.modal_coefs, dim=0)
        bundles_rep = torch.sum(bundles_feature * self.modal_coefs, dim=0)

        return users_rep, bundles_rep
    
    def bi_propagate(self, bundle_feature, item_feature, edge, layer_coefs):
        edge[1:] = edge[1:] + self.num_bundles
        feat = torch.concat([bundle_feature, item_feature], dim=0)
        feats = [feat]
        for conv in self.gat_convs:
            feat = conv(feat, edge)
            feats.append(feat)
        feats = torch.stack(feats, dim=1) * layer_coefs
        feats = torch.sum(feats, dim=1)
        bundle_feature, item_feature = torch.split([self.num_bundles, self.num_items], dim=0)
        return bundle_feature, item_feature


    def get_multi_modal_representations(self, test=False):
        #  =============================  UB graph propagation  =============================
        if test:
            # UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph_ori, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
        else:
            # UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)
            UB_users_feature, UB_bundles_feature = self.propagate(self.UB_propagation_graph, self.users_feature, self.bundles_feature, "UB", self.UB_layer_coefs, test)

        #  =============================  UI graph propagation  =============================
        # item_feat1, _ = self.iui_asym(self.items_feature, self.iui_edge_index, return_attention_weights=True)
        # item_feat1 = self.items_feature * 0.2 + item_feat1 * 0.8
        if test:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph_ori, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UI_items_feature, "BI", test)
        else:
            UI_users_feature, UI_items_feature = self.propagate(self.UI_propagation_graph, self.users_feature, self.items_feature, "UI", self.UI_layer_coefs, test)
            UI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UI_items_feature, "BI", test)

        #  =============================  BI graph propagation  =============================
        # item_feat2, _ = self.ibi_asym(self.items_feature, self.ibi_edge_index, return_attention_weights=True)
        # item_feat2 = self.items_feature * 0.2 + item_feat2 * 0.8
        if test:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph_ori, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph_ori, BI_items_feature, "UI", test)
        else:
            BI_bundles_feature, BI_items_feature = self.propagate(self.BI_propagation_graph, self.bundles_feature, self.items_feature, "BI", self.BI_layer_coefs, test)
            BI_users_feature = self.aggregate(self.UI_aggregation_graph, BI_items_feature, "UI", test)

        # ==============================  UBI graph propagation =============================
        # if test:
        #     UBI_users_feature, UBI_items_feature = self.propagate(self.UBI_propagation_graph_ori, self.users_feature, self.items_feature,"UBI", self.UBI_layer_coefs, test)
        #     UBI_bundles_feature = self.aggregate(self.BI_aggregation_graph_ori, UBI_items_feature, "BI", test)
        # else:
        #     UBI_users_feature, UBI_items_feature = self.propagate(self.UBI_propagation_graph, self.users_feature, self.items_feature, "UBI", self.UBI_layer_coefs, test)
        #     UBI_bundles_feature = self.aggregate(self.BI_aggregation_graph, UBI_items_feature, "BI", test)


        users_feature = [UB_users_feature, UI_users_feature, BI_users_feature]
        bundles_feature = [UB_bundles_feature, UI_bundles_feature, BI_bundles_feature]

        users_rep, bundles_rep = self.fuse_users_bundles_feature(users_feature, bundles_feature)
        return users_rep, bundles_rep


    def cal_c_loss(self, pos, aug):
        # pos: [batch_size, :, emb_size]
        # aug: [batch_size, :, emb_size]
        pos = pos[:, 0, :]
        aug = aug[:, 0, :]

        pos = F.normalize(pos, p=2, dim=1)
        aug = F.normalize(aug, p=2, dim=1)
        pos_score = torch.sum(pos * aug, dim=1) # [batch_size]
        ttl_score = torch.matmul(pos, aug.permute(1, 0)) # [batch_size, batch_size]

        pos_score = torch.exp(pos_score / self.c_temp) # [batch_size]
        ttl_score = torch.sum(torch.exp(ttl_score / self.c_temp), axis=1) # [batch_size]

        c_loss = - torch.mean(torch.log(pos_score / ttl_score))

        return c_loss


    def cal_loss(self, users_feature, bundles_feature):
        # users_feature / bundles_feature: [bs, 1+neg_num, emb_size]
        pred = torch.sum(users_feature * bundles_feature, 2)
        bpr_loss = cal_bpr_loss(pred)

        # cl is abbr. of "contrastive loss"
        u_view_cl = self.cal_c_loss(users_feature, users_feature)
        b_view_cl = self.cal_c_loss(bundles_feature, bundles_feature)
        c_losses = [u_view_cl, b_view_cl]
        c_loss = sum(c_losses) / len(c_losses)

        ids = torch.randperm(self.ibi_edge_index.shape[1])
        ids = ids[:self.bs]

        item_feat1 = self.items_feature[self.ibi_edge_index[:,ids]]
        item_feat2 = self.items_feature[self.ibi_edge_index[:,ids]]
        cosine_loss = self.cal_cosine_loss(item_feat1, item_feat2)

        return bpr_loss, c_loss + cosine_loss * 0.02


    def forward(self, batch, ED_drop=False):
        # the edge drop can be performed by every batch or epoch, should be controlled in the train loop
        if ED_drop:
            self.UB_propagation_graph = self.get_propagation_graph(self.ub_graph, self.conf["UB_ratio"])

            self.UI_propagation_graph = self.get_propagation_graph(self.ui_graph, self.conf["UI_ratio"])
            self.UI_aggregation_graph = self.get_aggregation_graph(self.ui_graph, self.conf["UI_ratio"])

            self.BI_propagation_graph = self.get_propagation_graph(self.bi_graph, self.conf["BI_ratio"])
            self.BI_aggregation_graph = self.get_aggregation_graph(self.bi_graph, self.conf["BI_ratio"])

        # users: [bs, 1]
        # bundles: [bs, 1+neg_num]
        users, bundles = batch
        users_rep, bundles_rep = self.get_multi_modal_representations()

        users_embedding = users_rep[users].expand(-1, bundles.shape[1], -1)
        bundles_embedding = bundles_rep[bundles]

        bpr_loss, c_loss = self.cal_loss(users_embedding, bundles_embedding)

        return bpr_loss, c_loss


    def evaluate(self, propagate_result, users):
        users_feature, bundles_feature = propagate_result
        scores = torch.mm(users_feature[users], bundles_feature.t())
        return scores
    
    def cal_topK_c_loss(self, pos, aug, kp=30, kn=2000, threshold=5e-1):
        '''
        contrastive loss for top k pairs
        kp: topk positive
        kn: topk negative
        '''
        pos = F.normalize(pos[:, 0, :], p=2, dim=1)
        aug = F.normalize(aug[:, 0, :], p=2, dim=1)

        sim = pos @ aug.T

        topK_p_set = torch.topk(sim, k=kp, dim=1)
        topk_n_set = torch.topk(sim, k=kn, dim=1)

        pos_score = torch.sum(torch.exp(topK_p_set.values / self.c_temp))
        neg_score = torch.sum(torch.exp(topk_n_set.values / self.c_temp))

        return -torch.mean(torch.log(pos_score / neg_score))
    
    def cal_cosine_loss(self, feat1, feat2):
        '''
        feat1 [batch_size, n_dim]
        feat2 [batch_size, n_dim]
        maximize cosine similarity between 2 item have dege
        return cosine similarity [batchsize,1]
        '''
        bs = feat1.shape[0]
        sum_dot_prod = torch.sum(feat1 * feat2, dim=1).view(bs, 1)
        norm1 = torch.sqrt(torch.sum(feat1 * feat1, dim=1)).view(bs, 1)
        norm2 = torch.sqrt(torch.sum(feat2 * feat2, dim=1)).view(bs, 1)

        return -torch.mean(sum_dot_prod / norm1 / norm2)
    

class Amatrix(nn.Module):
    def __init__(self, in_dim, out_dim, n_layer=1, dropout=0.0, heads=2, concat=False, self_loop=True, extra_layer=False):
        super(Amatrix, self).__init__()
        self.num_layer = n_layer
        self.dropout = dropout
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.concat = concat
        self.self_loop = self_loop
        self.extra_layer = extra_layer
        self.convs = nn.ModuleList([AsymMatrix(in_channels=self.in_dim, 
                                              out_channels=self.out_dim, 
                                              dropout=self.dropout,
                                              heads=self.heads,
                                              concat=self.concat,
                                              add_self_loops=self.self_loop,
                                              extra_layer=self.extra_layer) 
                                              for _ in range(self.num_layer)])


    def forward(self, x, edge_index, return_attention_weights=True):
        feats = [x]
        attns = []

        for conv in self.convs:
            x, attn = conv(x, edge_index, return_attention_weights=return_attention_weights)
            feats.append(x)
            attns.append(attn)

        feat = torch.stack(feats, dim=1)
        x = torch.mean(feat, dim=1)
        return x, attns