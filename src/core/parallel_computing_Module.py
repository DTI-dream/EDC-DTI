import torch
import numpy as np
from torch import nn


class ParallelComputing(nn.Module):
    def __init__(self, disease_feature_path, disease_in_channels, disease_out_channels, mat_x_disease_path, x_indexes,
                 x_features_dim,
                 device='cpu'):

        super().__init__()
        self.x_feature_2 = None
        self.x_feature_1 = None
        self.device = device
        self.disease_feature = torch.tensor(np.array(np.load(disease_feature_path, allow_pickle=True), dtype=float),
                                            dtype=torch.float,
                                            device=device)
        self.disease_in_channels = disease_in_channels
        self.disease_out_channels = disease_out_channels

        self.disease_fc = nn.Linear(self.disease_in_channels, self.disease_out_channels)

        self.mat_x_disease = np.load(mat_x_disease_path)

        self.x_indexes = x_indexes

        self.x_features_dim = x_features_dim

        self.dis_trans = nn.Parameter(torch.empty(self.disease_in_channels, self.disease_out_channels))
        nn.init.xavier_normal_(self.dis_trans)

        self.disease_dimension_transform = nn.Parameter(torch.empty(self.disease_out_channels, x_features_dim)).cuda()

        nn.init.xavier_normal_(self.disease_dimension_transform)
        self.dim_reduction_mat = nn.Parameter(torch.empty(self.x_features_dim * 2, self.x_features_dim)).cuda()
        nn.init.xavier_normal_(self.dim_reduction_mat)

        self.disease_feature_processing()

    def disease_feature_processing(self):

        self.disease_feature = torch.mm(self.disease_feature, self.dis_trans)

    def forward(self, x_features):

        self.x_feature_matrix = np.empty((len(self.x_indexes), 1, self.x_features_dim * 2))
        self.x_feature_matrix = torch.tensor(self.x_feature_matrix, dtype=torch.float, device=self.device)
        index_for_selected_one = self.x_indexes.tolist()

        out_inner_index_map = {}
        inner_index = 0
        for i in index_for_selected_one:
            i = int(i)
            out_inner_index_map[i] = inner_index
            x_feature = self.x_feature_generator(out_inner_index_map[i], x_features)
            self.x_feature_matrix[out_inner_index_map[i]] = x_feature
            inner_index += 1

        return self.x_feature_matrix

    def x_feature_generator(self, x_index, x_features):

        mask_mat = self.mask_matrix_generator(x_index)

        mask_mat = torch.tensor(mask_mat, dtype=torch.float, device=self.device)
        mask_dis_feature = torch.mm(mask_mat, self.disease_feature).cuda()

        mask_dis_feature = torch.mm(mask_dis_feature, self.disease_dimension_transform)
        x_feature = x_features[x_index]
        x_feature = x_feature.unsqueeze(-1)
        attention_coefficient = torch.mm(mask_dis_feature, x_feature)

        attention_coefficient = attention_coefficient.reshape(-1, 1)

        attention_coefficient = attention_coefficient.t()

        disease_contribution_for_x = torch.mm(attention_coefficient, mask_dis_feature)

        x_feature = x_feature.reshape(-1, 1)
        x_feature_transposed = x_feature.t()

        self.x_feature_1 = x_feature_transposed + disease_contribution_for_x

        self.x_feature_2 = torch.cat((x_feature_transposed, disease_contribution_for_x), 1)

        self.x_feature_2 = torch.mm(self.x_feature_2, self.dim_reduction_mat)
        x_feature_combine = torch.cat((self.x_feature_1, self.x_feature_2), 1)
        return x_feature_combine

    def mask_matrix_generator(self, x_index):

        true_disease_indexes = np.nonzero(self.mat_x_disease[x_index])
        disease_num_all = self.disease_feature.shape[0]
        np_mask_mat = np.zeros(disease_num_all * disease_num_all).reshape(disease_num_all, disease_num_all)
        for i in true_disease_indexes:
            np_mask_mat[i, i] = 1
        return np_mask_mat
