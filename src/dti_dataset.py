"""
用于生成每一对dti
格式
((protein_feature,drug_feature),label)
"""

import os

import numpy as np
from torch.utils.data import Dataset


class DTIDataset(Dataset):
    def __init__(self, pairs_info, pro_features, drug_features):
        self.pairs_info = pairs_info
        self.protein_feature = pro_features
        self.drug_feature = drug_features

    def __getitem__(self, item):
        pair, label = self.pairs_info[item]

        protein_id = pair[0]
        drug_id = pair[1]
        pair_index = [protein_id, drug_id]

        protein_feature = self.protein_feature[protein_id]
        drug_feature = self.drug_feature[drug_id]
        pro_dru_feature = (protein_feature, drug_feature)
        return (pro_dru_feature, pair_index), label

    def __len__(self):
        return len(self.pairs_info)



