import torch
import numpy as np
import os
from torch import nn
from src.configs import cfg
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))

from core.parallel_computing_Module import ParallelComputing

from core.RAN_encoder import RanEncoder
from core.RAN_decoder import RanDecoder



class HandSomeRAN(nn.Module):
    def __init__(self,
                 drug_in_channel, protein_in_channel,
                 drug_out_channel, protein_out_channel,
                 ):
        super().__init__()

        # self.protein_drug_index = protein_drug_index

        self.parallel_computing_for_protein = None
        self.parallel_computing_for_drug = None

        self.drug_in_channel = drug_in_channel
        self.drug_out_channel = drug_out_channel
        self.protein_in_channel = protein_in_channel
        self.protein_out_channel = protein_out_channel
        self.disease_out_channel = 48

        self.fc1_for_drug = nn.Linear(self.drug_in_channel, self.drug_out_channel)
        self.fc1_for_protein = nn.Linear(self.protein_in_channel, self.protein_out_channel)

        self.fc2_for_drug = nn.Linear(400, 64)
        self.fc2_for_protein = nn.Linear(400, 64)

        self.encoder = RanEncoder(8, 2, 4)
        self.decoder = RanDecoder(4, 64, 8)

    def forward(self, proteins_feature, drugs_feature, indexes):

        protein_index, drug_index = indexes
        self.parallel_computing_for_drug = ParallelComputing(cfg.disease_feature_path, 90,
                                                             disease_out_channels=self.disease_out_channel,
                                                             mat_x_disease_path=cfg.mat_drug_disease_path,
                                                             x_features_dim=200, x_indexes=drug_index)
        self.parallel_computing_for_protein = ParallelComputing(cfg.disease_feature_path, 90,
                                                                disease_out_channels=self.disease_out_channel,
                                                                mat_x_disease_path=cfg.mat_protein_disease_path,
                                                                x_features_dim=200,
                                                                x_indexes=protein_index)
        proteins_feature = proteins_feature.float()
        proteins_features = self.fc1_for_protein(proteins_feature)
        proteins_features = proteins_features.float()
        drugs_feature = drugs_feature.float()
        drugs_features = self.fc1_for_drug(drugs_feature)

        drugs_features_parallel = self.parallel_computing_for_drug(drugs_features)

        drugs_features = drugs_features.unsqueeze(1)
        drugs_features = drugs_features.repeat(1, 1, 2)

        first_dim = drugs_features.shape[0]
        drugs_features = drugs_features.reshape(first_dim, -1)
        drugs_features_parallel = drugs_features_parallel.reshape(first_dim, -1).cuda()
        drugs_features = self.fc2_for_drug(drugs_features)

        drugs_features_parallel = self.fc2_for_drug(drugs_features_parallel)

        drugs_features = drugs_features.unsqueeze(1)
        drugs_features_parallel = drugs_features_parallel.unsqueeze(1)
        drugs_features_integrated = torch.cat((drugs_features, drugs_features_parallel), 1)

        drugs_features_encoded = self.encoder(drugs_features_integrated)


        first_dim = proteins_features.shape[0]
        proteins_features_parallel = self.parallel_computing_for_protein(proteins_features)
        proteins_features_parallel = proteins_features_parallel.reshape(first_dim, -1).cuda()
        proteins_features = proteins_features.unsqueeze(1)
        proteins_features = proteins_features.repeat(1, 1, 2)
        proteins_features = proteins_features.reshape(first_dim, -1)

        proteins_features = self.fc2_for_protein(proteins_features)

        proteins_features_parallel = self.fc2_for_protein(proteins_features_parallel)

        proteins_features = proteins_features.unsqueeze(1)
        proteins_features_parallel = proteins_features_parallel.unsqueeze(1)
        proteins_features_integrated = torch.cat((proteins_features, proteins_features_parallel), 1)


        proteins_features_encoded = self.encoder(proteins_features_integrated)


        out = self.decoder(drugs_features_encoded, proteins_features_encoded)

        return out
