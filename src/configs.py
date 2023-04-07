import os
import sys


from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))


cfg = EasyDict()
cfg.workers = 4
cfg.batch_size = 64

cfg.lr_init = 0.01
cfg.momentum = 0.9
cfg.weight_decay = 1e-4
cfg.factor = 0.1
cfg.milestones = [30, 45]
cfg.max_epoch = 200
cfg.log_interval = 10
cfg.disease_feature_path = r"../Data/binary_data/disease_feature_90.npy"
cfg.mat_drug_disease_path = r"../Data/binary_data/mat_drug_disease_90.npy"
cfg.mat_protein_disease_path = r"../Data/binary_data/mat_protein_disease_90.npy"
cfg.disease_out_channel = 48
cfg.valid_batchSize = 32

