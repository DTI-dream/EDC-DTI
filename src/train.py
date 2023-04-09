"""
@author: Neo
@notion: think different
"""
import argparse
from dti_dataset import DTIDataset
import sys
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
from torch.utils.data import DataLoader
from main_ran import HandSomeRAN
from utils import *
from configs import cfg
import torch.optim as optim
from model_trainer import ModelTrainer
from loss_curve import *

setup_seed(12345) 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--drug_feature_path', type=str, required=False,
                        default=r"../Data/feature for protein drug/drug_vector_d100.txt",
                        help='Path to the drug feature file')
    parser.add_argument('--protein_feature_path', type=str, required=False,
                        default=r"../Data/feature for protein drug/protein_vector_d400.txt",
                        help='Path to the protein feature file')
    parser.add_argument('--disease_feature_path', type=str, required=False,
                        default="../Data/feature for protein drug/disease_feature.npy",
                        help='Path to the disease feature file')
    parser.add_argument('--train_val_path', type=str, required=False,
                        default="../Data/train_val",
                        help='Path to the train_val directory')
    args = parser.parse_args()

    res_dir = os.path.join(BASE_DIR, "..", "..", "results")
    logger, log_dir = make_logger(res_dir)

    drug_feature_path = args.drug_feature_path
    drugs_feature = np.loadtxt(drug_feature_path)

    protein_feature_path = args.protein_feature_path
    proteins_feature = np.loadtxt(protein_feature_path)

    disease_feature_path = args.disease_feature_path
    diseases_feature = np.load(disease_feature_path, allow_pickle=True)

    set_path = args.train_val_path
    train_set = np.load(os.path.join(set_path, 'train_set_1_1.npy'), allow_pickle=True)
    val_set = np.load(os.path.join(set_path, 'val_set_1_1.npy'), allow_pickle=True)
    test_set = np.load(os.path.join(set_path, 'test_set_1_1.npy'), allow_pickle=True)

    print(drugs_feature.shape, proteins_feature.shape, diseases_feature.shape, train_set.shape,
          val_set.shape, test_set.shape)

    train_data = DTIDataset(train_set, proteins_feature, drugs_feature)
    val_data = DTIDataset(val_set, proteins_feature, drugs_feature)

    train_loader = DataLoader(dataset=train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.workers)
    valid_loader = DataLoader(dataset=val_data, batch_size=cfg.valid_batchSize, num_workers=cfg.workers)

    model = HandSomeRAN(drug_in_channel=100, protein_in_channel=400, drug_out_channel=200,
                        protein_out_channel=200)
    model.to(device)

    loss_f = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, gamma=cfg.factor, milestones=cfg.milestones)

    logger.info("cfg:\n{}\n loss_f:\n{}\n scheduler:\n{}\n optimizer:\n{}\n model:\n{}".format(cfg, loss_f, scheduler,
                                                                                               optimizer, model))

    loss_rec = {"src": [], "valid": []}
    acc_rec = {"src": [], "valid": []}

    best_acc, best_epoch = 0, 0
    for epoch in range(cfg.max_epoch):

        loss_train, acc_train, path_error_train, auc_train = ModelTrainer.train(
            train_loader, model, loss_f, optimizer, epoch, device, logger, cfg)

        loss_valid, acc_valid, path_error_valid, auc_valid, f1_valid = ModelTrainer.valid(
            valid_loader, model, loss_f, device, cfg)

        logger.info(
            "Epoch[{:0>3}/{:0>3}] Train Acc: {:.2%} Valid Acc:{:.2%} Train loss:{:.4f} Valid loss:{:.4f} LR:{} "
            "train_auc:{:.2%}, valid_auc:{:.2%} valid_f1:{:.2%}". \
                format(epoch + 1, cfg.max_epoch, acc_train, acc_valid, loss_train, loss_valid,
                       optimizer.param_groups[0]["lr"], auc_train, auc_valid, f1_valid))
        scheduler.step()

        loss_rec["src"].append(loss_train), loss_rec["valid"].append(loss_valid)
        acc_rec["src"].append(acc_train), acc_rec["valid"].append(acc_valid)

        plt_x = np.arange(1, epoch + 2)
        plot_line(plt_x, loss_rec["src"], plt_x, loss_rec["valid"], mode="loss", out_dir=log_dir)
        plot_line(plt_x, acc_rec["src"], plt_x, acc_rec["valid"], mode="acc", out_dir=log_dir)

        if best_acc < acc_valid or epoch == cfg.max_epoch - 1:
            best_epoch = epoch if best_acc < acc_valid else best_epoch
            best_acc = acc_valid if best_acc < acc_valid else best_acc
            checkpoint = {"model_state_dict": model.state_dict(),
                          "optimizer_state_dict": optimizer.state_dict(),
                          "epoch": epoch,
                          "best_acc": best_acc}
            pkl_name = "checkpoint_{}.pkl".format(epoch) if epoch == cfg.max_epoch - 1 else "checkpoint_best.pkl"
            path_checkpoint = os.path.join(log_dir, pkl_name)
            torch.save(model, "./model.pth")
    logger.info("{} done, best acc: {} in :{}".format(
        datetime.strftime(datetime.now(), '%m-%d_%H-%M'), best_acc, best_epoch))

