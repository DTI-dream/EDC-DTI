import torch
import numpy as np
from collections import Counter
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

class ModelTrainer(object):

    @staticmethod
    def train(data_loader, model, loss_f, optimizer, epoch_idx, device, logger, cfg):
        global auc_value
        model.train()

        loss_sigma = []
        loss_mean = 0
        acc_avg = 0
        path_error = []
        label_list = []

        # conf_mat = torch.zeros((cfg.batch_size, cfg.batch_size))
        for i, data in enumerate(data_loader):

            # _, labels = data
            pairs_indexes, labels = data
            label_list.extend(labels.tolist())

            # inputs, labels = data
            # 这里需要将pairs_indexes转化成torch.tensor形式
            # 第一个是protein 400,第二个是durg 100
            # 我试一下分开传入
            pairs = pairs_indexes[0]
            proteins_feature, drugs_feature = pairs
            proteins_feature = proteins_feature.to(device)
            drugs_feature = drugs_feature.to(device)
            indexes = pairs_indexes[1]
            # pairs = [item.detach().numpy() for item in pairs]
            indexes = [item.detach().numpy() for item in indexes]
            indexes = torch.Tensor(np.array(indexes)).to(device)
            # pairs_indexes_list = [pairs, indexes]
            # pairs_indexes_list_tensor = torch.Tensor(pairs_indexes_list)
            # inputs = pairs_indexes_list_tensor.to(device)
            labels = labels.long().to(device)
            #labels = labels.unsqueeze(-1)
            # forward & backward
            outputs = model(proteins_feature, drugs_feature, indexes)
            optimizer.zero_grad()
            # 计算auc
            outputs_class = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            auc_value = roc_auc_score(labels.cpu().detach().numpy(), outputs_class)
            # 计算f1
            f1_train = f1_score(labels.cpu().detach().numpy(), outputs_class)

            loss = loss_f(outputs.cpu(), labels.cpu())
            loss.backward()
            optimizer.step()

            # 统计loss
            loss_sigma.append(loss.item())
            loss_mean = np.mean(loss_sigma)

            true_list = []
            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].detach().cpu().numpy()
                # conf_mat[cate_i, pre_i] += 1.
                if cate_i == pre_i:
                    true_list.append(j)

                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i))  # 记录错误样本的信息

            # 统计loss
            loss_sigma.append(loss.item())

            acc_avg = len(true_list) / len(labels)

            # 每10个iteration 打印一次训练信息
            if i % cfg.log_interval == cfg.log_interval - 1:
                logger.info("Training: Epoch[{:0>3}/{:0>3}] Iteration[{:0>3}/{:0>3}] Loss: {:.4f} Acc:{:.2%} train_auc:{:.2%} f1:{:.2%}".
                            format(epoch_idx + 1, cfg.max_epoch, i + 1, len(data_loader), loss_mean, acc_avg, auc_value, f1_train))
        logger.info("epoch:{} sampler: {}".format(epoch_idx, Counter(label_list)))
        return loss_mean, acc_avg, path_error, auc_value

    @staticmethod
    def valid(data_loader, model, loss_f, device, cfg):
        model.eval()

        # class_num = data_loader.dataset.cls_num
        # conf_mat = np.zeros((cfg.batch_size, cfg.batch_size))
        loss_sigma = []
        path_error = []

        for i, data in enumerate(data_loader):
            # _, labels = data
            pairs_indexes, labels = data
            # label_list.extend(labels.tolist())

            # inputs, labels = data
            # 这里需要将pairs_indexes转化成torch.tensor形式
            # 第一个是protein 400,第二个是durg 100
            # 我试一下分开传入
            pairs = pairs_indexes[0]
            proteins_feature, drugs_feature = pairs
            proteins_feature = proteins_feature.to(device)
            drugs_feature = drugs_feature.to(device)
            indexes = pairs_indexes[1]
            # pairs = [item.detach().numpy() for item in pairs]
            indexes = [item.detach().numpy() for item in indexes]
            indexes = torch.Tensor(np.array(indexes)).to(device)
            # pairs_indexes_list = [pairs, indexes]
            # pairs_indexes_list_tensor = torch.Tensor(pairs_indexes_list)
            # inputs = pairs_indexes_list_tensor.to(device)
            labels = labels.long().to(device)

            outputs = model(proteins_feature, drugs_feature, indexes)
            outputs_class = np.argmax(outputs.cpu().detach().numpy(), axis=1)
            # 计算测试的时候的auc
            auc_value = roc_auc_score(labels.cpu().detach().numpy(), outputs_class)
            # f1
            f1_valid = f1_score(labels.cpu().detach().numpy(), outputs_class)


            loss = loss_f(outputs.cpu(), labels.cpu())

            true_list = []
            # 统计混淆矩阵
            _, predicted = torch.max(outputs.data, 1)
            for j in range(len(labels)):
                cate_i = labels[j].cpu().numpy()
                pre_i = predicted[j].detach().cpu().numpy()
                #conf_mat[cate_i, pre_i] += 1.
                if cate_i == pre_i:
                    true_list.append(j)

                if cate_i != pre_i:
                    path_error.append((cate_i, pre_i))  # 记录错误样本的信息

            # 统计loss
            loss_sigma.append(loss.item())

            acc_avg = len(true_list) / len(labels)

        return np.mean(loss_sigma), acc_avg, path_error, auc_value, f1_valid
