# -*- encoding: utf-8 -*-
'''
@File        :   test.py
@Time        :   2024/06/17 16:35:57
@Author      :   Feng zhixin 
@Description :   加载模型参数，并进行各种指标的测试
'''

from sklearn import metrics
import os

import torch.utils
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
import torch
from transformers import BertTokenizer, BertModel, BertConfig
from model import BERTClass
from mydataset import Colab_ClassificationDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from opt import opt
from torch.utils.tensorboard import SummaryWriter
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING']='1'
from matplotlib import pyplot as plt


def test():
    args = opt()

    # 设置随机种子,确保结果可以重复
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = BertTokenizer(vocab_file=args.vocab_file, do_lower_case=False, do_basic_tokenize=False)
    tokenizer.model_max_length = args.max_len
    print("******** Tokenizer ********\n", tokenizer)
    # import pdb; pdb.set_trace()

    config = BertConfig()
    # config.update({"vocab_size":tokenizer.vocab_size+5}) 
    config.update({"vocab_size":len(tokenizer.vocab)+5})
    print("******** Config ********\n", config)

    test_dataset = Colab_ClassificationDataset(file_path=args.test_file, tokenizer=tokenizer, max_length=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    model = BERTClass(num_labels=args.num_labels, config=config)
    model_path = args.test_model_path
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()


    with torch.no_grad():
        final_targets = []
        final_outputs = []
        all_output = []
        acc = 0
        num = 0
        for _, data in enumerate(test_dataloader, 0):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            labels = data['targets'].to(device, dtype = torch.long)
            outputs = model(ids, mask, token_type_ids)['output'] # 查看output的shape
            # import pdb; pdb.set_trace()

            # 计算准确率
            _, preds = torch.max(outputs, dim=1)
            final_targets.extend(labels.cpu().detach().numpy().tolist())
            final_outputs.extend(preds.cpu().detach().numpy().tolist())
            all_output.extend(outputs.cpu().detach().numpy().tolist())

            num += torch.sum(preds == labels)
        

        # --------- 指标测算 ---------
        # Calculate accuracy
        accuracy = metrics.accuracy_score(final_targets, final_outputs)
        print(f"Accuracy: {accuracy}")

        # Calculate precision, recall, and F1 score
        precision = metrics.precision_score(final_targets, final_outputs, average='weighted')
        recall = metrics.recall_score(final_targets, final_outputs, average='weighted')
        f1 = metrics.f1_score(final_targets, final_outputs, average='weighted')

        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")

        # Calculate ROC curve and AUC for each class
        fpr = {}
        tpr = {}
        roc_auc = {}
        # import pdb; pdb.set_trace()
        n_classes = outputs.shape[1]

        final_targets_one_hot = np.eye(n_classes)[final_targets]  # Convert final_targets to one-hot encoding

        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(final_targets_one_hot[:, i], np.array(all_output)[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])

        # Plot ROC curve for each class
        plt.figure()
        for i in range(n_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        plt.savefig('roc_curve.png')

        # Calculate macro-average AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= n_classes
        macro_roc_auc = metrics.auc(all_fpr, mean_tpr)

        print(f"Macro-average AUC: {macro_roc_auc}")


        # import pandas as pd
        # # 将final_outputs和final_targets转换为DataFrame
        # df = pd.DataFrame({'predictions': final_outputs, 'targets': final_targets})
        # df.to_csv('predictions.csv', index=False)

        # # predictions = np.argmax(final_outputs, axis=1)
        # # final_targets = np.array(final_targets)

        # acc = num / len(test_dataset)
        # print("Acc: {}".format(acc))

        # # 将目标和预测值转换为 numpy 数组
        # final_targets = np.array(final_targets)
        # final_outputs = np.array(final_outputs)

        # # 找到唯一的类别
        # unique_labels = np.unique(final_targets)

        # # 创建一个 defaultdict，用于存储每个类别的预测分布
        # from collections import defaultdict
        # predictions_dict = defaultdict(lambda: np.zeros(len(unique_labels)))

        # # 填充 predictions_dict
        # for target, output in zip(final_targets, final_outputs):
        #     predictions_dict[target][output] += 1  # 将 output - 1 作为索引

        # # 绘制堆叠条形图
        # fig, ax = plt.subplots(figsize=(10, 6))

        # bar_width = 0.5
        # indices = np.arange(len(unique_labels))

        # bottoms = np.zeros(len(unique_labels))

        # for i, label in enumerate(unique_labels):
        #     counts = np.array([predictions_dict[l][i] for l in unique_labels])
        #     ax.bar(indices, counts, bar_width, bottom=bottoms, label=f'Predicted {label}')
        #     bottoms += counts

        # ax.set_xlabel('True Class')
        # ax.set_ylabel('Count')
        # ax.set_title('Stacked Bar Chart of Predictions for Each Class Label')
        # ax.set_xticks(indices)
        # ax.set_xticklabels(unique_labels)
        # ax.legend(title='Predicted Class', loc='upper right')
        # ax.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.savefig('histom.png')


if __name__ == '__main__':
    test()