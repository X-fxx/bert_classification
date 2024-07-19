# -*- encoding: utf-8 -*-
'''
@File        :   train.py
@Time        :   2024/06/04 12:05:39
@Author      :   Feng zhixin 
@Description :   BERT完成分类任务的训练代码
'''

# here put the import lib
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
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['CUDA_LAUNCH_BLOCKING']='1'


from opt import opt
import shutil
def train():
    args = opt()

    # 实例化tensorboard,并设置输出路径
    if args.log_dir:
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
    writer = SummaryWriter(log_dir=args.log_dir)

    # 复制一份opt文件在log_dir路径下
    current_file = os.path.abspath(__file__)
    if args.log_dir:
        shutil.copyfile("opt.py", os.path.join(args.log_dir, 'opt.py'))
        shutil.copy(current_file, os.path.join(args.log_dir, 'train.py'))

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

    # -------------------------------------------------
    train_dataset = Colab_ClassificationDataset(file_path=args.train_file, tokenizer=tokenizer, max_length=args.max_len)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)

    test_dataset = Colab_ClassificationDataset(file_path=args.test_file, tokenizer=tokenizer, max_length=args.max_len)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, drop_last=False)

    # -------------------------------------------------
    model = BERTClass(num_labels=args.num_labels, config=config)
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    test_acc = 0
    # -------------------------------------------------
    for epoch in tqdm(range(args.epochs)):
        # train
        model.train()
        train_acc = 0
        train_num = 0

        for data in tqdm(train_dataloader):
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
            labels = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask, token_type_ids)['output']
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # 计算准确率
            _, preds = torch.max(outputs, dim=1)
            train_num += torch.sum(preds == labels)

        train_acc = train_num / len(train_dataset)
        print("Epoch: {}, Acc: {}".format(epoch, train_acc))
        writer.add_scalar("Acc/train", train_acc, epoch)
        print("Epoch: {}, Train_Loss: {}".format(epoch, loss.item()))
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]['lr'], epoch)
        print("Epoch: {}, lr: {}".format(epoch, optimizer.param_groups[0]['lr']))

        # test
        # model.eval()
        with torch.no_grad():
            acc = 0
            num = 0
            for _, data in enumerate(test_dataloader, 0):
                ids = data['ids'].to(device, dtype = torch.long)
                mask = data['mask'].to(device, dtype = torch.long)
                token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
                labels = data['targets'].to(device, dtype = torch.long)
                outputs = model(ids, mask, token_type_ids)['output'] # 查看output的shape
                # 计算损失函数
                loss = loss_fn(outputs, labels)
                # 计算准确率
                _, preds = torch.max(outputs, dim=1)
                num += torch.sum(preds == labels)
            
            acc = num / len(test_dataset)
            print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
            print("Epoch: {}, Test_Acc: {}".format(epoch, acc))
            writer.add_scalar("Acc/test", acc, epoch)
            writer.add_scalar("Loss/test", loss.item(), epoch)

            if acc > test_acc:
                if not os.path.exists(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                if len(os.listdir(args.model_save_dir)) > 1:
                    # 删除这一个文件
                    file_name = os.listdir(args.model_save_dir)[0]
                    os.remove(os.path.join(args.model_save_dir, file_name))     
                torch.save(model.state_dict(), os.path.join(args.model_save_dir, "model_{}.pth".format(epoch)))
                test_acc = acc
            print("Epoch: {}, Acc: {}, Best_acc: {}".format(epoch, acc, test_acc))

            # 如果保存的模型超过5个，则删除最旧的3个
            # while len(os.listdir(args.model_save_dir)) > 5:
            #     old_model = os.path.join(args.model_save_dir, "model_{}.pth".format(epoch-5))
            #     if os.path.exists(old_model):
            #         os.remove(old_model)
               
if __name__ == "__main__":
    train()
