# -*- encoding: utf-8 -*-
'''
@File        :   dataset.py
@Time        :   2024/06/04 11:17:36
@Author      :   Feng zhixin 
@Description :   build dataset for classification
'''

# here put the import lib
from torch.utils.data import Dataset
import os
import json
import torch

class ClassificationDataset(Dataset):
    def __init__(self, file_path):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        
        # 从json文件中读取数据
        self.data = []
        with open(file_path, 'r') as f:
           for row in f:
            self.data.append(json.loads(row))
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.data[index]['cluster']
        cmd_block = self.data[index]['cmd']
        return cmd_block, label

class Colab_ClassificationDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        if os.path.isfile(file_path) is False:
            raise ValueError(f"Input file path {file_path} not found")
        
        # 从json文件中读取数据
        self.data = []
        with open(file_path, 'r') as f:
           for row in f:
            self.data.append(json.loads(row))

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        label = self.data[index]['cluster'] # 对应的cluster_id
        cmd_block = self.data[index]['cmd']   # 取出的cmd_block

        # 给数据添加特殊字符

        # 使用[sep]将cmd_block进行拼接
        cmd_block = ' [SEP] '.join(cmd_block)

        # tokens = self.tokenizer.tokenize(cmd_block)

        cmd_inputs = self.tokenizer.encode_plus(cmd_block,
                                                None,
                                                add_special_tokens=True,
                                                max_length=self.max_length,
                                                pad_to_max_length=True,
                                                return_token_type_ids=True)
        
        ids = cmd_inputs['input_ids']
        mask = cmd_inputs['attention_mask']
        token_type_ids = cmd_inputs['token_type_ids']
        
        return {'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'targets': torch.tensor(label, dtype=torch.float) }


if __name__ == "__main__":
    # dataset = ClassificationDataset('processed_data/alarm_test.json')
    # for i in range(len(dataset)):
    #     print(dataset[i])

    # 验证Colab_ClassificationDataset
    from transformers import BertTokenizer
    tokenizer = BertTokenizer(vocab_file='/home/log_generation/4_BERT/myBert/processed_data/alarm_test_vocab.txt', do_lower_case=False, do_basic_tokenize=False)
    tokenizer.model_max_length = 128
    print("******** Tokenizer ********\n", tokenizer)

    dataset = Colab_ClassificationDataset('/home/log_generation/4_BERT/myBert/processed_data/alarm_test.json', tokenizer, 128)
    for i in range(len(dataset)):
        print(dataset[i])

