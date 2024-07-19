# -*- encoding: utf-8 -*-
'''
@File        :   model.py
@Time        :   2024/06/04 15:33:56
@Author      :   Feng zhixin 
@Description :   用于分类的bert模型
'''

# here put the import lib
import torch
import transformers

class BERTClass(torch.nn.Module):
    def __init__(self, num_labels, config):
        super(BERTClass, self).__init__()
        self.l1 = transformers.BertModel.from_pretrained('bert-base-uncased', config=config, ignore_mismatched_sizes=True)
        # self.l2 = torch.nn.Dropout(0.3)
        self.l3 = torch.nn.Linear(768, num_labels)
    
    def forward(self, ids, mask, token_type_ids):
        # import pdb;pdb.set_trace()
        
        _, output_1= self.l1(ids, attention_mask=mask, token_type_ids=token_type_ids, return_dict=False)   # torch.Size([bsz, 128, 768]), torch.Size([bsz, 768])

        # output_2 = self.l2(output_1)
        # output = self.l3(output_2)
        output = self.l3(output_1)
        return {'output':output, 'output_1':output_1}
