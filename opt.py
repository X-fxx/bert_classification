# -*- encoding: utf-8 -*-
'''
@File        :   opt.py
@Time        :   2024/06/04 11:51:26
@Author      :   Feng zhixin 
@Description :   构建参数
'''

# here put the import lib
import argparse
import time

def opt():
    parser = argparse.ArgumentParser(description=globals()["__doc__"])
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # bert
    parser.add_argument("--max_len", type=int, default=128, help="max length of input sequence")
    parser.add_argument("--vocab_file", type=str, default="/home/log_generation/4_BERT/myBert/processed_data/generateand_without_p4_p5/alarm_train_vocab_without.txt", help="vocab file")

    # data
    parser.add_argument("--train_file", type=str, default="/home/log_generation/4_BERT/myBert/processed_data/generateand_without_p4_p5/alarm_train_without.json", help="train data file")
    parser.add_argument("--test_file", type=str, default="/home/log_generation/4_BERT/myBert/processed_data/generateand_without_p4_p5/alarm_test_without.json", help="test data file")

    # train
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")
    parser.add_argument("--epochs", type=int, default=150, help="epochs")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate")

    # classification
    parser.add_argument("--num_labels", type=int, default=10, help="number of labels")

    # save
    now = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    parser.add_argument("--output_dir", type=str, default=f"./result/{now}_output", help="output directory")

    # test
    parser.add_argument("--test_model_path", type=str, default="/home/log_generation/4_BERT/myBert/result/06-22-13-19-53_output_1e-05_128_150/model_weight/model_67.pth", help="test model path")

    args = parser.parse_args()

    args.log_dir = args.output_dir + f"_{args.lr}_{args.batch_size}_{args.epochs}"
    args.model_save_dir = args.log_dir + f"/model_weight/"

    return args