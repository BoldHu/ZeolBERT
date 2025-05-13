""" Trains a product extraction or role labeling model on a dataset.
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
import os
import wandb
#根据task不同选择实验操作抽取任务还是反应元素抽取任务

if __name__ == '__main__':
    if len(sys.argv) > 2:
        task = sys.argv[1]
        if task == "prod":
            from chemrxnextractor.prod_args import parse_train_args
            from chemrxnextractor.train import prod_train
            model_args, data_args, train_args = parse_train_args(sys.argv[2:])
            train_args.output_dir = train_args.output_dir + '/' + 'epoch'+str(train_args.num_train_epochs) + '/' + model_args.model_name_or_path.split('/')[-1]
            wandb_name = f"{task}_{model_args.model_name_or_path.split('/')[-1]}_epoch{train_args.num_train_epochs}"
            wandb.init(project='ChemIE', name=wandb_name)
            prod_train(model_args, data_args, train_args)
        elif task == "role":
            from chemrxnextractor.role_args import parse_train_args
            from chemrxnextractor.train import role_train
            # args = parse_train_args(sys.argv[2:])
            # role_train(*args)
            model_args, data_args, train_args = parse_train_args(sys.argv[2:])
            # train_args.output_dir = train_args.output_dir + '/' + 'epoch'+str(train_args.num_train_epochs) + '/' + model_args.model_name_or_path.split('/')[-1]
            train_args.output_dir = train_args.output_dir + '/' + model_args.model_name_or_path.split('/')[-1]+ '/' + 'wo_cls'
            #wandb_name = f"{task}_span_{model_args.model_name_or_path.split('/')[-1]}_epoch{train_args.num_train_epochs}"
            wandb_name = f"{task}_span_{model_args.model_name_or_path.split('/')[-1]}_wo_cls"
            wandb.init(project='ChemIE', name = wandb_name)
            role_train(model_args, data_args, train_args)
    else:
        print(f'Usage: {sys.argv[0]} [task] [options]', file=sys.stderr)

