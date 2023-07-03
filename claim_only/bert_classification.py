import pandas as pd
import numpy as np
import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader, RandomSampler
from datetime import datetime
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import transformers
from transformers import AdamW, AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import warnings
import time
import re
import wandb
import random
import argparse
import logging
import json
import pickle
from transformers import logging as lg

lg.set_verbosity_error()
warnings.filterwarnings("ignore")
torch.manual_seed(555)


class WarmupLinearScheduler(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, warmup_steps, scheduler_steps, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.scheduler_steps = scheduler_steps
        super(WarmupLinearScheduler, self).__init__(
            optimizer, self.lr_lambda, last_epoch=last_epoch
        )

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(
            0.0, float(self.scheduler_steps - step) / float(max(1, self.scheduler_steps - self.warmup_steps))
        )


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', required=True, type=str)
    parser.add_argument('--train_data_path', required=True, type=str)
    parser.add_argument('--valid_data_path', required=True, type=str)

    parser.add_argument('--model_name', default="bert-base-uncased", type=str)
    parser.add_argument('--load_model_path', default=None, type=str)
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--eval_batch_size', default=8, type=int)
    parser.add_argument('--max_eval_sample', default=1000, type=int)

    parser.add_argument('--total_step', default=1000000, type=int)
    parser.add_argument('--total_epoch', default=None, type=int)
    parser.add_argument('--warmup_steps', default=0, type=int)
    parser.add_argument('--scheduler_steps', default=None, type=int)
    parser.add_argument('--accumulation_steps', default=1, type=int)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')  
    parser.add_argument('--clip', type=float, default=None, help='gradient clipping')
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--scheduler', type=str, default='fixed')
    parser.add_argument('--weight_decay', type=float, default=0.1)

    parser.add_argument('--report_every_step', default=10, type=int)
    parser.add_argument('--save_every_step', default=500, type=int)
    parser.add_argument('--eval_every_step', default=100, type=int)

    parser.add_argument('--early_stopping_patience', default=np.inf, type=int)
    parser.add_argument('--patience', default=10, type=int)
        
    args = parser.parse_args()
    return args

class CompDataset(Dataset):

    def __init__(self, df, tokenizer):
        self.pickle = df
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        
        all_list = list(self.pickle)
        sentence = all_list[index]
        label = self.pickle[sentence]['Label'][0]

        if label == True:
            target = 1
        else:
            target = 0

        encoded_dict = self.tokenizer.encode_plus(
                    sentence,
                    add_special_tokens = True,      
                    max_length = 128,           
                    pad_to_max_length = True,
                    truncation=True,
                    return_attention_mask = True,   
                    return_tensors = 'pt',          
               )
        
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        
        sample = (padded_token_list, att_mask, target)

        return sample

    def __len__(self):
        return len(list(self.pickle))

def set_optim(args,model):
    if args.optim == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)       
    if args.scheduler == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=args.patience)
    elif args.scheduler == 'linear':
        if args.scheduler_steps is None:
            scheduler_steps = args.total_step
        else:
            scheduler_steps = args.scheduler_steps
        scheduler = WarmupLinearScheduler(optimizer, warmup_steps=args.warmup_steps, scheduler_steps=scheduler_steps)
    return optimizer, scheduler


def main(args):

    exp_name = args.exp_name
    model_name = args.model_name    
    train_data_path = args.train_data_path
    valid_data_path = args.valid_data_path

    # define logger
    directory = f'exp_{exp_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    if 'bionlp' in  model_name and not os.path.exists(directory+'/'+'bionlp'):
        os.makedirs(directory+'/'+'bionlp')

    model_directory = model_name.split('/')[0].split('-')[0]
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    file_name = datetime.fromtimestamp(time.time())
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    datefmt = '%Y-%m-%d %H:%M:%S'
    format = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt)
    file_handler = logging.FileHandler(os.path.join(directory, model_name+'_'+str(file_name.strftime(datefmt)).replace(':','.')+'.log'))
    file_handler.setFormatter(format)
    logger.addHandler(file_handler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log = f"\nExperiment name: {exp_name}\n" + \
          f"Current device: {device}\n" + \
          f"Model name: {model_name}\n" 
    logger.info(log)

    tokenizer = AutoTokenizer.from_pretrained(model_name, do_lower_case=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels = 2).to(device)

    if args.load_model_path:
        model.load_state_dict(torch.load(args.load_model_path))
        logger.info(f'Model loaded from {args.load_model_path}')

    optimizer, scheduler = set_optim(args, model)

    with open(train_data_path, 'rb') as pickle_file:
        train_pickle = pickle.load(pickle_file)
    with open(valid_data_path, 'rb') as pickle_file:
        valid_pickle = pickle.load(pickle_file)

    train_data = CompDataset(train_pickle, tokenizer)
    val_data = CompDataset(valid_pickle, tokenizer)

    train_dataloader = DataLoader(train_data,
                                batch_size=args.batch_size,
                                shuffle=True,
                                num_workers=8)

    val_dataloader = DataLoader(val_data,
                            batch_size=args.eval_batch_size,
                            shuffle=False,
                            num_workers=8)

    logger.info(f"loaded {len(train_dataloader)} training examples from {train_data_path}")
    logger.info(f"loaded {len(val_dataloader)} development examples from {valid_data_path}")

    loss_fn = torch.nn.CrossEntropyLoss()

    step = 0
    for epoch in range(args.total_epoch):
        
        targets_list = []

        # ========================================
        #               Training
        # ========================================
        
        print('Training...')

        model.train() 
        torch.set_grad_enabled(True)

        total_train_loss = 0

        for i, batch in enumerate(train_dataloader):
            step += 1
            train_status = 'Batch ' + str(i) + ' of ' + str(len(train_dataloader))
            
            print(train_status, end='\r')

            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            model.zero_grad()        

            outputs = model(b_input_ids, 
                        attention_mask=b_input_mask)

            loss = loss_fn(outputs[0], b_labels)
    
            total_train_loss = total_train_loss + loss.item()
        
            optimizer.zero_grad()
    
            loss.backward()
            train_loss = round(loss.item()*args.accumulation_steps, 8)
    
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step() 
    
            #print('Train loss:' ,total_train_loss)
            if step % args.report_every_step==0:
                lr = optimizer.param_groups[0]['lr']
                log = f'Epoch: {epoch} | Step {step} | Training Loss: {train_loss} | Learning rate: {lr}'
                logger.info(log)

            if step % 100 == 0:
                print('Evaluation...')
                model.eval()
                torch.set_grad_enabled(False)

                for i, tar_dataloader in enumerate([val_dataloader]):
                    
                    acc_name = ['total accuracy'][i]
                    total_val_loss = 0
                    targets_list = []
                    for j, batch in enumerate(tar_dataloader):
                        
                        val_status = 'Batch ' + str(j) + ' of ' + str(len(val_dataloader))
                        
                        print(val_status, end='\r')

                        b_input_ids = batch[0].to(device)
                        b_input_mask = batch[1].to(device)
                        b_labels = batch[2].to(device)      
                        outputs = model(b_input_ids, attention_mask=b_input_mask)
                        preds = outputs[0]

                        loss = loss_fn(preds, b_labels)
                        total_val_loss = total_val_loss + loss.item()
                    
                        val_preds = preds.detach().cpu().numpy()
                        
                        targets_np = b_labels.to('cpu').numpy()

                        targets_list.extend(targets_np)

                        if j == 0: 
                            stacked_val_preds = val_preds
                        else:
                            stacked_val_preds = np.vstack((stacked_val_preds, val_preds))
                    
                    y_true = targets_list
                    y_pred = np.argmax(stacked_val_preds, axis=1)
                    
                    val_acc = accuracy_score(y_true, y_pred)
                    print('Validation accuracy: ', val_acc*100)

                model.train()
                torch.set_grad_enabled(True)
        
        if not os.path.exists(f'./{model_directory}/checkpoint-{epoch}/'):
            os.makedirs(f'./{model_directory}/checkpoint-{epoch}/')

        torch.save(model.state_dict(), f'./{model_directory}/checkpoint-{epoch}/pytorch_model.bin')
    
if __name__ == '__main__':
    args = define_argparser()
    main(args)