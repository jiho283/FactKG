import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast

    
class FactkGDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df_data = df
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sentence1 = self.df_data.loc[index, 'inputs']
        encoded_dict = self.tokenizer.encode_plus(
                    sentence1,
                    add_special_tokens = True,      
                    max_length = 256,           
                    pad_to_max_length = True,
                    truncation=True,
                    return_attention_mask = True,   
                    return_tensors = 'pt',          
               )
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]

        target_ = torch.tensor(self.df_data.loc[index, 'hop'])
        target = target_-1
        sample = (padded_token_list, att_mask, target)

        return sample

    def __len__(self):
        return len(self.df_data)
    

class FactKGTestDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.df_data = df
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        sentence1 = self.df_data.loc[index, 'inputs']

        encoded_dict = self.tokenizer.encode_plus(
                    sentence1, 
                    add_special_tokens = True,      
                    max_length = 256,           
                    pad_to_max_length = True,
                    return_attention_mask = True,   
                    truncation=True,
                    return_tensors = 'pt',          
               )
        
        padded_token_list = encoded_dict['input_ids'][0]
        att_mask = encoded_dict['attention_mask'][0]
        sample = (padded_token_list, att_mask, sentence1)  # Returning raw sentence here
        
        return sample

    def __len__(self):
        return len(self.df_data)

def create_datasets(train_path, dev_path, test_path,model_name):
    model_name = model_name
    tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=True)

    train_json = pd.read_json(train_path)
    dev_json = pd.read_json(dev_path)
    test_json = pd.read_json(test_path)

    train_data = FactkGDataset(train_json, tokenizer)
    val_data = FactkGDataset(dev_json, tokenizer)
    test_data = FactKGTestDataset(test_json, tokenizer)

    return train_data, val_data, test_data
