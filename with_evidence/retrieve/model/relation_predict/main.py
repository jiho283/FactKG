import torch
import pytorch_lightning as pl
from model import FactKGRelationClassifier
from data import FactKGRelationDataModule
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pickle
import argparse
import yaml


def load_data_from_yaml(yaml_file_path):
    with open(yaml_file_path, 'r') as f:
        config = yaml.safe_load(f)
    data_path = config['data_path']
    relation_path = config['relation_path']
    model_name = config['model_name']
    eval_batch_size = config['eval_batch_size']
    train_batch_size = config['train_batch_size']
    max_input_length = config['max_input_length']
    max_epoch = config['max_epoch']
    top_k = config['top_k']

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(relation_path, 'rb') as f:
        relation_set = pickle.load(f)

    return {'data': data, 'relation_set': relation_set, 'model_name': model_name, 'eval_batch_size': eval_batch_size, 'train_batch_size': train_batch_size, 'max_input_length': max_input_length, 'max_epoch': max_epoch,'top_k':top_k}
def trainer(data,relation_set, model_name, eval_batch_size,train_batch_size,max_input_length,max_epoch,top_k):
    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, problem_type="multi_label_classification")
    relation_data_module  = FactKGRelationDataModule(relation_set, tokenizer, data, batch_size=train_batch_size, max_input_len=max_input_length)

    _model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(relation_set), problem_type="multi_label_classification")
    model = FactKGRelationClassifier(relation_set, _model,top_k, learning_rate=5e-5)

    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        max_epochs=max_epoch,
        gpus=1,
    )
    trainer.fit(model, relation_data_module) 
    

def evaluator(data,relation_set, model_name, eval_batch_size,train_batch_size,max_input_length,max_epoch,model_path,top_k):
    
    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name, problem_type="multi_label_classification")
    _model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(relation_set), problem_type="multi_label_classification")
    relation_data_module  = FactKGRelationDataModule(relation_set, tokenizer, data, batch_size=eval_batch_size, max_input_len=max_input_length)

    # Load the model from the checkpoint and pass in the required arguments
    model = FactKGRelationClassifier.load_from_checkpoint(
        model_path,
        relations=relation_set, 
        model=_model,
        top_k=top_k
    )

    # Test the model
    trainer = pl.Trainer(
        precision=16,
        accelerator="gpu",
        gpus=1,
    )
    trainer.test(model, datamodule=relation_data_module)

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, help="'train' for training the model and 'eval' for evaluation")
    parser.add_argument('--config', type=str, default='config.yaml', help="Path to the config yaml file")
    parser.add_argument('--model_path', type=str, required=False, help="Path to the saved model for evaluation")
    args = parser.parse_args()

    if args.mode == 'eval' and args.model_path is None:
        parser.error("--mode 'eval' requires --model_path")

    return args

if __name__ == '__main__':
    args = define_argparser()
    config_data = load_data_from_yaml(args.config)
    if args.mode == 'train':
        trainer(config_data['data'], config_data['relation_set'], config_data['model_name'], config_data['eval_batch_size'], config_data['train_batch_size'], config_data['max_input_length'], config_data['max_epoch'],config_data['top_k'])
    elif args.mode == 'eval':
        evaluator(config_data['data'], config_data['relation_set'], config_data['model_name'], config_data['eval_batch_size'], config_data['train_batch_size'], config_data['max_input_length'], config_data['max_epoch'], args.model_path,config_data['top_k'])
    else:
        raise ValueError("Invalid mode. Choose between 'train' or 'eval'") 
        