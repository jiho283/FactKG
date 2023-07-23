import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, AdamW
from transformers import BertTokenizerFast
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import time
import numpy as np

class HopPredictorManager:
    def __init__(self, model_name, num_labels, learning_rate, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.tokenizer = BertTokenizerFast.from_pretrained(model_name, do_lower_case=False)
        self.model = self.model.to(device)
        self.optimizer = AdamW(self.model.parameters(),  lr=float(learning_rate))
        self.loss_fn = nn.CrossEntropyLoss()
        self.device = device

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def flat_accuracy(self, preds, labels):
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return accuracy_score(labels_flat, pred_flat)

    def train_epoch(self, dataloader, device):
        self.model.train()
        total_train_loss = 0
        start_time = time.time()

        for step, batch in enumerate(tqdm(dataloader)):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            self.model.zero_grad()
            outputs = self.model(b_input_ids, attention_mask=b_input_mask)
            loss = self.loss_fn(outputs[0], b_labels)
            total_train_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            if step % 50 == 0 and not step == 0:
                elapsed = time.time() - start_time
                print('  Batch {:>5,}  of  {:>5,}. Loss: {:.3f}. Elapsed: {:.2f}'.format(step, len(dataloader), loss.item(), elapsed))

        avg_train_loss = total_train_loss / len(dataloader)
        return avg_train_loss

    def evaluate(self, dataloader, device):
        self.model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        for batch in tqdm(dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)

            loss = self.loss_fn(outputs[0], b_labels)
            total_eval_loss += loss.item()
            preds = outputs[0]
            total_eval_accuracy += self.flat_accuracy(preds.detach().cpu().numpy(), b_labels.to('cpu').numpy())

        avg_val_loss = total_eval_loss / len(dataloader)
        avg_val_accuracy = total_eval_accuracy / len(dataloader)
        return avg_val_loss, avg_val_accuracy

    def predict(self, dataloader, device):
        self.model.eval()
        predictions = []
        sentences = []  # This will store the original sentences

        for batch in tqdm(dataloader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            raw_sentences = batch[2]  # Get the raw sentences from the DataLoader

            # Extract only the part before '[sep]' from raw sentences
            for sentence in raw_sentences:
                claim = sentence.split('[sep]')[0].strip()  # Extract the claim
                sentences.append(claim)

            with torch.no_grad():
                outputs = self.model(b_input_ids, attention_mask=b_input_mask)
            preds = outputs[0]
            predictions.extend(torch.argmax(preds, dim=1).tolist())

        return sentences, predictions

    
def predict(self, dataloader, device):
    self.model.eval()
    predictions = []
    sentences = []  # This will store the original sentences

    for batch in tqdm(dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        raw_sentences = batch[2]  # Get the raw sentences from the DataLoader

        # Extract only the part before '[sep]' from raw sentences
        for sentence in raw_sentences:
            claim = sentence.split('[sep]')[0].strip()  # Extract the claim
            sentences.append(claim)

        with torch.no_grad():
            outputs = self.model(b_input_ids, attention_mask=b_input_mask)
        preds = outputs[0]
        predictions.extend(torch.argmax(preds, dim=1).tolist())

    return sentences, predictions





            
