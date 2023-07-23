import torch
import argparse
import pickle as pkl
from itertools import chain
from dataclasses import dataclass
from tqdm import trange, tqdm
from random import choices
import torch.nn as nn
from termcolor import colored
from transformers import BertModel, AutoModel, AutoTokenizer, PreTrainedTokenizerBase, AutoConfig, AutoModel
from preprocess import prepare_input
import os


parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, help='')
parser.add_argument('--kg_path', type=str, help='')
parser.add_argument('--lr', default=5e-5, type=float, help='')
parser.add_argument('--model_cls', default="cat", type=str, help='')
parser.add_argument('--epoch', default=10, type=int, help='')
# parser.add_argument('--db_name', default="", type=str, help='')
parser.add_argument('--n_candid', default="3", type=str, help='')
parser.add_argument('--scratch', action='store_true', help='')
args = parser.parse_args()

torch.manual_seed(42)

PT_CLS = "bert-base-cased"

class Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        split: str,
        claims: list,
        evis: list,
        labels: list, 
        types: list = None,
    ):
        super().__init__()
        
        self.claims = claims
        self.labels = labels
        self.split = split
        self.evis = evis
        self.types = types
        
        assert len(self.evis) == len(self.claims)
        assert len(self.evis) == len(self.labels)
        if self.types is not None:
            assert len(self.evis) == len(self.types)
        
    def __len__(self):
        return len(self.evis)
    
    def __getitem__(self, i):

        if self.split == "test":
            if "negation" in self.types[i]:
                rtype = 4
            elif "num1" in self.types[i]:
                rtype = 0
            elif "multi hop" in self.types[i]:
                rtype = 1
            elif "multi claim" in self.types[i]:
                rtype = 2
            elif "existence" in self.types[i]:
                rtype = 3
            else:
                raise ValueError()

            sample = {
                "e":(" | ".join([",".join(c) for c in self.evis[i][0]]), " | ".join([",".join(c) for c in self.evis[i][1]])),
                "c":self.claims[i],
                "l":self.labels[i],
                "type":rtype,
            }
        else:
            sample = {
                "e":(" | ".join([",".join(c) for c in self.evis[i][0]]), " | ".join([",".join(c) for c in self.evis[i][1]])),
                "c":self.claims[i],
                "l":self.labels[i],
            }
        
        return sample
    
@dataclass
class DataCollator:
    split: str
    tokenizer: PreTrainedTokenizerBase

    def tensorize(self, batch):
        for k, v in batch.items():
            if isinstance(v, list):
                o = torch.tensor(v).cuda()
            else:
                o = {_k:_v.cuda() for _k, _v in v.items()}
            batch[k] = o
            
        return batch

    def batchfy(self, features):
        keys = set(features[0].keys())
        batch = {k: [e[k] if k in e else None for e in features] for k in keys}

        claim = batch.pop("c")
        tokenized_claim = self.tokenizer(
                            claim,
                            padding="longest",
                            max_length=128,
                            truncation=True,
                            return_tensors="pt",
                        )
        
        evidence = batch.pop("e")
        seq_evidence = [f"{self.tokenizer.sep_token.join(evi)} {self.tokenizer.sep_token}" for evi in evidence]

        tokenized_evidence = self.tokenizer(
                            seq_evidence,
                            padding="longest",
                            max_length=512-len(tokenized_claim["input_ids"][0]),
                            truncation=True,
                            add_special_tokens=False,
                            return_tensors="pt",
                        )
        
        label = [int(x) for x in list(chain(*batch.pop("l")))]
            
        pt_batch = {
            "evidence": tokenized_evidence,
            "claim": tokenized_claim,
            "label": label,
        }

        if "type" in batch:
            pt_batch["type"] = batch["type"]

        return pt_batch

    def __call__(self, features):
        batch = self.batchfy(features)
        
        return self.tensorize(batch)


data_path = args.data_path
kg_path = args.kg_path

prepare_input(data_path, kg_path)

with open(os.path.join(data_path, 'factkg_train.pickle'), 'rb') as pkf:
    db = pkl.load(pkf)
    print(f"Load train DB, # samples: {len(db)}")

with open(f'./train_candid_paths.bin', 'rb') as pkf:
    candids = pkl.load(pkf)
    print(f"Load train candids, # samples: {len(candids)}")
    
train_claims = list()
train_evis = list()
train_labels = list() 

for i, (s, m) in enumerate(db.items()):
    train_claims.append(s)
    train_labels.append(m["Label"])
    evis = [candids[s]["connected"], candids[s]["walkable"]]
    train_evis.append(evis)

with open(os.path.join(data_path, 'factkg_dev.pickle'), 'rb') as pkf:
    db = pkl.load(pkf)
    print(f"Load dev DB, # samples: {len(db)}")

with open('./dev_candid_paths.bin', 'rb') as pkf:
    candids = pkl.load(pkf)
    print(f"Load dev candids, # samples: {len(candids)}")

dev_claims = list()
dev_evis = list()
dev_labels = list()

for i, (s, m) in enumerate(db.items()):
    dev_claims.append(s)
    dev_labels.append(m["Label"])
    evis = [candids[s]["connected"], candids[s]["walkable"]]
    dev_evis.append(evis)

with open(os.path.join(data_path, 'factkg_test.pickle'), 'rb') as pkf:
    db = pkl.load(pkf)
    print(f"Load Test DB, # samples: {len(db)}")

with open(f'./test_candid_paths_top{args.n_candid}.bin', 'rb') as pkf:
    candids = pkl.load(pkf)
    print(f"Load test candids, # samples: {len(candids)}")

test_claims = list()
test_evis = list()
test_labels = list()
test_types = list()
for i, (s, m) in enumerate(db.items()):
    test_claims.append(s)
    test_labels.append(m["Label"])
    evis = [candids[s]["connected"], candids[s]["walkable"]]
    test_evis.append(evis)
    test_types.append(m["types"])

tokenizer = AutoTokenizer.from_pretrained(PT_CLS)

train_dataset = Dataset("train", train_claims, train_evis, train_labels)
collator = DataCollator("train", tokenizer)
train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    shuffle=True, 
    batch_size=32, 
    collate_fn=collator, 
    drop_last=True,
    num_workers=0, 
    pin_memory=False
)

dev_dataset = Dataset("dev", dev_claims, dev_evis, dev_labels)
collator = DataCollator("dev", tokenizer)
dev_loader = torch.utils.data.DataLoader(
    dev_dataset, 
    shuffle=False, 
    batch_size=32, 
    collate_fn=collator, 
    drop_last=True,
    num_workers=0, 
    pin_memory=False
)

test_dataset = Dataset("test", test_claims, test_evis, test_labels, test_types)
collator = DataCollator("test", tokenizer)
test_loader = torch.utils.data.DataLoader(
    test_dataset, 
    shuffle=False, 
    batch_size=32, 
    collate_fn=collator, 
    drop_last=True,
    num_workers=0, 
    pin_memory=False
)

print(test_dataset[-1])

class ConcatClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(PT_CLS)
        self.shallow_classifier = nn.Sequential(
                                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.config.hidden_size, 2)
                                )
        if args.scratch:
            print("Random init models")
            self.encoder = BertModel(self.config)
        else:
            self.encoder = AutoModel.from_pretrained(PT_CLS)
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(
        self,
        inputs
    ):
        # process input
        cated_inputs = {k:torch.cat([inputs["claim"][k], inputs["evidence"][k]], dim=-1) for k in inputs["claim"]}
        encoder_outputs = self.encoder(
            **cated_inputs,
            return_dict=False
        )
        cls_output = encoder_outputs[0][:, 0]
        
        assert cls_output.shape[-1]==self.config.hidden_size
        
        logit = self.shallow_classifier(cls_output)
        loss = self.loss_fct(logit, inputs["label"])
        
        return loss, logit

class SentenceClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(PT_CLS)
        self.shallow_classifier = nn.Sequential(
                                    nn.Linear(self.config.hidden_size, self.config.hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(self.config.hidden_size, 2)
                                )
        if args.scratch:
            print("Random init models")
            self.encoder = BertModel(self.config)
        else:
            self.encoder = AutoModel.from_pretrained(PT_CLS)
        self.loss_fct = nn.CrossEntropyLoss()
    
    def forward(
        self,
        inputs
    ):
        # process input
        cated_inputs = inputs["claim"]
        encoder_outputs = self.encoder(
            **cated_inputs,
            return_dict=False
        )
        cls_output = encoder_outputs[0][:, 0]
        
        assert cls_output.shape[-1]==self.config.hidden_size
        
        logit = self.shallow_classifier(cls_output)
        loss = self.loss_fct(logit, inputs["label"])
        
        return loss, logit

model = {
    "sent":SentenceClassifier,
    "cat":ConcatClassifier,
}[args.model_cls]().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))
best = -1
stop_counter = 0

for epoch in range(args.epoch):
    model.train()
    losses = list()
    scores = list()
    for i, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train {epoch}", leave=False):
        optimizer.zero_grad()
        loss, logit = model(batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        pred = logit.max(dim=1).indices.bool()
        gt = batch["label"].bool()
        score = pred==gt
        scores.append(score.detach().cpu().squeeze())
        if i%(len(train_loader)//5) == 0:
            loss = f"{torch.Tensor(losses).mean().item():.5f}"
            accuracy = f"{torch.cat(scores).float().mean().item():.4f}"
            print(f"Epoch {colored(epoch, 'yellow')}, Loss: {colored(loss, 'yellow')}, Acc: {colored(accuracy, 'yellow')}")
            losses = list()
            scores = list()
            
    model.eval()
    with torch.no_grad():
        scores = list()
        gts = list()
        for i, batch in tqdm(enumerate(dev_loader), total=len(dev_loader), desc=f"Dev", leave=False):
            _, logit = model(batch)
            pred = logit.max(dim=1).indices.bool()
            gt = batch["label"].bool()
            score = pred==gt
            scores.append(score.cpu().squeeze())
            gts.append(gt.cpu().squeeze())
        accuracy = torch.cat(scores).float().mean().item()
        str_accuracy = f"{accuracy:.4f}"
        print(f"Dev Acc: {colored(str_accuracy, 'green')}")

    with open('./valid_pred.bin', "wb") as pkf:
        result = {
            "hit": [i for i, hit in enumerate(torch.cat(scores)) if hit],
            "label": torch.cat(gts)
        }
        pkl.dump(result, pkf)

    if best < accuracy:
        best = accuracy
        best_param = model.state_dict()
        stop_counter = 0
    else:
        stop_counter += 1
    if stop_counter > 3:
        break

model.load_state_dict(best_param)
model.eval()

with torch.no_grad():
    scores = list()
    rtypes = list()
    gts = list()
    for i, batch in tqdm(enumerate(test_loader), total=len(test_loader), desc=f"Test", leave=False):
        rtype = batch.pop("type")
        _, logit = model(batch)
        pred = logit.max(dim=1).indices.bool()
        gt = batch["label"].bool()
        score = pred==gt
        gts.append(gt.cpu().squeeze())
        scores.append(score.cpu().squeeze())
        rtypes.append(rtype)
    total_score = torch.cat(scores).float()
    total_rtype = torch.cat(rtypes)
    for rt in total_rtype.unique():
        idcs = total_rtype==rt
        print(f"-- # examples in {rt.item()}: {idcs.sum().item()} --")
        print(f"Acc for type {colored(str(rt.item()), 'yellow')}: {total_score[idcs.cpu()].mean().item():.4f}")

    accuracy = f"{torch.cat(scores).float().mean().item():.4f}"
    print(f"Total Test Acc: {colored(accuracy, 'green')}") 

with open('./test_pred.bin', "wb") as pkf:
    result = {
        "hit": [i for i, hit in enumerate(total_score.bool()) if hit],
        "label": torch.cat(gts)
    }
    pkl.dump(result, pkf)