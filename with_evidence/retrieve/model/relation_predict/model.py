from transformers import AutoModelForSequenceClassification
import pytorch_lightning as pl
import torch
from functools import reduce
import numpy as np
import random
import pandas as pd



class FactKGRelationClassifier(pl.LightningModule):
  def __init__(self, relations, model, top_k, learning_rate=0.00001):
    super().__init__()
    self.relations = relations
    self.model = model
    self.top_k=top_k
    self.learning_rate = learning_rate


  def forward(self, batch):
    model_inputs, label_ids = batch
    outputs = self.model(**model_inputs, labels=label_ids)
    loss = outputs.loss
    logits = outputs.logits
    return loss, logits

  def training_step(self, batch, batch_idx):
    loss, logits = self.forward(batch)
    return {"loss": loss}
    
  def validation_step(self, batch, batch_idx):
    return self._evaluation_step(batch) 

  def validation_epoch_end(self, outputs):
    self._evaluation_epoch_end(outputs, phase='test')

  def test_step(self, batch, batch_idx):
    return self._evaluation_step_eval(batch) 

  def test_epoch_end(self, outputs):
    self._evaluation_epoch_end_eval(outputs, phase='test')

  def _evaluation_step(self, batch):
    loss, logits = self.forward(batch)
    _, label_ids = batch
    gts = [list(map(self.relations.__getitem__, label_id.nonzero(as_tuple=True)[0])) for label_id in label_ids]
    logits = torch.sigmoid(logits)
    pr_label_ids = torch.where(logits > 0.4, 1, 0)
    prs = [list(map(self.relations.__getitem__, label_id.nonzero(as_tuple=True)[0])) for label_id in pr_label_ids]

    return {
        "loss": loss.item(),
        "gts": gts,
        "prs": prs,
    }

  def _evaluation_epoch_end(self, outputs, phase=None):
    ave_loss = np.mean([x["loss"] for x in outputs])
    gts = reduce(lambda x,y: x + y, [x["gts"] for x in outputs], [])
    prs = reduce(lambda x,y: x + y, [x["prs"] for x in outputs], [])
    acc = sum([list(set(gt).difference(set(['Unknown']))) == pr for gt, pr in zip(gts, prs)]) / len(gts)
    
    target_ids = random.sample(range(len(gts)), 5)
    
    print("="*50)
    print(f"ave_loss: {ave_loss}")
    print(f"ACC: {acc}")
    if phase == "test":
      print("GT" + "-"*40)
      for target_id in target_ids:
        print(f"GT: {gts[target_id]}\t\t\tPR: {prs[target_id]}")

  def _evaluation_step_eval(self, batch):
    model_inputs, input_texts = batch
    outputs = self.model(**model_inputs)
    logits = outputs.logits
    pr_label_ids = torch.topk(logits, dim=-1, k=self.top_k).indices
    prs = [list(map(self.relations.__getitem__, pr_label_id)) for pr_label_id in pr_label_ids]
    sentences = [text.split('[sep]')[0] for text in input_texts]
    entities = [text.split('[sep]')[1] for text in input_texts]
    return {
        "logits": logits,
        "prs": prs,
        "sentences": sentences,
        "entities":entities
    }

  def _evaluation_epoch_end_eval(self, outputs, phase=None):
    logits = [x["logits"] for x in outputs]
    prs = reduce(lambda x,y: x + y, [x["prs"] for x in outputs], [])
    #import pdb; pdb.set_trace()
    sentences = reduce(lambda x,y: x + y, [x["sentences"] for x in outputs], [])
    #input_texts = reduce(lambda x,y: x + y, [x["sentences"] for x in outputs], [])
    entities = reduce(lambda x,y: x + y, [x["entities"] for x in outputs], [])
    df = pd.DataFrame({'claims': sentences, 'entity': entities, 'output': prs})
    df.to_json(f"test_relations_top{self.top_k}.json")

  def configure_optimizers(self):
      grouped_params = [
          {
              "params": list(filter(lambda p: p.requires_grad, self.parameters())),
              "lr": self.learning_rate,
          },
      ]

      optimizer = torch.optim.AdamW(
          grouped_params,
          lr=self.learning_rate, 
      )
      return {"optimizer": optimizer}
