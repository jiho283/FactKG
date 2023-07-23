from sklearn.preprocessing import MultiLabelBinarizer
import torch
import pytorch_lightning as pl
import pickle


class FactKGRelationDataModule(pl.LightningDataModule):
  def __init__(self, relations, tokenizer, data, batch_size=16, max_input_len=512):
    super().__init__()
    self.relations = relations
    self.tokenizer = tokenizer
    self.data = data
    self.batch_size = batch_size
    self.max_input_len = max_input_len
    self.mlb = MultiLabelBinarizer()
    self.mlb.classes = self.relations 
    
  def setup(self, stage):
    pass

  def train_dataloader(self):
    return torch.utils.data.DataLoader(
        self.data["train"],
        batch_size=self.batch_size,
        shuffle=True,
        collate_fn=self._collate_fn
    )

  def val_dataloader(self):
    return torch.utils.data.DataLoader(
        self.data["dev"],
        batch_size=self.batch_size * 2,
        shuffle=False,
        collate_fn=self._collate_fn
    )

  def test_dataloader(self):
    return torch.utils.data.DataLoader(
        self.data["test"],
        batch_size=self.batch_size * 2,
        shuffle=False,
        collate_fn=self._collate_fn_test
    )

  def _collate_fn_test(self, batch):
    input_texts = [x["inputs"] for x in batch]
    model_inputs = self.tokenizer(
        input_texts,
        max_length=self.max_input_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return model_inputs, input_texts 

  def _collate_fn(self, batch):
    input_texts = [x["inputs"] for x in batch]
    answers = [[_ans if _ans in self.relations else "Unknown" for _ans in x["relation"]] for x in batch]
    answers = self.mlb.fit_transform(answers)
    label_ids = torch.FloatTensor(answers)
    model_inputs = self.tokenizer(
        input_texts,
        max_length=self.max_input_len,
        padding=True,
        truncation=True,
        return_tensors="pt"
    )
    return model_inputs, label_ids  
