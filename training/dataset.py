import dataset
import random
import pandas as pd
import torch
from transformers import AutoTokenizer
from typing import Mapping, Tuple




class QGDataset():

    def __init__(self,dataframe, tokenizer, pad_mask_id, max_length_source, max_length_target):
        self.tokenizer = tokenizer
        self.max_length_source = max_length_source
        self.max_length_target = max_length_target
        self.pad_mask_id = pad_mask_id
        self.data = dataframe
        self.source_text = self.data["source_text"]
        self.target_text = self.data["target_text"]
        
    def __len__(self)-> int:
        return len(self.source_text)
    
    
    def __getitem__(self, idx:int) -> Mapping[str,torch.Tensor] :

        source_text = self.source_text[idx]
        target_text = self.target_text[idx]

        # Calculate the maximum length available for the input and target text
        max_length_source_with_special_tokens = self.max_length_source - self.tokenizer.num_special_tokens_to_add()
        max_length_target_with_special_tokens = self.max_length_target - self.tokenizer.num_special_tokens_to_add()

        # Tokenize source text
        source_text_tokenized = self.tokenizer(source_text, max_length=max_length_source_with_special_tokens,
                                            padding="max_length", truncation=True, add_special_tokens=True,
                                            return_tensors='pt')

        # Tokenize target text
        target_tokenized = self.tokenizer(target_text, max_length=max_length_target_with_special_tokens,
                                        padding="max_length", truncation=True, add_special_tokens=True,
                                        return_tensors='pt')

        # Ensure the labels do not exceed max_length_source
        labels = self._mask_label_padding(target_tokenized["input_ids"])

        return {
            "input_ids": source_text_tokenized["input_ids"].squeeze(),
            "attention_mask": source_text_tokenized["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }


    def _mask_label_padding(self, labels: torch.Tensor) -> torch.Tensor:
        labels[labels == self.tokenizer.pad_token_id] = self.pad_mask_id
        return labels


