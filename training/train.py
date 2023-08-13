from transformers import AutoModel, T5Config, T5ForConditionalGeneration, T5Tokenizer
from dataset import QGDataset 
import pandas as pd
from qg_trainer import Trainer
import torch


def get_tokenizer(checkpoint:str) -> T5Tokenizer:

    tokenizer = T5Tokenizer.from_pretrained(checkpoint)

    tokenizer.add_special_tokens(
        {'additional_special_tokens': ['<sep>']}
    )

    return tokenizer


def get_model(checkpoint:str,device: str,tokenizer:T5Tokenizer) -> T5ForConditionalGeneration:

    model = T5ForConditionalGeneration.from_pretrained(checkpoint) # will specify local path later  
    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)
    
    model.config.max_length = 512
    tokenizer.model_max_length = 512

    return model



if __name__ == "__main__":


    device = "cuda:0"

    qg_model = '../t5-small-bahasa-cased'
    max_length_source = 512
    max_length_target = 128
    pad_mask_id = 0
    save_dir = "./t5-small-bahasa-cased-question-generator"
    learning_rate = 1e-4
    dataloader_workers = 2
    epochs = 10
    train_batch_size = 5
    valid_batch_size = 10


    train = pd.read_json(r"/home/aisyahrzak/question-generation/data/train.jsonl", lines=True)
    dev = pd.read_json(r"/home/aisyahrzak/question-generation/data/dev.jsonl", lines=True)


    tokenizer = get_tokenizer(qg_model)
    
    train_set = QGDataset(train,tokenizer, pad_mask_id,512,128)
    valid_set = QGDataset(dev,tokenizer, pad_mask_id,512,128)


    model = get_model(qg_model,device, tokenizer)


    trainer = Trainer(
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        model=model,
        save_dir=save_dir,
        tokenizer=tokenizer,
        train_batch_size=train_batch_size,
        train_set=train_set,
        valid_batch_size=valid_batch_size,
        valid_set=valid_set
    )

    trainer.train()


   