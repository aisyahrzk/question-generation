import itertools
import logging
from typing import Optional, Dict, Union
import torch
from transformers import(
    AutoModelForSeq2SeqLM, 
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)




class QuestionGenerator:

    def __init__(self):

        self.checkpoint = "t5-small-bahasa-cased-question-generator"
        self.tokenizer = "t5-small-bahasa-cased-question-generator"
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, use_fast=False)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.checkpoint)

    def generate(self,context:str):

        
        predictions = []

        inputs = self.prepare_inputs(context)


        input_length = inputs["input_ids"].shape[-1]


        result = self.model.generate( 
                input_ids= inputs['input_ids'], 
                attention_mask=inputs['attention_mask'],
                num_beams=4,
                max_length=128)


        prediction = self.tokenizer.decode(result[0],skip_special_token = True)

        # predictions = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in result]

        print(prediction)

        questions = prediction.split("<sep>")

        questions = [question.replace('<pad>', '').replace('</s>', '').strip() for question in questions if len(question.strip().replace('<pad>', '').replace('</s>', '')) > 1]

        questions = list(set(questions))


        return questions


    def prepare_inputs(self,context:str):

        source_text = f"generate questions: {context}"

        inputs = self.tokenize([source_text])

        return inputs

    
    def tokenize(self,input):


        inputs = self.tokenizer(
            input, 
            max_length=128,
            add_special_tokens=True,
            truncation=True,
            padding= False,
            return_tensors="pt"

        )

        return inputs

    