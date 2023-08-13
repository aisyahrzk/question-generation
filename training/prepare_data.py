import json
import pandas as pd
from pandas import json_normalize
import numpy as np


hl_token = '[ANSS]'
he_token = '[ANSE]'
sep_token = '<sep>'

def _get_correct_alignement(context, answer):
    """ Some original examples in SQuAD have indices wrong by 1 or 2 character. We test and fix this here. """
    gold_text = answer['text']
    start_idx = answer['answer_start']
    end_idx = start_idx + len(gold_text)
    if context[start_idx:end_idx] == gold_text:
        return start_idx, end_idx       # When the gold label position is good
    elif context[start_idx-1:end_idx-1] == gold_text:
        return start_idx-1,  end_idx-1   # When the gold label is off by one character
    elif context[start_idx-2:end_idx-2] == gold_text:
        return start_idx-2, end_idx-2   # When the gold label is off by two character
    else:
        raise ValueError()


def process_qg_text(context, question, answer):

    answer_text = answer['text'].strip()
    start_pos, end_pos = _get_correct_alignement(context, answer)
    que_gen_input = f"generate question: {context[:start_pos]} {hl_token} {answer_text} {he_token} {context[end_pos:]}"

    que_gen_target = f"{question}"
    return {"source_text": que_gen_input, "target_text": que_gen_target, "task": "qg"}


def process_e2e_qg(paragraph):

    source_text = f"generate questions: {paragraph['context'].strip()}" +" </s>"
    questions = [qas['question'].strip() for qas in paragraph['qas']]
    target_text = f" {sep_token} ".join(questions)
    target_text = f"{target_text} {sep_token}"
    return {"source_text": source_text, "target_text": target_text, "task": "e2e_qg"}

    

if __name__ == "__main__":

    count = 0

    with open(r"/home/aisyahrzak/question-generation/data/ms-dev-2.0.json") as f:
        squad = json.load(f)


    with open("dev.jsonl","a") as f:
        for article in squad["data"]:
            title = article.get("title", "").strip()
            for paragraph in article["paragraphs"]:
                context = paragraph["context"].strip()
                
                json.dump(process_e2e_qg(paragraph), f)
                f.write("\n")
                count += 1

                # for qa in paragraph["qas"]:
                #     question = qa["question"].strip()
                #     id_ = qa["id"]
                #     answers = [answer["text"].strip() for answer in qa["answers"]]
                #     json.dump(process_qg_text(context, question, qa["answers"][0]), f)
                #     f.write("\n")
                #     count += 1