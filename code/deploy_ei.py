import json
import logging
import os
import sys

import torch
import torch.utils.data
import torch.utils.data.distributed
from transformers import BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 64  # this is the max length of the sentence

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_model = torch.jit.load(os.path.join(model_dir, "traced_bert.pt"))
    return loaded_model.to(device)


def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        sentence = json.loads(request_body)

        input_ids = []
        encoded_sent = tokenizer.encode(sentence, add_special_tokens=True)
        input_ids.append(encoded_sent)

        # pad shorter sentences
        input_ids_padded = []
        for i in input_ids:
            while len(i) < MAX_LEN:
                i.append(0)
            input_ids_padded.append(i)
        input_ids = input_ids_padded

        # mask; 0: added, 1: otherwise
        attention_masks = []
        # For each sentence...
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)

        # convert to PyTorch data types.
        train_inputs = torch.tensor(input_ids)
        train_masks = torch.tensor(attention_masks)

        return train_inputs, train_masks

    raise ValueError("Unsupported content type: {}".format(request_content_type))


def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    
    with torch.no_grad():
        with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
            y = model(input_id, attention_mask=input_mask)[0]
        return y
