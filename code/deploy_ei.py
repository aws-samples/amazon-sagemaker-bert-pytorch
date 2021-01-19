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
    print("================== objects in model_dir =====================")
    print(os.listdir(model_dir))
    loaded_model = torch.jit.load(os.path.join(model_dir, "traced_bert.pt"))
    print("================== model loaded =============================")
    
    return loaded_model.to(device)



def input_fn(request_body, request_content_type):
    """An input_fn that loads a pickled tensor"""
    if request_content_type == "application/json":
        data = json.loads(request_body)
        print("================ input sentences ===============")
        print(data)
        
        if isinstance(data, str):
            data = [data]
        elif isinstance(data, list) and len(data) > 0 and isinstance(data[0], str):
            pass
        else:
            raise ValueError("Unsupported input type. Input type can be a string or an non-empty list. \
                             I got {}".format(data))
                       

        #encoded = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        #encoded = tokenizer(data, add_special_tokens=True) 
        
        # for backward compatibility use the following way to encode 
        # https://github.com/huggingface/transformers/issues/5580
        input_ids = [tokenizer.encode(x, add_special_tokens=True) for x in data]
        
        print("================ encoded sentences ==============")
        print(input_ids)

        # pad shorter sentence
        padded =  torch.zeros(len(input_ids), MAX_LEN) 
        for i, p in enumerate(input_ids):
            padded[i, :len(p)] = torch.tensor(p)
     
        # create mask
        mask = (padded != 0)
        
        print("================= padded input and attention mask ================")
        print(padded, '\n', mask)

        return padded.long(), mask.long()

    raise ValueError("Unsupported content type: {}".format(request_content_type))
    
    

def predict_fn(input_data, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    input_id, input_mask = input_data
    input_id = input_id.to(device)
    input_mask = input_mask.to(device)
    
    with torch.no_grad():
        try:
            with torch.jit.optimized_execution(True, {"target_device": "eia:0"}):
                print("==================== using elastic inference ====================")
                y = model(input_id, attention_mask=input_mask)[0]
        except TypeError:
            y = model(input_id, attention_mask=input_mask)[0]
        
    print("==================== inference result =======================")
    print(y)
    return y

          