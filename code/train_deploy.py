import argparse
import json
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader, RandomSampler, TensorDataset
from transformers import AdamW, BertForSequenceClassification, BertTokenizer

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))

MAX_LEN = 64  # this is the max length of the sentence

print("Loading BERT tokenizer...")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def _get_train_data_loader(batch_size, training_dir, is_distributed):
    logger.info("Get train data loader")

    dataset = pd.read_csv(os.path.join(training_dir, "train.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
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
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    if is_distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    else:
        train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    return train_dataloader


def _get_test_data_loader(test_batch_size, training_dir):
    dataset = pd.read_csv(os.path.join(training_dir, "test.csv"))
    sentences = dataset.sentence.values
    labels = dataset.label.values

    input_ids = []
    for sent in sentences:
        encoded_sent = tokenizer.encode(sent, add_special_tokens=True)
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
    train_labels = torch.tensor(labels)
    train_masks = torch.tensor(attention_masks)

    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=test_batch_size)

    return train_dataloader


def train(args):
    is_distributed = len(args.hosts) > 1 and args.backend is not None
    logger.debug("Distributed training - %s", is_distributed)
    use_cuda = args.num_gpus > 0
    logger.debug("Number of gpus available - %d", args.num_gpus)
    device = torch.device("cuda" if use_cuda else "cpu")

    if is_distributed:
        # Initialize the distributed environment.
        world_size = len(args.hosts)
        os.environ["WORLD_SIZE"] = str(world_size)
        host_rank = args.hosts.index(args.current_host)
        os.environ["RANK"] = str(host_rank)
        dist.init_process_group(backend=args.backend, rank=host_rank, world_size=world_size)
        logger.info(
            "Initialized the distributed environment: '%s' backend on %d nodes. "
            "Current host rank is %d. Number of gpus: %d",
            args.backend, dist.get_world_size(),
            dist.get_rank(), args.num_gpus
        )

    # set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed(args.seed)

    train_loader = _get_train_data_loader(args.batch_size, args.data_dir, is_distributed)
    test_loader = _get_test_data_loader(args.test_batch_size, args.test)

    logger.debug(
        "Processes {}/{} ({:.0f}%) of train data".format(
            len(train_loader.sampler),
            len(train_loader.dataset),
            100.0 * len(train_loader.sampler) / len(train_loader.dataset),
        )
    )

    logger.debug(
        "Processes {}/{} ({:.0f}%) of test data".format(
            len(test_loader.sampler),
            len(test_loader.dataset),
            100.0 * len(test_loader.sampler) / len(test_loader.dataset),
        )
    )

    logger.info("Starting BertForSequenceClassification\n")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",  # Use the 12-layer BERT model, with an uncased vocab.
        num_labels=args.num_labels,  # The number of output labels--2 for binary classification.
        output_attentions=False,  # Whether the model returns attentions weights.
        output_hidden_states=False,  # Whether the model returns all hidden-states.
    )

    model = model.to(device)
    if is_distributed and use_cuda:
        # multi-machine multi-gpu case
        model = torch.nn.parallel.DistributedDataParallel(model)
    else:
        # single-machine multi-gpu case or single-machine or multi-machine cpu case
        model = torch.nn.DataParallel(model)
    optimizer = AdamW(
        model.parameters(),
        lr=2e-5,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
        eps=1e-8,  # args.adam_epsilon - default is 1e-8.
    )

    logger.info("End of defining BertForSequenceClassification\n")
    for epoch in range(1, args.epochs + 1):
        total_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)
            model.zero_grad()

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            loss = outputs[0]

            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            if step % args.log_interval == 0:
                logger.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}".format(
                        epoch,
                        step * len(batch[0]),
                        len(train_loader.sampler),
                        100.0 * step / len(train_loader),
                        loss.item(),
                    )
                )

        logger.info("Average training loss: %f\n", total_loss / len(train_loader))

        test(model, test_loader, device)

    logger.info("Saving tuned model.")
    model_2_save = model.module if hasattr(model, "module") else model
    model_2_save.save_pretrained(save_directory=args.model_dir)


def test(model, test_loader, device):
    model.eval()
    _, eval_accuracy = 0, 0

    with torch.no_grad():
        for batch in test_loader:
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            outputs = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)
            logits = outputs[0]
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to("cpu").numpy()
            tmp_eval_accuracy = flat_accuracy(logits, label_ids)
            eval_accuracy += tmp_eval_accuracy

    logger.info("Test set: Accuracy: %f\n", tmp_eval_accuracy)


def model_fn(model_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("================ objects in model_dir ===================")
    print(os.listdir(model_dir))
    model = BertForSequenceClassification.from_pretrained(model_dir)
    print("================ model loaded ===========================")
    return model.to(device)




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
    print("============== encoded data =================")
    print(input_id, input_mask)
    with torch.no_grad():
        y = model(input_id, attention_mask=input_mask)[0]
        print("=============== inference result =================")
        print(y)
    return y

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument(
        "--num_labels", type=int, default=2, metavar="N", help="input batch size for training (default: 64)"
    )

    parser.add_argument(
        "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--test-batch-size", type=int, default=1000, metavar="N", help="input batch size for testing (default: 1000)"
    )
    parser.add_argument("--epochs", type=int, default=2, metavar="N", help="number of epochs to train (default: 10)")
    parser.add_argument("--lr", type=float, default=0.01, metavar="LR", help="learning rate (default: 0.01)")
    parser.add_argument("--momentum", type=float, default=0.5, metavar="M", help="SGD momentum (default: 0.5)")
    parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="backend for distributed training (tcp, gloo on cpu and gloo, nccl on gpu)",
    )

    # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--test", type=str, default=os.environ["SM_CHANNEL_TESTING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])

    train(parser.parse_args())
