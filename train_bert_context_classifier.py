# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import logging
import argparse
import random
import pickle
import itertools
import time
import pprint
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter

import models
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from models.bert_modeling_inputoffsetemb import BertOffsetForSequenceClassification
from models.bert_modeling_posattention import BertPosattnForSequenceClassification
from models.bert_modeling_mask import BertMaskForSequenceClassification

from procedural_extraction.relation_preprocessor import RelationProcessor, inflate_examples

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, offset1, offset2=None):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.offset1 = offset1
        self.offset2 = offset2

def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, max_offset, ap_only=False):
    """Loads a data file into a list of `InputBatch`s."""
    examples = inflate_examples(examples, tokenizer, max_offset, ap_only)

    label_map = {label : i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a, tokens_b = example['tokens']
        offset1_a, offset1_b = example['offset1']
        offset2_a, offset2_b = example['offset2']
        label_id = label_map[example['label']]
        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)
        offset1 = [offset1_a[0]] + offset1_a[:len(tokens_a)] + [offset1_a[len(tokens_a)-1]]
        offset2 = [offset2_a[0]] + offset2_a[:len(tokens_a)] + [offset2_a[len(tokens_a)-1]]

        tokens += tokens_b + ["[SEP]"]
        segment_ids += [1] * (len(tokens_b) + 1)
        offset1 += offset1_b[:len(tokens_b)] + [offset1_b[len(tokens_b)-1]]
        offset2 += offset2_b[:len(tokens_b)] + [offset2_b[len(tokens_b)-1]]

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        offset1 += padding
        offset2 += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(offset1) == max_seq_length
        assert len(offset2) == max_seq_length

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example['guid']))
            logger.info("tokens: %s" % " ".join(
                    [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info(
                    "offset1: %s" % " ".join([str(x) for x in offset1]))
            #logger.info(
            #        "offset2: %s" % " ".join([str(x) for x in offset2]))
            logger.info("label: %s (id = %d)" % (example['label'], label_id))

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,
                              offset1=offset1,
                              offset2=offset2))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x/warmup
    return 1.0 - x

def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--task_name",
                        default='rel',
                        type=str,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default='ckpts',
                        type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--log_dir",
                    default='logs',
                    type=str,
                    help="The log directory where the tensorboard log will be written.")
    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        default=False,
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval_on_train",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set when training.")
    parser.add_argument("--do_eval",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=12,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--max_offset',
                        type=int,
                        default=10,
                        help="Maximum offset size")
    parser.add_argument('--offset_emb',
                        type=int,
                        default=30,
                        help="offset embedding dimension")
    parser.add_argument('--comment',
                        default='',
                        help='string to show in tensorboard name')
    parser.add_argument('--offset_fusion',
                        default='none',
                        choices=['postattn', 'segemb', 'none', 'mask'],
                        help="ways to infuse offset embedding")

    args = parser.parse_args()

    processors = {
        "rel": RelationProcessor,
    }

    num_labels_task = {
        "rel": 3,
    }


    device = torch.device("cuda")
    n_gpu = torch.cuda.device_count()

    logger.info("device: {} n_gpu: {}".format(
        device, n_gpu))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    def dir_check(*dir_path, ontrain):
        dir_path = os.path.join(*dir_path)
        if os.path.exists(dir_path) and os.listdir(dir_path) and ontrain:
            raise ValueError("Directory ({}) already exists and is not empty.".format(dir_path))
        if not ontrain and (not os.path.exists(dir_path) or not os.listdir(dir_path)):
            raise ValueError("Directory ({}) is empty.".format(dir_path))
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    outputdir = dir_check(args.output_dir, args.comment, ontrain=args.do_train)
    logdir = dir_check(args.log_dir, args.comment, ontrain=True)
    with open(os.path.join(outputdir, 'hyperparas.txt'), 'w') as f:
        pprint.pprint(vars(args), f, width=1)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    num_labels = num_labels_task[task_name]
    label_list = processor.get_labels()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_steps = None
    if args.do_train:
        len_train_examples = len(processor.get_train_examples(args.data_dir)) 
        num_train_steps = int(
            len_train_examples / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)

    # Prepare model
    if args.offset_fusion == 'postattn':
        model = BertPosattnForSequenceClassification.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                num_labels = num_labels, max_offset=args.max_offset, offset_emb=args.offset_emb)
    elif args.offset_fusion == 'segemb':
        model = BertOffsetForSequenceClassification.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                num_labels = num_labels, max_offset=args.max_offset)
    elif args.offset_fusion == 'mask':
        model = BertMaskForSequenceClassification.from_pretrained(args.bert_model,
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                num_labels = num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained(args.bert_model, 
                cache_dir=PYTORCH_PRETRAINED_BERT_CACHE,
                num_labels = num_labels)
    model.to(device)
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    t_total = num_train_steps

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_steps)

    # Prepare data
    def get_dataloader(prefix):
        if prefix == 'train':
            eval_examples = processor.get_train_examples(args.data_dir)
        elif prefix == 'eval':
            eval_examples = processor.get_dev_examples(args.data_dir)
        elif prefix == 'test':
            eval_examples = processor.get_test_examples(args.data_dir)
        eval_features = convert_examples_to_features(
            eval_examples, label_list, args.max_seq_length, tokenizer, args.max_offset)
        logger.info("***** %s dataset *****" % prefix)
        logger.info("  Num examples = %d", len(eval_examples))
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        all_offset1s = torch.tensor([f.offset1 for f in eval_features], dtype=torch.long)
        all_offset2s = torch.tensor([f.offset2 for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids, all_offset1s,
                                  all_offset2s)
        # Run prediction for full data
        if prefix == 'train':
            eval_sampler = RandomSampler(eval_data)
            bsz = args.train_batch_size
        else:
            eval_sampler = SequentialSampler(eval_data)
            bsz = args.eval_batch_size
        return DataLoader(eval_data, sampler=eval_sampler, batch_size=bsz)

    if args.do_train:
        train_dataloader = get_dataloader('train')
    if args.do_eval_on_train:
        eval_dataloader = get_dataloader('eval')
    if args.do_eval:
        test_dataloader = get_dataloader('test')

    # Execution
    writer = SummaryWriter(logdir)
    global_step = 0
    best_micro_f1 = -1
    if args.do_train:
        epoch_steps = int(len_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
        tr_loss = []
    def train_epoch():
        nonlocal global_step
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, offset1s, offset2s = batch
            if args.offset_fusion != 'none':
                loss, logits = model(input_ids, offset1s, offset2s, segment_ids, input_mask, label_ids)
            else:
                loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss.append(loss.item())
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step/t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1
                maxsteps = min(len(tr_loss), epoch_steps)
                writer.add_scalar('train/loss', sum(tr_loss[-maxsteps:]) / maxsteps, global_step)
                writer.add_scalar('train/lr', lr_this_step, global_step)


    def eval(prefix, eval_dataloader):
        nonlocal best_micro_f1
        model.eval()
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        label_accuracy, nb_label_examples, nb_label_predicted = [0 for _ in label_list], [0 for _ in label_list], [0 for _ in label_list]
        for _, batch in enumerate(eval_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, offset1s, offset2s = batch
            with torch.no_grad():
                if args.offset_fusion != 'none':
                    eval_loss, logits = model(input_ids, offset1s, offset2s, segment_ids, input_mask, label_ids)
                else:
                    logits = model(input_ids, segment_ids, input_mask)
                    eval_loss = model(input_ids, segment_ids, input_mask, label_ids)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            tmp_eval_accuracy = np.sum(outputs == label_ids)

            for label in list(range(num_labels)):
                label_accuracy[label] += np.sum(np.logical_and(outputs == label_ids, label_ids == label))
                nb_label_examples[label] += np.sum(label_ids == label)
                nb_label_predicted[label] += np.sum(outputs == label)

            eval_loss += eval_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples

        writer.add_scalar(prefix+'/dev_loss', eval_loss, global_step)
        writer.add_scalar(prefix+'/accuracy', eval_accuracy, global_step)
        for (idx, label_accu) in enumerate(label_accuracy):
            recall = label_accu / nb_label_examples[idx]
            precision = label_accu / nb_label_predicted[idx]
            f1 = (2 * precision * recall) / (precision + recall)

            writer.add_scalar(prefix+'/recall_label_' + str(idx), recall, global_step)
            writer.add_scalar(prefix+'/precision_label' + str(idx), precision, global_step)
            writer.add_scalar(prefix+'/F1_label' + str(idx), f1, global_step)
        micro_precision = sum(label_accuracy[1:]) / sum(nb_label_predicted[1:])
        micro_recall = sum(label_accuracy[1:]) / sum(nb_label_examples[1:])
        micro_f1 = (2 * micro_precision * micro_recall) / (micro_precision + micro_recall)
        if micro_f1 > best_micro_f1:
            best_micro_f1 = micro_f1
            # Save a trained model
            model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-sel
            torch.save(model_to_save.state_dict(), output_model_file)

        writer.add_scalar(prefix+'/precision_micro_avg', micro_precision, global_step)
        writer.add_scalar(prefix+'/recall_micro_avg', micro_recall, global_step)
        writer.add_scalar(prefix+'/F1_micro_avg', micro_f1, global_step)
        writer.add_scalar(prefix+'/best_F1_micro_avg', best_micro_f1, global_step)

    output_model_file = os.path.join(outputdir, "best_model.bin")
    if args.do_train:
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            train_epoch()
            if args.do_eval_on_train:
                eval('eval', eval_dataloader)

    if args.do_eval:
        # Load a trained model that you have fine-tuned
        model_state_dict = torch.load(output_model_file)
        if args.offset_fusion == 'postattn':
            model = BertPosattnForSequenceClassification.from_pretrained(args.bert_model,
                    state_dict=model_state_dict,
                    num_labels = num_labels, max_offset=args.max_offset, offset_emb=args.offset_emb)
        elif args.offset_fusion == 'segemb':
            model = BertOffsetForSequenceClassification.from_pretrained(args.bert_model,
                    state_dict=model_state_dict,
                    num_labels = num_labels, max_offset=args.max_offset)
        elif args.offset_fusion == 'mask':
            model = BertMaskForSequenceClassification.from_pretrained(args.bert_model,
                    state_dict=model_state_dict,
                    num_labels = num_labels)
        else:
            model = BertForSequenceClassification.from_pretrained(args.bert_model, 
                    state_dict=model_state_dict,
                    num_labels = num_labels)
        model.to(device)

        eval('test', test_dataloader)

    writer.export_scalars_to_json(logdir + '.json')
    writer.close()


if __name__ == "__main__":
    main()
