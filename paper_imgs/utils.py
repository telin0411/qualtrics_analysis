import os
import sys
import csv
import random
import json
import jsonlines
import tqdm
import pandas as pd
import numpy as np
import argparse
import glob
import pprint
from collections import Counter

import nltk
from nltk import sent_tokenize
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk import agreement

from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score


# get the samples and mappings of our quatrics survey
def label_samples(args):

    # get the label
    labels = []
    f = open(args.data_lst, "r")
    for line in f:
        labels.append(int(line.strip()))

    preds = None
    if args.mcq_model_preds is not None or args.bin_model_preds is not None:
        if args.task == "physicaliqa":
            curr_model_preds = args.mcq_model_preds
            prefix = "mcq"
        elif args.task == "physicalbinqa":
            curr_model_preds = args.bin_model_preds
            prefix = "bin"
        preds = {}
        for mcq_model_pred in curr_model_preds:
            pred_f = os.path.join(mcq_model_pred, "pred.lst")
            assert os.path.exists(pred_f)
            model_name = pred_f.split("/")[-2]
            model_name = prefix + "_" + model_name
            preds[model_name] = []
            pred_f = open(pred_f, "r")
            for line in pred_f:
                preds[model_name].append(int(line.strip()))
        pass

    # get original data
    qa_pair_dict = {}
    f = open(args.data_jsonl, "r")
    line_cnt = 0
    for line in f:
        data_raw = json.loads(line.strip())

        # get the raw data
        if args.task == "physicaliqa":
            goal = data_raw["goal"].strip()
            sol1 = data_raw["sol1"].strip()
            sol2 = data_raw["sol2"].strip()
            key = goal + "##" + sol1 + "##" + sol2
        elif args.task == "physicalbinqa":
            goal = data_raw["goal"].strip()
            sol  = data_raw["sol"].strip()
            key = goal + "##" + sol
        else:
            raise NotImplementedError("Not handled yet!")

        if preds is not None:
            data_raw["model_preds"] = {}
            for model_name in preds:
                data_raw["model_preds"][model_name] = preds[model_name][line_cnt]
            if args.task == "physicalbinqa":
                data_raw["model_preds_compl"] = {}
                for model_name in preds:
                    if line_cnt % 2 == 0:
                        compl_pred = preds[model_name][line_cnt+1]
                    else:
                        compl_pred = preds[model_name][line_cnt-1]
                    data_raw["model_preds_compl"][model_name] = compl_pred

        # store gt labels in
        data_raw["gt_label"] = labels[line_cnt]
        qa_pair_dict[key] = data_raw

        # add line counts
        line_cnt += 1

    f.close()

    # get the sorted qualtrics data
    if args.task == "physicaliqa":
        csv_header = ["goal", "sol1", "sol2"]
    elif args.task == "physicalbinqa":
        csv_header = ["goal", "sol"]
    else:
        raise NotImplementedError("Not handled yet!")

    f = open(args.samples_csv, 'r')
    csv_reader = csv.DictReader(f, delimiter=',', fieldnames=csv_header,
                                skipinitialspace=True)
    next(csv_reader) # skip the first line

    start_block_num = 2 # the first survey block is block 2
    start_qa_num = 1 # the first question labeled as 1 in each block
    qualt_sorted_dict = {}
    qualt_id_ai2_id_mapping = {}

    for row in csv_reader:
        if args.task == "physicaliqa":
            goal = row["goal"].strip()
            sol1 = row["sol1"].strip()
            sol2 = row["sol2"].strip()
            key = goal + "##" + sol1 + "##" + sol2
        elif args.task == "physicalbinqa":
            goal = row["goal"].strip()
            sol  = row["sol"].strip()
            key = goal + "##" + sol
        else:
            raise NotImplementedError("Not handled yet!")

        if len(goal) == 0: # reset
            end_qa_num = start_qa_num
            start_qa_num = 1
            start_block_num += 1
            continue

        # assert the current goal sol pair exists
        assert key in qa_pair_dict, "Something is wrong! {}".format(key)
        
        # store the raw data to each qualtrics id dict
        curr_qualtrics_label = "{}_Q{}".format(start_qa_num, start_block_num)
        qualt_sorted_dict[curr_qualtrics_label] = qa_pair_dict[key]

        # qualtrics-id to ai2-id mapping
        id_ = qa_pair_dict[key]["id"]
        qualt_id_ai2_id_mapping[curr_qualtrics_label] = id_

        start_qa_num += 1

    print ("[INFO] Num Each Block:   {}".format(end_qa_num-1))
    print ("[INFO] Num Total Blocks: {}".format(start_block_num-2))
    assert start_block_num - 2 == args.num_total_blocks
    assert end_qa_num - 1 == args.num_questions_each

    f.close()

    ai2_id_qualt_id_mapping = {v: u for u, v in qualt_id_ai2_id_mapping.items()}
    
    return qualt_sorted_dict, qualt_id_ai2_id_mapping, ai2_id_qualt_id_mapping
