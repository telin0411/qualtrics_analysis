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

import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(color_codes=True)

import nltk
from nltk import sent_tokenize
from nltk.corpus import words
from nltk.tokenize import word_tokenize
from nltk import agreement

from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score


# dicts for each categorical selections
EDUCATION_LEVEL = {
    "Most 10 year olds": 0,
    "Not most 10 year olds": 1,
    "Only domain experts": 2,
    "No one": 3,
    "not sure": 4,
}
EDUCATION_LEVEL_ID = {v: u for u, v in EDUCATION_LEVEL.items()}


CLEARNESS = {
    "everything": 0,
    "did not understand": 1,
    "did not make any sense": 2,
    "underspecified": 3,
    "not sure": 4,
    "None of the above": 5,
}
CLEARNESS_ID = {v: u for u, v in CLEARNESS.items()}


CATEGORIES = {
    "Typical Functions": 0,
    "Affordances": 1,
    "Environment/Spatial Relationships": 2,
    "Definitional Attributes": 3,
    "Everyday Knowledge": 4,
    "None of the above": 5,
}
CATEGORIES_ID = {v: u for u, v in CATEGORIES.items()}


IF_COMMON_SENSE = {
    "Yes": 1,
    "No": 0,
}
IF_COMMON_SENSE_ID = {v: u for u, v in IF_COMMON_SENSE.items()}


TASK_ABR = {
    "physicaliqa": "piqa",
    "physicalbinqa": "binpiqa",
}


CAT_NAMES = {
    "edu": "Educational Levels",
    "cat": "Physical Common Sense Categories",
    "com": "If Common Sense",
}
CAT_ID_DICTS = {
    "edu": EDUCATION_LEVEL_ID,
    "cat": CATEGORIES_ID,
    "com": IF_COMMON_SENSE_ID,
}


# arguments
def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('--task', type=str, default='physicalbinqa',
                        choices=["physicaliqa", "physicalbinqa"],
                        help='the task name')
    # for joint analysis
    parser.add_argument('--mcq_samples_csv', type=str, default=None, required=True,
                        help='sampled questions csv file for mcq-piqa')
    parser.add_argument('--bin_samples_csv', type=str, default=None, required=True,
                        help='sampled questions csv file for bin-piqa')
    parser.add_argument('--data_jsonl', type=str, default=None,
                        help='original questions jsonl file')
    parser.add_argument('--data_lst', type=str, default=None,
                        help='original questions label file')
    parser.add_argument('--mcq_data_jsonl', type=str, default=None,
                        help='original questions jsonl file')
    parser.add_argument('--mcq_data_lst', type=str, default=None,
                        help='original questions label file')
    parser.add_argument('--bin_data_jsonl', type=str, default=None,
                        help='original questions jsonl file')
    parser.add_argument('--bin_data_lst', type=str, default=None,
                        help='original questions label file')
    parser.add_argument('--mcq_processed_ids_json_file', default=None,
                        help='the processed json files for ids')
    parser.add_argument('--bin_processed_ids_json_file', default=None,
                        help='the processed json files for ids')
    parser.add_argument('--mcq_qualt_dict', default=None,
                        help='the processed json files')
    parser.add_argument('--bin_qualt_dict', default=None,
                        help='the processed json files')
    parser.add_argument('--num_examples_to_show', type=int, default=5)
    parser.add_argument('--mcq_model_preds', nargs="+", default=None,
                        help='the mcq model predictions')
    parser.add_argument('--bin_model_preds', nargs="+", default=None,
                        help='the bin model predictions')
    # for single analysis
    parser.add_argument('--samples_csv', type=str, default=None,
                        help='sampled questions csv file')
    parser.add_argument('--input_csv_files', nargs="+", default=None,
                        help='the input qualtrics csv files')
    parser.add_argument('--num_questions_each', type=int, default=30)
    parser.add_argument('--num_total_blocks', type=int, default=24)
    parser.add_argument('--start_block', type=int, default=None)
    parser.add_argument('--end_block', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default='outputs',
                        help='dir to save output files')
    parser.add_argument('--figs_dir', type=str, default='figs',
                        help='dir to save output figures')
    parser.add_argument('--top_k_words', type=int, default=30,
                        help='top k words to show')
    parser.add_argument('--start_k', type=int, default=None)
    parser.add_argument('--end_k', type=int, default=None)
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='if verbose')
    return parser


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
        elif args.task == "physicalbinqa":
            curr_model_preds = args.bin_model_preds
        preds = {}
        for mcq_model_pred in curr_model_preds:
            pred_f = os.path.join(mcq_model_pred, "pred.lst")
            assert os.path.exists(pred_f)
            model_name = pred_f.split("/")[-2]
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

 
def joint_qualitative_if_cs(mcq_res_ids, bin_res_ids, mqc_qualt_sorted_dict, 
                            bin_qualt_sorted_dict, ai2_id_qualt_id_mapping,
                            mcq_qualt_dict, bin_qualt_dict,
                            args=None):
    print ('-'*50)
    print ("[If Common Sense]")

    mcq_cs_ids = mcq_res_ids["com"]["1"]
    mcq_not_cs_ids = mcq_res_ids["com"]["0"]
    bin_cs_ids = bin_res_ids["com"]["1"]
    bin_not_cs_ids = bin_res_ids["com"]["0"]

    mcq_cs_ids = set(mcq_cs_ids)
    mcq_not_cs_ids = set(mcq_not_cs_ids)
    bin_cs_ids = set(bin_cs_ids)
    bin_not_cs_ids = set(bin_not_cs_ids)

    joint_cs_ids = mcq_cs_ids & bin_cs_ids
    joint_not_cs_ids = mcq_not_cs_ids & bin_not_cs_ids

    mcq_cs_bin_not_cs = mcq_cs_ids & bin_not_cs_ids
    bin_cs_mcq_not_cs = bin_cs_ids & mcq_not_cs_ids

    print ()
    print ('.'*50)
    print ("MCQ CS BIN Not CS Examples")
    print ('.'*50)
    show_cnt = 1
    for id_ in list(mcq_cs_bin_not_cs):
        if show_cnt > args.num_examples_to_show:
            break
        qualt_id = ai2_id_qualt_id_mapping[id_]
        goal_mcq = mqc_qualt_sorted_dict[qualt_id]["goal"]
        goal_bin = bin_qualt_sorted_dict[qualt_id]["goal"]
        assert goal_mcq == goal_bin
        sol1 = mqc_qualt_sorted_dict[qualt_id]["sol1"]
        sol2 = mqc_qualt_sorted_dict[qualt_id]["sol2"]
        sol = bin_qualt_sorted_dict[qualt_id]["sol"]
        print ('{}: {}'.format(show_cnt, goal_mcq))
        print ('mcq_sol1: {}'.format(sol1))
        print ('mcq_sol2: {}'.format(sol2))
        print ('bin_sol:  {}'.format(sol))
        print ('mcq_gt:   sol{}'.format(mqc_qualt_sorted_dict[qualt_id]["gt_label"]+1))
        print ('mcq_pred: {}'.format(mcq_qualt_dict[qualt_id]["annotations"]["1. choice"]))
        print ('bin_pred: {}'.format(bin_qualt_dict[qualt_id]["annotations"]["1. choice"]))
        print ('.'*50)
        show_cnt += 1

    print ()
    print ("BIN CS MCQ Not CS Examples")
    print ('.'*50)
    show_cnt = 1
    for id_ in list(bin_cs_mcq_not_cs):
        if show_cnt > args.num_examples_to_show:
            break
        qualt_id = ai2_id_qualt_id_mapping[id_]
        goal_mcq = mqc_qualt_sorted_dict[qualt_id]["goal"]
        goal_bin = bin_qualt_sorted_dict[qualt_id]["goal"]
        assert goal_mcq == goal_bin
        sol1 = mqc_qualt_sorted_dict[qualt_id]["sol1"]
        sol2 = mqc_qualt_sorted_dict[qualt_id]["sol2"]
        sol = bin_qualt_sorted_dict[qualt_id]["sol"]
        print ('{}: {}'.format(show_cnt, goal_mcq))
        print ('mcq_sol1: {}'.format(sol1))
        print ('mcq_sol2: {}'.format(sol2))
        print ('bin_sol:  {}'.format(sol))
        print ('mcq_gt:   sol{}'.format(mqc_qualt_sorted_dict[qualt_id]["gt_label"]+1))
        print ('mcq_pred: {}'.format(mcq_qualt_dict[qualt_id]["annotations"]["1. choice"]))
        print ('bin_pred: {}'.format(bin_qualt_dict[qualt_id]["annotations"]["1. choice"]))
        print ('.'*50)
        show_cnt += 1

    print ()
    print ("MCQ CS:     {}".format(len(mcq_cs_ids)))
    print ("MCQ Not CS: {}".format(len(mcq_not_cs_ids)))
    print ("BIN CS:     {}".format(len(bin_cs_ids)))
    print ("BIN Not CS: {}".format(len(bin_not_cs_ids)))
    print ('.'*50)
    print ('ALL CS:     {}'.format(len(joint_cs_ids)))
    print ('ALL Not CS: {}'.format(len(joint_not_cs_ids)))
    print ('.'*50)
    print ('MCQ CS BIN Not CS: {}'.format(len(mcq_cs_bin_not_cs)))
    print ('BIN CS MCQ Not CS: {}'.format(len(bin_cs_mcq_not_cs)))

    return None


def computer_iaas(mcq_qualt_dict, bin_qualt_dict, args, mode="mcq"):
    assert mode in ["mcq", "bin", "joint"]

    # decide the number of annotators
    mcq_num_annotators_per_question = 0
    bin_num_annotators_per_question = 0
    len_data = len(mcq_qualt_dict)
    assert len(mcq_qualt_dict) == len(bin_qualt_dict)

    gt_sol1_600 = 0
    gt_sol2_600 = 0
    gt_sol1_120 = 0
    gt_sol2_120 = 0

    for key in mcq_qualt_dict:
        if "annotations" not in mcq_qualt_dict[key]:
            continue
        annots = mcq_qualt_dict[key]["annotations"]["1. choice"]

        if int(key.split("Q")[-1]) >= 22:
            if mcq_qualt_dict[key]["gt_label"] == 0:
                gt_sol1_120 += 1
            else:
                gt_sol2_120 += 1
        else:
            if mcq_qualt_dict[key]["gt_label"] == 0:
                gt_sol1_600 += 1
            else:
                gt_sol2_600 += 1

        num_annots = len(annots)
        mcq_num_annotators_per_question = max(mcq_num_annotators_per_question, num_annots)

    # print ("gt_sol1_600:", gt_sol1_600)
    # print ("gt_sol2_600:", gt_sol2_600)
    # print ("gt_sol1_120:", gt_sol1_120)
    # print ("gt_sol2_120:", gt_sol2_120)
    # raise

    for key in bin_qualt_dict:
        if "annotations" not in bin_qualt_dict[key]:
            continue
        annots = bin_qualt_dict[key]["annotations"]["1. choice"]
        num_annots = len(annots)
        bin_num_annotators_per_question = max(bin_num_annotators_per_question, num_annots)

    if mode == "joint":
        num_iaas = mcq_num_annotators_per_question + bin_num_annotators_per_question
    elif mode == "mcq":
        num_iaas = mcq_num_annotators_per_question
    elif mode == "bin":
        num_iaas = bin_num_annotators_per_question

    print ()
    print ('-'*50)
    print ("Number of Annotators: {}".format(num_iaas))
    print ("IAA of mode: {}".format(mode))

    choices = np.zeros((len_data, num_iaas))
    if_cs = np.zeros((len_data, num_iaas))
    entry_count = [0 for i in range(num_iaas)]

    qualt_ids = sorted(list(mcq_qualt_dict.keys()))
    assert qualt_ids == sorted(list(bin_qualt_dict.keys()))

    joint_cs_correct_mcq = 0
    joint_not_cs_correct_mcq = 0
    joint_cs_correct_bin = 0
    joint_not_cs_correct_bin = 0
    joint_cs_cnt = 0
    joint_not_cs_cnt = 0

    bin_if_cs_iaa_cnts = []

    if mode == "mcq" or mode == "joint":
        
        for i in range(len_data):
            qualt_id = qualt_ids[i]
            mcq_annots = mcq_qualt_dict[qualt_id]
            if "annotations" not in mcq_annots:
                continue
            res_choice = mcq_annots["annotations"]["1. choice"]
            res_if_com = mcq_annots["annotations"]["4. if common sense"]
            if len(res_choice) < mcq_num_annotators_per_question:
                continue
            res_choice = np.asarray(res_choice)
            res_if_com = np.asarray(res_if_com)
            choices[i, :mcq_num_annotators_per_question] = res_choice
            if_cs[i, :mcq_num_annotators_per_question] = res_if_com

        pass

    if mode == "bin" or mode == "joint":
        
        for i in range(len_data):
            qualt_id = qualt_ids[i]
            bin_annots = bin_qualt_dict[qualt_id]
            if "annotations" not in bin_annots:
                continue
            res_choice = bin_annots["annotations"]["1. choice"]
            res_if_com = bin_annots["annotations"]["4. if common sense"]
            if len(res_choice) < bin_num_annotators_per_question:
                continue
            res_choice = np.asarray(res_choice)
            res_if_com = np.asarray(res_if_com)

            bin_if_cs_iaa_cnts.append(sum(res_if_com))

            if mode == "bin":
                choices[i, :bin_num_annotators_per_question] = res_choice
                if_cs[i, :bin_num_annotators_per_question] = res_if_com
            elif mode == "joint":
                choices[i, mcq_num_annotators_per_question:] = res_choice
                if_cs[i, mcq_num_annotators_per_question:] = res_if_com

                mcq_gt = mcq_qualt_dict[qualt_id]["gt_label"]
                bin_gt = bin_qualt_dict[qualt_id]["gt_label"]
                if sum(if_cs[i]) == num_iaas * 1:
                    if sum(choices[i, :mcq_num_annotators_per_question]) == mcq_num_annotators_per_question * mcq_gt:
                        joint_cs_correct_mcq += 1
                    if sum(choices[i, mcq_num_annotators_per_question:]) == bin_num_annotators_per_question * bin_gt:
                        joint_cs_correct_bin += 1
                    joint_cs_cnt += 1
                elif sum(if_cs[i]) == num_iaas * 0:
                    if sum(choices[i, :mcq_num_annotators_per_question]) == mcq_num_annotators_per_question * mcq_gt:
                        joint_not_cs_correct_mcq += 1
                    if sum(choices[i, mcq_num_annotators_per_question:]) == bin_num_annotators_per_question * bin_gt:
                        joint_not_cs_correct_bin += 1
                    joint_not_cs_cnt += 1

        pass

    choices = np.transpose(choices).astype(np.int32)
    if_cs = np.transpose(if_cs).astype(np.int32)

    if_cs_y1 = if_cs[0]
    if_cs_y2 = if_cs[1]
    skkappa = cohen_kappa_score(if_cs_y1, if_cs_y2)
    print (skkappa)

    taskdata_choices = []
    taskdata_if_cs = []
    for i in range(num_iaas):
        tmp = [[i, str(j), str(choices[i][j])] for j in range(len_data)]
        taskdata_choices += tmp
        tmp = [[i, str(j), str(if_cs[i][j])] for j in range(len_data)]
        taskdata_if_cs += tmp

    rating_choices = agreement.AnnotationTask(data=taskdata_choices)
    rating_if_cs = agreement.AnnotationTask(data=taskdata_if_cs)

    print ('.'*50)
    print ("Choices:")
    print ("kappa: {}".format(rating_choices.kappa()))
    print ("fleiss: {}".format(rating_choices.multi_kappa()))
    print ("alpha: {}".format(rating_choices.alpha()))
    # print ("scotts: {}".format(rating_choices.pi()))
    print ("If Common Sense:")
    print ("kappa: {}".format(rating_if_cs.kappa()))
    print ("fleiss: {}".format(rating_if_cs.multi_kappa()))
    print ("alpha: {}".format(rating_if_cs.alpha()))

    if mode == "joint":
        print ('.'*50)
        joint_cs_correct_mcq_acc = float(joint_cs_correct_mcq) / float(joint_cs_cnt) * 100.
        joint_not_cs_correct_mcq_acc = float(joint_not_cs_correct_mcq) / float(joint_not_cs_cnt) * 100.
        joint_cs_correct_bin_acc = float(joint_cs_correct_bin) / float(joint_cs_cnt) * 100.
        joint_not_cs_correct_bin_acc = float(joint_not_cs_correct_bin) / float(joint_not_cs_cnt) * 100.
        print ("Joint CS MCQ Correct =     {}/{} = {}%".format(
            joint_cs_correct_mcq, joint_cs_cnt, joint_cs_correct_mcq_acc))
        print ("Joint Not CS MCQ Correct = {}/{} = {}%".format(
            joint_not_cs_correct_mcq, joint_not_cs_cnt, joint_not_cs_correct_mcq_acc))
        print ("Joint CS BIN Correct =     {}/{} = {}%".format(
            joint_cs_correct_bin, joint_cs_cnt, joint_cs_correct_bin_acc))
        print ("Joint Not CS BIN Correct = {}/{} = {}%".format(
            joint_not_cs_correct_bin, joint_not_cs_cnt, joint_cs_correct_bin_acc))

    print ('-'*50)

    bin_if_cs_iaa_cnts = Counter(bin_if_cs_iaa_cnts)
    print (bin_if_cs_iaa_cnts)

    return None


def bin_vs_mcq_inspect(mcq_qualt_dict, bin_qualt_dict, ai2_id_qualt_id_mapping,
                       mcq_res_ids, bin_res_ids, args=None):

    # TODO: new way to compute bin accuracy
    bin_labels = []
    bin_preds = []
    bin_new_preds = {}
    bin_labels_f = open(args.bin_data_lst, "r")
    bin_preds_f = open(os.path.join(args.bin_model_preds[0], "pred.lst"), "r")
    for line in bin_labels_f:
        bin_labels.append(int(line.strip()))
    for line in bin_preds_f:
        bin_preds.append(int(line.strip()))
    bin_labels = np.asarray(bin_labels)
    bin_preds = np.asarray(bin_preds)
    bin_model_acc_org = np.mean(bin_labels==bin_preds)
    print ("BIN-Models Original Acc: {:.4f}".format(bin_model_acc_org))
    bin_data_f = open(args.bin_data_jsonl, "r")
    bin_idx = 0
    bin_new_acc = 0
    for line in bin_data_f:
        data_raw = json.loads(line.strip())
        id_curr = data_raw["id"]
        if id_curr in bin_new_preds:
            continue
        bin_pred1 = bin_preds[bin_idx]
        bin_pred2 = bin_preds[bin_idx+1]
        bin_gt1 = bin_labels[bin_idx]
        bin_gt2 = bin_labels[bin_idx+1]
        if bin_pred1 == bin_gt1 and bin_pred2 == bin_gt2:
            bin_new_preds[id_curr] = True
            bin_new_acc += 1
        else:
            bin_new_preds[id_curr] = False
        bin_idx += 2

    bin_new_acc = float(bin_new_acc) / len(bin_new_preds)
    print ("BIN-Models New Acc: {:.4f}".format(bin_new_acc))

    # decide the number of annotators
    num_annotators_per_question = 0
    len_data = len(mcq_qualt_dict)
    assert len(mcq_qualt_dict) == len(bin_qualt_dict)

    for key in mcq_qualt_dict:
        annots = mcq_qualt_dict[key]["annotations"]["1. choice"]
        num_annots = len(annots)
        num_annotators_per_question = max(num_annotators_per_question, num_annots)

    num_iaas = num_annotators_per_question * 2

    # TODO:
    qualt_ids = sorted(mcq_qualt_dict.keys())
    assert qualt_ids == sorted(bin_qualt_dict.keys())

    mcq_bin_perf_f = open(os.path.join(args.out_dir, "mcq_bin_performances_details.txt"), "w")
    mcq_bin_perf_f.write('-'*50+'\n')

    humans_mcq_y_bin_n_if_cs = []
    humans_mcq_y_bin_y_if_cs = []
    humans_mcq_n_bin_y_if_cs = []
    humans_mcq_n_bin_n_if_cs = []

    model_mcq_y_bin_n_if_cs = []
    model_mcq_y_bin_y_if_cs = []
    model_mcq_n_bin_y_if_cs = []
    model_mcq_n_bin_n_if_cs = []
    
    mcq_choice_iaa_if_cs_iaa_pred = []
    mcq_choice_iaa_if_cs_iaa_if_cs = []
    mcq_model_if_cs_iaa_pred = []
    mcq_model_if_cs_iaa_if_cs = []

    bin_choice_iaa_if_cs_iaa_pred = []
    bin_choice_iaa_if_cs_iaa_if_cs = []
    bin_model_if_cs_iaa_pred = []
    bin_model_if_cs_iaa_if_cs = []

    bin_model_correct_cond_mcq_model_correct = []
    mcq_model_correct_cond_bin_model_correct = []

    mcq_humans_perf_if_cs_iaa = []
    bin_humans_perf_if_cs_iaa = []
    mcq_models_perf_if_cs_iaa = []
    bin_models_perf_if_cs_iaa = []
    bin_models_new_perf_if_cs_iaa = []

    mcq_humans_perf_if_not_cs_iaa = []
    bin_humans_perf_if_not_cs_iaa = []
    mcq_models_perf_if_not_cs_iaa = []
    bin_models_perf_if_not_cs_iaa = []
    bin_models_new_perf_if_not_cs_iaa = []

    mcq_humans_perf_ties_iaa = []
    bin_humans_perf_ties_iaa = []
    mcq_models_perf_ties_iaa = []
    bin_models_perf_ties_iaa = []
    bin_models_new_perf_ties_iaa = []

    both_mcq_bin_if_cs_cnt = 0
    both_mcq_bin_if_not_cs_cnt = 0
    either_mcq_bin_if_cs_cnt = 0

    for qualt_id in qualt_ids:

        id_curr = mcq_qualt_dict[qualt_id]["id"]
        assert id_curr == bin_qualt_dict[qualt_id]["id"]

        mcq_data = mcq_qualt_dict[qualt_id]
        bin_data = bin_qualt_dict[qualt_id]
        mcq_annots = mcq_data["annotations"]
        bin_annots = bin_data["annotations"]

        mcq_gt = mcq_data["gt_label"]
        bin_gt = bin_data["gt_label"]

        mcq_choices = mcq_annots["1. choice"]
        bin_choices = bin_annots["1. choice"]
        mcq_choices_counter = Counter(mcq_choices)
        bin_choices_counter = Counter(bin_choices)
        mcq_choice, mcq_choice_cnt = mcq_choices_counter.most_common(1)[0]
        bin_choice, bin_choice_cnt = bin_choices_counter.most_common(1)[0]

        mcq_if_css = mcq_annots["4. if common sense"]
        bin_if_css = bin_annots["4. if common sense"]
        mcq_if_css_counter = Counter(mcq_if_css)
        bin_if_css_counter = Counter(bin_if_css)
        mcq_if_cs, mcq_if_cs_cnt = mcq_if_css_counter.most_common(1)[0]
        bin_if_cs, bin_if_cs_cnt = bin_if_css_counter.most_common(1)[0]

        mcq_bin_if_css_counter = Counter(mcq_if_css+bin_if_css)
        mcq_bin_if_cs, mcq_bin_if_cs_cnt = mcq_bin_if_css_counter.most_common(1)[0]

        if "model_preds" in mcq_data and "model_preds" in bin_data:
            assert len(mcq_data["model_preds"]) == 1
            assert len(bin_data["model_preds"]) == 1
            mcq_model_name = list(mcq_data["model_preds"].keys())[0]
            bin_model_name = list(bin_data["model_preds"].keys())[0]
            mcq_model_choice = mcq_data["model_preds"][mcq_model_name]
            bin_model_choice = bin_data["model_preds"][bin_model_name]

            # if_cs iaa
            if mcq_if_cs_cnt > num_annotators_per_question // 2:
                mcq_model_if_cs_iaa_pred.append(mcq_model_choice==mcq_gt)
                mcq_model_if_cs_iaa_if_cs.append(mcq_if_cs)
            if bin_if_cs_cnt > num_annotators_per_question // 2:
                bin_model_if_cs_iaa_pred.append(bin_model_choice==bin_gt)
                bin_model_if_cs_iaa_if_cs.append(bin_if_cs)

            if mcq_bin_if_cs_cnt == num_annotators_per_question:
                if mcq_model_choice == mcq_gt:
                    mcq_models_perf_ties_iaa.append(1)
                else:
                    mcq_models_perf_ties_iaa.append(0)
                if bin_model_choice == bin_gt:
                    bin_models_perf_ties_iaa.append(1)
                else:
                    bin_models_perf_ties_iaa.append(0)
                if bin_new_preds[id_curr]:
                    bin_models_new_perf_ties_iaa.append(1)
                else:
                    bin_models_new_perf_ties_iaa.append(0)
                if mcq_choice_cnt > num_annotators_per_question // 2:
                    mcq_humans_perf_ties_iaa.append(mcq_choice==mcq_gt)
                if bin_choice_cnt > num_annotators_per_question // 2:
                    bin_humans_perf_ties_iaa.append(bin_choice==bin_gt)

                either_mcq_bin_if_cs_cnt += 1
                
            # if mcq_if_cs_cnt > num_annotators_per_question // 2 \
            #         and bin_if_cs_cnt > num_annotators_per_question // 2:
            if mcq_bin_if_cs_cnt > num_annotators_per_question:
                mcq_goal = mcq_data["goal"]
                mcq_sol1 = mcq_data["sol1"]
                mcq_sol2 = mcq_data["sol2"]
                bin_goal = bin_data["goal"]
                assert mcq_goal == bin_goal
                bin_sol  = bin_data["sol"]
                assert bin_sol == mcq_sol1 or bin_sol == mcq_sol2

                # if common sense
                # if mcq_if_cs == 1 and bin_if_cs == 1:
                if mcq_bin_if_cs == 1:
                    if mcq_model_choice == mcq_gt:
                        if bin_model_choice == bin_gt:
                            bin_model_correct_cond_mcq_model_correct.append(1)
                        else:
                           bin_model_correct_cond_mcq_model_correct.append(0)
                    if bin_model_choice == bin_gt:
                        if mcq_model_choice == mcq_gt:
                            mcq_model_correct_cond_bin_model_correct.append(1)
                        else:
                            mcq_model_correct_cond_bin_model_correct.append(0)
                    if mcq_model_choice == mcq_gt:
                        mcq_models_perf_if_cs_iaa.append(1)
                    else:
                        mcq_models_perf_if_cs_iaa.append(0)
                    if bin_model_choice == bin_gt:
                        bin_models_perf_if_cs_iaa.append(1)
                    else:
                        bin_models_perf_if_cs_iaa.append(0)
                    if bin_new_preds[id_curr]:
                        bin_models_new_perf_if_cs_iaa.append(1)
                    else:
                        bin_models_new_perf_if_cs_iaa.append(0)
                    if mcq_choice_cnt > num_annotators_per_question // 2:
                        mcq_humans_perf_if_cs_iaa.append(mcq_choice==mcq_gt)
                    if bin_choice_cnt > num_annotators_per_question // 2:
                        bin_humans_perf_if_cs_iaa.append(bin_choice==bin_gt)

                    both_mcq_bin_if_cs_cnt += 1

                # if mcq_if_cs == 0 and bin_if_cs == 0:
                if mcq_bin_if_cs == 0:
                    both_mcq_bin_if_not_cs_cnt += 1
                # if mcq_if_cs != bin_if_cs:
                #     either_mcq_bin_if_cs_cnt += 1

                # if not common sense
                # if mcq_if_cs == 0 and bin_if_cs == 0:
                # if mcq_if_cs != bin_if_cs or (mcq_if_cs == 0 and bin_if_cs == 0):
                if mcq_bin_if_cs == 0:
                    if mcq_model_choice == mcq_gt:
                        mcq_models_perf_if_not_cs_iaa.append(1)
                    else:
                        mcq_models_perf_if_not_cs_iaa.append(0)
                    if bin_model_choice == bin_gt:
                        bin_models_perf_if_not_cs_iaa.append(1)
                    else:
                        bin_models_perf_if_not_cs_iaa.append(0)
                    if bin_new_preds[id_curr]:
                        bin_models_new_perf_if_not_cs_iaa.append(1)
                    else:
                        bin_models_new_perf_if_not_cs_iaa.append(0)
                    if mcq_choice_cnt > num_annotators_per_question // 2:
                        mcq_humans_perf_if_not_cs_iaa.append(mcq_choice==mcq_gt)
                    if bin_choice_cnt > num_annotators_per_question // 2:
                        bin_humans_perf_if_not_cs_iaa.append(bin_choice==bin_gt)

                mcq_humans_pred_str = "MCQ Humans Pred: {}".format(mcq_choices)
                bin_humans_pred_str = "BIN Humans Pred: {}".format(bin_choices)

                mcq_bin_perf_f.write("MCQ Goal: {}\n".format(mcq_goal))
                mcq_bin_perf_f.write("MCQ Sol1: {}\n".format(mcq_sol1))
                mcq_bin_perf_f.write("MCQ SOl2: {}\n".format(mcq_sol2))
                mcq_gt_str = "MCQ GT: {}".format(mcq_gt)
                mcq_pred_str = "MCQ Model Pred: {} ({})".format(
                    mcq_model_choice, "Correct" if mcq_model_choice==mcq_gt else "Incorrect")
                mcq_if_cs_str = "MCQ If Common Sense: {} ({})".format(
                    mcq_if_cs, "Yes" if mcq_if_cs==1 else "No")
                mcq_bin_perf_f.write(mcq_gt_str+"\n")
                mcq_bin_perf_f.write(mcq_pred_str+"\n")
                mcq_bin_perf_f.write(mcq_humans_pred_str+"\n")
                mcq_bin_perf_f.write(mcq_if_cs_str+"\n")
                mcq_bin_perf_f.write('.'*50+'\n')

                mcq_bin_perf_f.write("BIN Goal: {}\n".format(bin_goal))
                mcq_bin_perf_f.write("BIN SOl:  {}\n".format(bin_sol))
                bin_gt_str = "BIN GT: {}".format(bin_gt)
                bin_pred_str = "BIN Model Pred: {} ({})".format(
                    bin_model_choice, "Correct" if bin_model_choice==bin_gt else "Incorrect")
                bin_if_cs_str = "BIN If Common Sense: {} ({})".format(
                    bin_if_cs, "Yes" if bin_if_cs==1 else "No")
                mcq_bin_perf_f.write(bin_gt_str+"\n")
                mcq_bin_perf_f.write(bin_pred_str+"\n")
                mcq_bin_perf_f.write(bin_humans_pred_str+"\n")
                mcq_bin_perf_f.write(bin_if_cs_str+"\n")
                mcq_bin_perf_f.write('-'*50+'\n')

            # mcq_y_bin_n
            if mcq_model_choice == mcq_gt and bin_model_choice != bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        model_mcq_y_bin_n_if_cs.append(1)
                    else:
                        model_mcq_y_bin_n_if_cs.append(0)
            # mcq_y_bin_y
            if mcq_model_choice == mcq_gt and bin_model_choice == bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        model_mcq_y_bin_y_if_cs.append(1)
                    else:
                        model_mcq_y_bin_y_if_cs.append(0)
            # mcq_n_bin_y
            if mcq_model_choice != mcq_gt and bin_model_choice == bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        model_mcq_n_bin_y_if_cs.append(1)
                    else:
                        model_mcq_n_bin_y_if_cs.append(0)
            # mcq_n_bin_n
            if mcq_model_choice != mcq_gt and bin_model_choice != bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        model_mcq_n_bin_n_if_cs.append(1)
                    else:
                        model_mcq_n_bin_n_if_cs.append(0)

        # choice iaa if_cs iaa
        if mcq_choice_cnt > num_annotators_per_question // 2 \
                and mcq_if_cs_cnt > num_annotators_per_question // 2:
            mcq_choice_iaa_if_cs_iaa_pred.append(mcq_choice==mcq_gt)
            mcq_choice_iaa_if_cs_iaa_if_cs.append(mcq_if_cs)
        if bin_choice_cnt > num_annotators_per_question // 2 \
                and bin_if_cs_cnt > num_annotators_per_question // 2:
            bin_choice_iaa_if_cs_iaa_pred.append(bin_choice==bin_gt)
            bin_choice_iaa_if_cs_iaa_if_cs.append(bin_if_cs)
                
        # individual IAA
        if mcq_choice_cnt > num_annotators_per_question // 2 \
                and bin_choice_cnt > num_annotators_per_question // 2:
            
            # mcq_y_bin_n
            if mcq_choice == mcq_gt and bin_choice != bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        humans_mcq_y_bin_n_if_cs.append(1)
                    else:
                        humans_mcq_y_bin_n_if_cs.append(0)
            # mcq_y_bin_y
            if mcq_choice == mcq_gt and bin_choice == bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        humans_mcq_y_bin_y_if_cs.append(1)
                    else:
                        humans_mcq_y_bin_y_if_cs.append(0)
            # mcq_n_bin_y
            if mcq_choice != mcq_gt and bin_choice == bin_gt:
                if bin_if_cs_cnt > num_annotators_per_question // 2:
                    if bin_if_cs == 1:
                        humans_mcq_n_bin_y_if_cs.append(1)
                    else:
                        humans_mcq_n_bin_y_if_cs.append(0)
            # mcq_n_bin_n
            if mcq_choice != mcq_gt and bin_choice != bin_gt:
                if mcq_if_cs_cnt > num_annotators_per_question // 2:
                    if mcq_if_cs == 1:
                        humans_mcq_n_bin_n_if_cs.append(1)
                    else:
                        humans_mcq_n_bin_n_if_cs.append(0)

    humans_mcq_y_bin_n_if_cs = np.asarray(humans_mcq_y_bin_n_if_cs)
    humans_mcq_y_bin_y_if_cs = np.asarray(humans_mcq_y_bin_y_if_cs)
    humans_mcq_n_bin_y_if_cs = np.asarray(humans_mcq_n_bin_y_if_cs)
    humans_mcq_n_bin_n_if_cs = np.asarray(humans_mcq_n_bin_n_if_cs)

    humans_mcq_y_bin_n_if_cs_acc = np.mean(humans_mcq_y_bin_n_if_cs)
    humans_mcq_y_bin_y_if_cs_acc = np.mean(humans_mcq_y_bin_y_if_cs)
    humans_mcq_n_bin_y_if_cs_acc = np.mean(humans_mcq_n_bin_y_if_cs)
    humans_mcq_n_bin_n_if_cs_acc = np.mean(humans_mcq_n_bin_n_if_cs)

    model_mcq_y_bin_n_if_cs = np.asarray(model_mcq_y_bin_n_if_cs)
    model_mcq_y_bin_y_if_cs = np.asarray(model_mcq_y_bin_y_if_cs)
    model_mcq_n_bin_y_if_cs = np.asarray(model_mcq_n_bin_y_if_cs)
    model_mcq_n_bin_n_if_cs = np.asarray(model_mcq_n_bin_n_if_cs)

    model_mcq_y_bin_n_if_cs_acc = np.mean(model_mcq_y_bin_n_if_cs)
    model_mcq_y_bin_y_if_cs_acc = np.mean(model_mcq_y_bin_y_if_cs)
    model_mcq_n_bin_y_if_cs_acc = np.mean(model_mcq_n_bin_y_if_cs)
    model_mcq_n_bin_n_if_cs_acc = np.mean(model_mcq_n_bin_n_if_cs)

    print ("MCQ_Y_BIN_N_CS: {} (of {})".format(humans_mcq_y_bin_n_if_cs_acc,
        len(humans_mcq_y_bin_n_if_cs)))
    print ("MCQ_Y_BIN_Y_CS: {} (of {})".format(humans_mcq_y_bin_y_if_cs_acc,
        len(humans_mcq_y_bin_y_if_cs)))
    print ("MCQ_N_BIN_Y_CS: {} (of {})".format(humans_mcq_n_bin_y_if_cs_acc,
        len(humans_mcq_n_bin_y_if_cs)))
    print ("MCQ_N_BIN_N_CS: {} (of {})".format(humans_mcq_n_bin_n_if_cs_acc,
        len(humans_mcq_n_bin_n_if_cs)))

    print ('.'*50)
    print ("Model_MCQ_Y_BIN_N_CS: {} (of {})".format(model_mcq_y_bin_n_if_cs_acc,
        len(model_mcq_y_bin_n_if_cs)))
    print ("Model_MCQ_Y_BIN_Y_CS: {} (of {})".format(model_mcq_y_bin_y_if_cs_acc,
        len(model_mcq_y_bin_y_if_cs)))
    print ("Model_MCQ_N_BIN_Y_CS: {} (of {})".format(model_mcq_n_bin_y_if_cs_acc,
        len(model_mcq_n_bin_y_if_cs)))
    print ("Model_MCQ_N_BIN_N_CS: {} (of {})".format(model_mcq_n_bin_n_if_cs_acc,
        len(model_mcq_n_bin_n_if_cs)))

    # TODO Pearson Correlation:
    mcq_choice_iaa_if_cs_iaa_pred = np.asarray(mcq_choice_iaa_if_cs_iaa_pred) * 1.
    mcq_choice_iaa_if_cs_iaa_if_cs = np.asarray(mcq_choice_iaa_if_cs_iaa_if_cs) * 1.
    mcq_model_if_cs_iaa_pred = np.asarray(mcq_model_if_cs_iaa_pred) * 1.
    mcq_model_if_cs_iaa_if_cs = np.asarray(mcq_model_if_cs_iaa_if_cs) * 1.

    bin_choice_iaa_if_cs_iaa_pred = np.asarray(bin_choice_iaa_if_cs_iaa_pred) * 1.
    bin_choice_iaa_if_cs_iaa_if_cs = np.asarray(bin_choice_iaa_if_cs_iaa_if_cs) * 1.
    bin_model_if_cs_iaa_pred = np.asarray(bin_model_if_cs_iaa_pred) * 1.
    bin_model_if_cs_iaa_if_cs = np.asarray(bin_model_if_cs_iaa_pred) * 1.

    mcq_correct_if_cs_pr = pearsonr(mcq_choice_iaa_if_cs_iaa_pred, mcq_choice_iaa_if_cs_iaa_if_cs)
    mcq_model_correct_if_cs_pr = pearsonr(mcq_model_if_cs_iaa_pred, mcq_model_if_cs_iaa_if_cs)
    bin_correct_if_cs_pr = pearsonr(bin_choice_iaa_if_cs_iaa_pred, bin_choice_iaa_if_cs_iaa_if_cs)
    bin_model_correct_if_cs_pr = pearsonr(bin_model_if_cs_iaa_pred, bin_model_if_cs_iaa_if_cs)

    print ('.'*50)
    print ("MCQ-Humans, Correct vs. Common Sense Pearson: {:.3f}".format(mcq_correct_if_cs_pr[0]))
    print ("MCQ-Models, Correct vs. Common Sense Pearson: {:.3f}".format(mcq_model_correct_if_cs_pr[0]))
    print ("BIN-Humans, Correct vs. Common Sense Pearson: {:.3f}".format(bin_correct_if_cs_pr[0]))
    print ("BIN-Models, Correct vs. Common Sense Pearson: {:.3f}".format(bin_model_correct_if_cs_pr[0]))
    
    print ('.'*50)
    mcq_if_cs_indices = mcq_choice_iaa_if_cs_iaa_if_cs==1
    bin_if_cs_indices = bin_choice_iaa_if_cs_iaa_if_cs==1
    mcq_model_if_cs_indices = mcq_model_if_cs_iaa_if_cs==1
    bin_model_if_cs_indices = bin_model_if_cs_iaa_if_cs==1
    # Compute
    mcq_correct_cond_if_cs = np.mean(mcq_choice_iaa_if_cs_iaa_pred[mcq_if_cs_indices]==mcq_choice_iaa_if_cs_iaa_if_cs[mcq_if_cs_indices])
    mcq_model_correct_cond_if_cs = np.mean(mcq_model_if_cs_iaa_pred[mcq_model_if_cs_indices]==mcq_model_if_cs_iaa_if_cs[mcq_model_if_cs_indices])
    bin_correct_cond_if_cs = np.mean(bin_choice_iaa_if_cs_iaa_pred[bin_if_cs_indices]==bin_choice_iaa_if_cs_iaa_if_cs[bin_if_cs_indices])
    bin_model_correct_cond_if_cs = np.mean(bin_model_if_cs_iaa_pred[bin_model_if_cs_indices]==bin_model_if_cs_iaa_if_cs[bin_model_if_cs_indices])
    print ("MCQ-Humans, Correct conditioned on Common Sense: {:.3f}".format(mcq_correct_cond_if_cs))
    print ("MCQ-Models, Correct conditioned on Common Sense: {:.3f}".format(mcq_model_correct_cond_if_cs))
    print ("BIN-Humans, Correct conditioned on Common Sense: {:.3f}".format(bin_correct_cond_if_cs))
    print ("BIN-Models, Correct conditioned on Common Sense: {:.3f}".format(bin_model_correct_cond_if_cs))

    bin_model_correct_cond_mcq_model_correct = np.asarray(bin_model_correct_cond_mcq_model_correct)
    mcq_model_correct_cond_bin_model_correct = np.asarray(mcq_model_correct_cond_bin_model_correct)

    bin_model_correct_cond_mcq_model_correct_acc = np.mean(bin_model_correct_cond_mcq_model_correct)
    mcq_model_correct_cond_bin_model_correct_acc = np.mean(mcq_model_correct_cond_bin_model_correct)

    print ('.'*50)
    print (len(mcq_model_correct_cond_bin_model_correct))
    print (len(bin_model_correct_cond_mcq_model_correct))
    print ("MCQ Model Correct Conditioned on BIN Model Correct: {:.4f}".format(mcq_model_correct_cond_bin_model_correct_acc))
    print ("BIN Model Correct Conditioned on MCQ Model Correct: {:.4f}".format(bin_model_correct_cond_mcq_model_correct_acc))

    print ('.'*50)
    mcq_humans_perf_if_cs_iaa = np.asarray(mcq_humans_perf_if_cs_iaa)
    bin_humans_perf_if_cs_iaa = np.asarray(bin_humans_perf_if_cs_iaa)
    mcq_humans_perf_if_cs_iaa_acc = np.mean(mcq_humans_perf_if_cs_iaa)
    bin_humans_perf_if_cs_iaa_acc = np.mean(bin_humans_perf_if_cs_iaa)
    mcq_models_perf_if_cs_iaa = np.asarray(mcq_models_perf_if_cs_iaa)
    bin_models_perf_if_cs_iaa = np.asarray(bin_models_perf_if_cs_iaa)
    mcq_models_perf_if_cs_iaa_acc = np.mean(mcq_models_perf_if_cs_iaa)
    bin_models_perf_if_cs_iaa_acc = np.mean(bin_models_perf_if_cs_iaa)
    bin_models_new_perf_if_cs_iaa = np.asarray(bin_models_new_perf_if_cs_iaa)
    bin_models_new_perf_if_cs_iaa_acc = np.mean(bin_models_new_perf_if_cs_iaa)
    print (len(mcq_humans_perf_if_cs_iaa))
    print (len(bin_humans_perf_if_cs_iaa))
    print (len(mcq_models_perf_if_cs_iaa))
    print (len(bin_models_perf_if_cs_iaa))
    print ("MCQ Humans Both MCQ-BIN-CS: {:.4f}".format(mcq_humans_perf_if_cs_iaa_acc))
    print ("BIN Humans Both MCQ-BIN-CS: {:.4f}".format(bin_humans_perf_if_cs_iaa_acc))
    print ("MCQ Models Both MCQ-BIN-CS: {:.4f}".format(mcq_models_perf_if_cs_iaa_acc))
    print ("BIN Models Both MCQ-BIN-CS: {:.4f}".format(bin_models_perf_if_cs_iaa_acc))
    print ("BIN Models Both MCQ-BIN-CS: {:.4f} (New)".format(bin_models_new_perf_if_cs_iaa_acc))

    print ('.'*50)
    mcq_humans_perf_if_not_cs_iaa = np.asarray(mcq_humans_perf_if_not_cs_iaa)
    bin_humans_perf_if_not_cs_iaa = np.asarray(bin_humans_perf_if_not_cs_iaa)
    mcq_humans_perf_if_not_cs_iaa_acc = np.mean(mcq_humans_perf_if_not_cs_iaa)
    bin_humans_perf_if_not_cs_iaa_acc = np.mean(bin_humans_perf_if_not_cs_iaa)
    mcq_models_perf_if_not_cs_iaa = np.asarray(mcq_models_perf_if_not_cs_iaa)
    bin_models_perf_if_not_cs_iaa = np.asarray(bin_models_perf_if_not_cs_iaa)
    mcq_models_perf_if_not_cs_iaa_acc = np.mean(mcq_models_perf_if_not_cs_iaa)
    bin_models_perf_if_not_cs_iaa_acc = np.mean(bin_models_perf_if_not_cs_iaa)
    bin_models_new_perf_if_not_cs_iaa = np.asarray(bin_models_new_perf_if_not_cs_iaa)
    bin_models_new_perf_if_not_cs_iaa_acc = np.mean(bin_models_new_perf_if_not_cs_iaa)
    print (len(mcq_humans_perf_if_not_cs_iaa))
    print (len(bin_humans_perf_if_not_cs_iaa))
    print (len(mcq_models_perf_if_not_cs_iaa))
    print (len(bin_models_perf_if_not_cs_iaa))
    print ("MCQ Humans Neither MCQ-BIN-CS: {:.4f}".format(mcq_humans_perf_if_not_cs_iaa_acc))
    print ("BIN Humans Neither MCQ-BIN-CS: {:.4f}".format(bin_humans_perf_if_not_cs_iaa_acc))
    print ("MCQ Models Neither MCQ-BIN-CS: {:.4f}".format(mcq_models_perf_if_not_cs_iaa_acc))
    print ("BIN Models Neither MCQ-BIN-CS: {:.4f}".format(bin_models_perf_if_not_cs_iaa_acc))
    print ("BIN Models Neither MCQ-BIN-CS: {:.4f} (New)".format(bin_models_new_perf_if_not_cs_iaa_acc))

    print ('.'*50)
    mcq_humans_perf_ties_iaa = np.asarray(mcq_humans_perf_ties_iaa)
    bin_humans_perf_ties_iaa = np.asarray(bin_humans_perf_ties_iaa)
    mcq_humans_perf_ties_iaa_acc = np.mean(mcq_humans_perf_ties_iaa)
    bin_humans_perf_ties_iaa_acc = np.mean(bin_humans_perf_ties_iaa)
    mcq_models_perf_ties_iaa = np.asarray(mcq_models_perf_ties_iaa)
    bin_models_perf_ties_iaa = np.asarray(bin_models_perf_ties_iaa)
    mcq_models_perf_ties_iaa_acc = np.mean(mcq_models_perf_ties_iaa)
    bin_models_perf_ties_iaa_acc = np.mean(bin_models_perf_ties_iaa)
    bin_models_new_perf_ties_iaa = np.asarray(bin_models_new_perf_ties_iaa)
    bin_models_new_perf_ties_iaa_acc = np.mean(bin_models_new_perf_ties_iaa)
    print (len(mcq_humans_perf_ties_iaa))
    print (len(bin_humans_perf_ties_iaa))
    print (len(mcq_models_perf_ties_iaa))
    print (len(bin_models_perf_ties_iaa))
    print ("MCQ Humans Ties MCQ-BIN-CS: {:.4f}".format(mcq_humans_perf_ties_iaa_acc))
    print ("BIN Humans Ties MCQ-BIN-CS: {:.4f}".format(bin_humans_perf_ties_iaa_acc))
    print ("MCQ Models Ties MCQ-BIN-CS: {:.4f}".format(mcq_models_perf_ties_iaa_acc))
    print ("BIN Models Ties MCQ-BIN-CS: {:.4f}".format(bin_models_perf_ties_iaa_acc))
    print ("BIN Models Ties MCQ-BIN-CS: {:.4f} (New)".format(bin_models_new_perf_ties_iaa_acc))

    print ('.'*50)
    print ("Both MCQ-BIN-CS Count:       ", both_mcq_bin_if_cs_cnt)
    print ("Neither MCQ-BIN-Not-CS Count:", both_mcq_bin_if_not_cs_cnt)
    print ("Ties MCQ-BIN-CS Count:       ", either_mcq_bin_if_cs_cnt)

    return None


def sample_compl_binpiqa(ai2_id_qualt_id_mapping, mcq_qualt_dict, bin_qualt_dict, args=None):
    qualt_ids = sorted(list(mcq_qualt_dict.keys()))
    assert qualt_ids == sorted(list(bin_qualt_dict.keys()))

    csv_out = open(os.path.join(args.out_dir, "binpiqa_survey_sampels_complementary.csv"), "w")
    fieldnames = ["goal", "sol", "gt", "ai2id", "qualtid_org", "qualtid_new"]
    wr = csv.DictWriter(csv_out, fieldnames=fieldnames)
    wr.writeheader()

    samples_cnt = 1
    qualt_block = 2
    qualt_block_org_start = 2
    qualt_block_org_end = len(qualt_ids) // 30

    for qualt_block_org in range(qualt_block_org_start, qualt_block_org_end):
        for qualt_ques_idx in range(1, 31):
            qualt_id = "{}_Q{}".format(qualt_ques_idx, qualt_block_org)
            mcq_data = mcq_qualt_dict[qualt_id]
            bin_data = bin_qualt_dict[qualt_id]
            mcq_goal = mcq_data["goal"]
            mcq_sol1 = mcq_data["sol1"]
            mcq_sol2 = mcq_data["sol2"]
            mcq_id   = mcq_data["id"]
            mcq_gt   = mcq_data["gt_label"]
            bin_goal = bin_data["goal"]
            bin_sol  = bin_data["sol"]
            bin_id   = bin_data["id"]
            bin_gt   = bin_data["gt_label"]
            assert bin_sol == mcq_sol1 or bin_sol == mcq_sol2
            assert mcq_id == bin_id
            bin_annots = bin_data["annotations"]
            bin_choices = bin_annots["1. choice"]
            
            bin_choices_counter = Counter(bin_choices)
            bin_choice, bin_choice_cnt = bin_choices_counter.most_common(1)[0]

            if samples_cnt > 30:
                samples_cnt = 1
                qualt_block += 1
                row = {"goal":"", "sol":"", "gt":"", "ai2id":"", "qualtid_org":"", "qualtid_new":""}
                wr.writerow(row)

            if bin_choice == bin_gt and bin_choice_cnt > len(bin_choices) // 2:
                new_bon_goal = bin_goal
                new_bin_sol = mcq_sol1 if bin_sol == mcq_sol2 else mcq_sol2
                new_bin_gt = 1 - bin_gt
                new_bin_ai2id = bin_id
                qualtid_org = qualt_id
                qualtid_new = "{}_Q{}".format(samples_cnt, qualt_block)
                samples_cnt += 1
                row = {
                    "goal": new_bon_goal,
                    "sol": new_bin_sol,
                    "gt": new_bin_gt,
                    "ai2id": new_bin_ai2id,
                    "qualtid_org": qualtid_org,
                    "qualtid_new": qualtid_new,
                }
                wr.writerow(row)

    return None


# the main function
def analyze_pipeline(args):
    pp = pprint.PrettyPrinter(indent=2)

    # the task
    print ('-'*50)
    print ("[INFO] Task: {}".format(args.task))

    # get the qualtrics question id dict and the qualtrics-id to ai2-id mappings
    mcq_qualt_dict = json.load(open(args.mcq_qualt_dict, "r"))
    bin_qualt_dict = json.load(open(args.bin_qualt_dict, "r"))

    # mcq
    args.task = "physicaliqa"
    args.data_jsonl = args.mcq_data_jsonl
    args.data_lst = args.mcq_data_lst
    args.samples_csv = args.mcq_samples_csv
    mqc_qualt_sorted_dict, qualt_id_ai2_id_mapping, ai2_id_qualt_id_mapping = label_samples(args)
    
    # bin
    args.task = "physicalbinqa"
    args.data_jsonl = args.bin_data_jsonl
    args.data_lst = args.bin_data_lst
    args.samples_csv = args.bin_samples_csv
    bin_qualt_sorted_dict, _, _ = label_samples(args)

    # sample complementary bin-piqa
    # sample_compl_binpiqa(ai2_id_qualt_id_mapping, mcq_qualt_dict, bin_qualt_dict, args)
    # raise

    # merge with model preds
    if args.mcq_model_preds is not None and args.bin_model_preds is not None:
        for qualt_id  in mqc_qualt_sorted_dict:
            if qualt_id not in mcq_qualt_dict:
                continue
            if qualt_id not in bin_qualt_dict:
                continue
            mcq_qualt_dict[qualt_id]["model_preds"] = mqc_qualt_sorted_dict[qualt_id]["model_preds"]
            bin_qualt_dict[qualt_id]["model_preds"] = bin_qualt_sorted_dict[qualt_id]["model_preds"]

    # TODO: Main joint analysis
    mcq_res_ids = json.load(open(args.mcq_processed_ids_json_file, "r"))
    bin_res_ids = json.load(open(args.bin_processed_ids_json_file, "r"))

    # TODO: if common sense
    joint_qualitative_if_cs(mcq_res_ids, bin_res_ids, mqc_qualt_sorted_dict,
                            bin_qualt_sorted_dict, ai2_id_qualt_id_mapping,
                            mcq_qualt_dict, bin_qualt_dict, args)

    # IAAs
    computer_iaas(mcq_qualt_dict, bin_qualt_dict, args, mode="mcq")
    computer_iaas(mcq_qualt_dict, bin_qualt_dict, args, mode="bin")
    computer_iaas(mcq_qualt_dict, bin_qualt_dict, args, mode="joint")

    # BIN vs. MCQ Inspection
    bin_vs_mcq_inspect(mcq_qualt_dict, bin_qualt_dict,
                       ai2_id_qualt_id_mapping, mcq_res_ids, bin_res_ids, args)

    # TODO: add models performances here
    pass

    print ()
    print ('-'*50)
    print ("[INFO] Analysis Done!")
    print ('-'*50)

    return None


# execution
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # output dir
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    
    # main analysis pipeline
    analyze_pipeline(args)
