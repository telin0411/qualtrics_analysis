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

 
# reading the qualtrics csv files
def read_qualtric_raw_csv(qualtrics_csv, qualt_sorted_dict, args, pp):
    print ('-'*50)
    print ("[INFO] Procssing {} ...".format(qualtrics_csv))

    qualtrics_csv_file = open(qualtrics_csv, "r")
    csv_reader = csv.DictReader(qualtrics_csv_file)

    if args.start_block is None:
        start_block = 2
    else:
        start_block = max(2, args.start_block)

    if args.end_block is None:
        end_block = args.num_total_blocks + 2
    else:
        end_block = min(args.num_total_blocks + 2, args.end_block)
    
    row_cnt = 0
    for row in csv_reader:

        # the annotated data will have "email" in "DistributionChannel"
        if row["DistributionChannel"] == "email":
            row_cnt += 1

            for block_idx in range(start_block, end_block):
                for question_idx in range(1, args.num_questions_each+1):

                    # annotation dict
                    qualtric_id = "{}_Q{}".format(question_idx, block_idx)
                    if "annotations" not in qualt_sorted_dict[qualtric_id]:
                        qualt_sorted_dict[qualtric_id]["annotations"] = {}
                    curr_annot_dict = qualt_sorted_dict[qualtric_id]["annotations"]

                    for sub_question_idx in range(2, 10): # question ranging from 2 to 9

                        # qualtric sub id
                        qualtric_sub_id = "{}_Q{}.{}".format(question_idx, block_idx, sub_question_idx)
                        qualtric_sub_id_ = "{}_Q{}.{}_1".format(question_idx, block_idx, sub_question_idx)
                        qualtric_sub_id_TEXT = "{}_Q{}.{}_7_TEXT".format(question_idx, block_idx, sub_question_idx)
                        assert qualtric_sub_id in row or qualtric_sub_id_ in row
                        if qualtric_sub_id in row:
                            answer = row[qualtric_sub_id]
                        elif qualtric_sub_id_ in row:
                            answer = row[qualtric_sub_id_]
                        
                        # skip if not annotated
                        if len(answer) <= 0:
                            continue

                        # for which to choose
                        if sub_question_idx == 2:
                            if "1. choice" not in curr_annot_dict:
                                curr_annot_dict["1. choice"] = []
                            if args.task == "physicaliqa":
                                if "sol1" in answer:
                                    curr_annot_dict["1. choice"].append(0)
                                elif "sol2" in answer:
                                    curr_annot_dict["1. choice"].append(1)
                            elif args.task == "physicalbinqa":
                                if "Incorrect" in answer:
                                    curr_annot_dict["1. choice"].append(0)
                                elif "Correct" in answer:
                                    curr_annot_dict["1. choice"].append(1)

                        # for confidence level
                        if sub_question_idx == 3:
                            if "2. confidence" not in curr_annot_dict:
                                curr_annot_dict["2. confidence"] = []
                            curr_annot_dict["2. confidence"].append(float(answer))

                        # for others agreement
                        if sub_question_idx == 4:
                            if "3. others agreement" not in curr_annot_dict:
                                curr_annot_dict["3. others agreement"] = []
                            curr_annot_dict["3. others agreement"].append(float(answer))

                        # if common sense
                        if sub_question_idx == 5:
                            if "4. if common sense" not in curr_annot_dict:
                                curr_annot_dict["4. if common sense"] = []
                            if "Yes" in answer:
                                curr_annot_dict["4. if common sense"].append(1)
                            elif "No" in answer:
                                curr_annot_dict["4. if common sense"].append(0)

                        # education level
                        if sub_question_idx == 6:
                            if "5. education level" not in curr_annot_dict:
                                curr_annot_dict["5. education level"] = []
                            for key in EDUCATION_LEVEL:
                                if key in answer:
                                    answer_id = EDUCATION_LEVEL[key]
                                    curr_annot_dict["5. education level"].append(answer_id)

                        # clearness of the question prompt and solution(s)
                        if sub_question_idx == 7:
                            if "6. clearness" not in curr_annot_dict:
                                curr_annot_dict["6. clearness"] = []
                            for key in CLEARNESS:
                                if key in answer:
                                    answer_id = CLEARNESS[key]
                                    curr_annot_dict["6. clearness"].append(answer_id)

                        # missing text
                        if sub_question_idx == 8:
                            if answer != "N/A":
                                if "6. clearness" not in curr_annot_dict:
                                    curr_annot_dict["6. clearness"] = []
                                    curr_annot_dict["6. clearness"].append(CLEARNESS["None of the above"])
                                elif "6. clearness" in curr_annot_dict and \
                                        len(curr_annot_dict["6. clearness"]) == 0:
                                    curr_annot_dict["6. clearness"].append(CLEARNESS["None of the above"])

                                if "6.1. Missing Words" not in curr_annot_dict:
                                    curr_annot_dict["6.1. Missing Words"] = []
                                curr_annot_dict["6.1. Missing Words"].append(answer)

                        # categories of the data instance
                        if sub_question_idx == 9:
                            if "7. category" not in curr_annot_dict:
                                curr_annot_dict["7. category"] = []
                            answer_ids = []
                            for key in CATEGORIES:
                                if key in answer:
                                    answer_id = CATEGORIES[key]
                                    answer_ids.append(answer_id)
                            curr_annot_dict["7. category"].append(answer_ids)

                            # None of the above text field
                            if qualtric_sub_id_TEXT in row:
                                if len(row[qualtric_sub_id_TEXT]) > 0:
                                    answer_ids = [CATEGORIES["None of the above"]]
                                    curr_annot_dict["7. category"].append(answer_ids)
                    
                    # pp.pprint(qualt_sorted_dict[qualtric_id])
            pass
            ####

    qualtrics_csv_file.close()
    print ("[INFO] This part has been annotated by {} annotators".format(row_cnt))

    return qualt_sorted_dict


# a function to display categorical data
def display_categorical(arr, data_type, data_type_str):
    d = {x: 0 for x in range(len(data_type))}
    total_cnt = 0

    for ele in arr:
        d[ele] += 1
        total_cnt += 1
    
    for cat in sorted(d):
        cat_cnt = d[cat]
        cat_name = data_type[cat]
        cat_perc = float(cat_cnt) / float(total_cnt) * 100.0
        print ("[Analysis] [{}] {}: {}/{} = {:.2f} % ".format(data_type_str,
            cat_name, cat_cnt, total_cnt, cat_perc))

    return d


# some simple statistics function
def simple_analysis(qualt_sorted_dict, args, pp):

    # generals
    num_annotators_per_question = 0

    # statistics holders
    correct_cnt = 0
    total_cnt = 0
    confidences = []
    agreements = []
    if_common_sense = []
    edu_level = []
    clearness = []
    categories = []

    confidences_when_correct = []
    confidences_when_wrong = []

    which_correct_iaa = []
    which_correct_iaa_and_correct = []

    if_common_sense_iaa = []
    if_common_sense_iaa_cs_and_correct = []
    if_common_sense_iaa_not_cs_and_correct = []
    if_common_sense_iaa_cs_confs = []
    if_common_sense_iaa_not_cs_confs = []

    edu_level_iaa = []
    edu_level_iaa_counts = []
    edu_level_iaa_perf = {}

    cat_iaa = {
        0: {"cnt": 0, "correct": 0},
        1: {"cnt": 0, "correct": 0},
        2: {"cnt": 0, "correct": 0},
        3: {"cnt": 0, "correct": 0},
        4: {"cnt": 0, "correct": 0},
        5: {"cnt": 0, "correct": 0},
    }
    cat_iaa_1 = {
        0: {"cnt": 0, "correct": 0},
        1: {"cnt": 0, "correct": 0},
        2: {"cnt": 0, "correct": 0},
        3: {"cnt": 0, "correct": 0},
        4: {"cnt": 0, "correct": 0},
        5: {"cnt": 0, "correct": 0},
    }
    cat_iaa_ids = {
        0: [],
        1: [],
        2: [],
        3: [],
        4: [],
        5: [],
    }

    if_cs_ids_f = open(os.path.join(args.out_dir, "{}_common_sense_iaa_ids.txt".format(TASK_ABR[args.task])), "w")
    if_not_cs_ids_f = open(os.path.join(args.out_dir, "{}_not_common_sense_iaa_ids.txt".format(TASK_ABR[args.task])), "w")
    if_cs_ids_dict = {}
    if_not_cs_ids_dict = {}

    # TODO: all categorical data
    all_cat_data_ids = {
        "edu": {c: [] for c in EDUCATION_LEVEL_ID},
        "cat": {c: [] for c in CATEGORIES_ID},
        "com": {c: [] for c in IF_COMMON_SENSE_ID},
        "all": []
    }
    ai2_ids2data = {}

    missing_words_list = []

    # looping the data
    for qualt_id in qualt_sorted_dict:
    
        # TODO: sometimes we can analyze a range of results
        if "annotations" not in qualt_sorted_dict[qualt_id]:
            continue

        curr_annots = qualt_sorted_dict[qualt_id]["annotations"]
        if len(curr_annots) == 0: # not yet annotated!
            continue

        gt = qualt_sorted_dict[qualt_id]["gt_label"]
        id_curr = qualt_sorted_dict[qualt_id]["id"]
        all_cat_data_ids["all"].append(id_curr)

        ai2_ids2data[id_curr] = {"goal": qualt_sorted_dict[qualt_id]["goal"]}
        if args.task == "physicaliqa":
            ai2_ids2data[id_curr]["sol1"] = qualt_sorted_dict[qualt_id]["sol1"]
            ai2_ids2data[id_curr]["sol2"] = qualt_sorted_dict[qualt_id]["sol2"]
        elif args.task == "physicalbinqa":
            ai2_ids2data[id_curr]["sol"] = qualt_sorted_dict[qualt_id]["sol"]

        # human accuracies
        choices = curr_annots["1. choice"]
        for choice in choices:
            if choice ==  gt:
                correct_cnt += 1
        total_cnt += len(choices)
        num_annotators_per_question = max(len(choices), num_annotators_per_question)
        if len(choices) > 1:
            which_correct_iaa.append(sum(choices))
            # FIXME can't be 0.5 in the future
            if float(sum(choices)) / float(len(choices)) > 0.5:
                for choice in choices:
                    if choice == gt:
                        which_correct_iaa_and_correct.append(1)
                    else:
                        which_correct_iaa_and_correct.append(0)
            elif float(sum(choices)) / float(len(choices)) < 0.5:
                for choice in choices:
                    if choice == gt:
                        which_correct_iaa_and_correct.append(1)
                    else:
                        which_correct_iaa_and_correct.append(0)

        # confidences
        confs = curr_annots["2. confidence"]
        confidences += confs
        for choice_idx in range(len(choices)):
            choice = choices[choice_idx]
            if choice == gt:
                confidences_when_correct.append(confs[choice_idx])
            else:
                confidences_when_wrong.append(confs[choice_idx])

        # others agreements
        agrees = curr_annots["3. others agreement"]
        agreements += agrees

        # if common sense
        if_cs = curr_annots["4. if common sense"]
        if_common_sense += if_cs
        if len(if_cs) > 1:
            if_common_sense_iaa.append(sum(if_cs))
            # if IAA >= 50% and answered correctly
            # FIXME can't be 0.5 in the future
            if float(sum(if_cs)) / float(len(if_cs)) > 0.5:
                if id_curr not in if_cs_ids_dict:
                    if_cs_ids_dict[id_curr] = {"gt": gt, "human_preds": []}

                for choice in choices:
                    if choice == gt:
                        if_common_sense_iaa_cs_and_correct.append(1)
                    else:
                        if_common_sense_iaa_cs_and_correct.append(0)
                    if_cs_ids_dict[id_curr]["human_preds"].append(choice)
                if_cs_ids_f.write(id_curr+'\n')
                all_cat_data_ids["com"][1].append(id_curr)

                if_common_sense_iaa_cs_confs += confs

            elif float(sum(if_cs)) / float(len(if_cs)) < 0.5:
                if id_curr not in if_not_cs_ids_dict:
                    if_not_cs_ids_dict[id_curr] = {"gt": gt, "human_preds": []}

                for choice in choices:
                    if choice == gt:
                        if_common_sense_iaa_not_cs_and_correct.append(1)
                    else:
                        if_common_sense_iaa_not_cs_and_correct.append(0)
                    if_not_cs_ids_dict[id_curr]["human_preds"].append(choice)
                if_not_cs_ids_f.write(id_curr+'\n')
                all_cat_data_ids["com"][0].append(id_curr)

                if_common_sense_iaa_not_cs_confs += confs

        # educational level
        edu = curr_annots["5. education level"]
        edu_level += edu
        if len(edu) > 1:
            edu_counter = Counter(edu)
            edu_iaa_curr = edu_counter.most_common(1)[0]
            edu_iaa_ele, edu_iaa_cnt = edu_iaa_curr
            if edu_iaa_cnt > len(edu) // 2:
                edu_level_iaa.append(edu_iaa_ele)
                edu_level_iaa_counts.append(1)
                all_cat_data_ids["edu"][edu_iaa_ele].append(id_curr)
                for choice in choices:
                    if choice == gt:
                        if edu_iaa_ele not in edu_level_iaa_perf:
                            edu_level_iaa_perf[edu_iaa_ele] = 1
                        else:
                            edu_level_iaa_perf[edu_iaa_ele] += 1
            else:
                edu_level_iaa_counts.append(0)

        # question clearness
        if "6. clearness" in curr_annots:
            clr = curr_annots["6. clearness"]
            clearness += clr

        # categories
        if "7. category" in curr_annots:
            ctrg = curr_annots["7. category"]
            ctrg_curr = []
            for ctrg_user in ctrg:
                categories += ctrg_user
                ctrg_curr += ctrg_user

            if len(ctrg) > 1:
                ctrg_curr = Counter(ctrg_curr)
                for ctrg_num in ctrg_curr:
                    ctrg_num_cnt = ctrg_curr[ctrg_num]
                    if ctrg_num_cnt > len(ctrg) // 2:
                        cat_iaa[ctrg_num]["cnt"] += 1
                        all_cat_data_ids["cat"][ctrg_num].append(id_curr)
                        cat_iaa_ids[ctrg_num].append(id_curr)
                        for choice in choices:
                            if choice == gt:
                                cat_iaa[ctrg_num]["correct"] += 1
                    if ctrg_num_cnt == len(ctrg) // 2:
                        cat_iaa_1[ctrg_num]["cnt"] += 1
                        for choice in choices:
                            if choice == gt:
                                cat_iaa_1[ctrg_num]["correct"] += 1

        pass

        if "6.1. Missing Words" in curr_annots:
            for missing_words_curr in curr_annots["6.1. Missing Words"]:
                if 'n/a' not in missing_words_curr and 'N/a' not in missing_words_curr:
                    missing_words_list.append(missing_words_curr)
        
        pass

    ###########################################################################

    # post processing
    confidences = np.asarray(confidences)
    agreements = np.asarray(agreements)
    if_common_sense = np.asarray(if_common_sense)
    edu_level = np.asarray(edu_level)
    clearness = np.asarray(clearness)
    categories = np.asarray(categories)
    confidences_when_correct = np.asarray(confidences_when_correct)
    confidences_when_wrong = np.asarray(confidences_when_wrong)
    if_common_sense_iaa_cs_confs = np.asarray(if_common_sense_iaa_cs_confs)
    if_common_sense_iaa_not_cs_confs = np.asarray(if_common_sense_iaa_not_cs_confs)

    # results
    human_acc = float(correct_cnt) / float(total_cnt) * 100.0
    mean_confidences = np.mean(confidences)
    mean_agreements = np.mean(agreements)
    mean_if_cs = np.mean(if_common_sense) * 100.0
    mean_conf_when_correct = np.mean(confidences_when_correct)
    mean_conf_when_wrong = np.mean(confidences_when_wrong)
    mean_if_cs_iaa_cs_confs = np.mean(if_common_sense_iaa_cs_confs)
    mean_if_cs_iaa_not_cs_confs = np.mean(if_common_sense_iaa_not_cs_confs)

    print ("[Analysis] Human Accuracy: {:.2f} %".format(human_acc))
    print ("[Analysis] Mean Confidence: {:.2f} %".format(mean_confidences))
    print ("[Analysis] Mean Others Agree: {:.2f} %".format(mean_agreements))
    print ("[Analysis] Mean If Common Sense: {:.2f} %".format(mean_if_cs))
    print ("[Analysis] Mean Confidence when Correct: {:.2f} %".format(mean_conf_when_correct))
    print ("[Analysis] Mean Confidence when Wrong: {:.2f} %".format(mean_conf_when_wrong))
    print ("[Analysis] Mean Confidence Common Sense: {:.2f} %".format(mean_if_cs_iaa_cs_confs))
    print ("[Analysis] Mean Confidence Not Common Sense: {:.2f} %".format(mean_if_cs_iaa_not_cs_confs))

    # display histograms of categorical data
    print ('.'*50)
    display_categorical(edu_level, EDUCATION_LEVEL_ID, "EDUCATION_LEVEL")
    print ('.'*50)
    display_categorical(clearness, CLEARNESS_ID, "CLEARNESS")
    print ('.'*50)
    display_categorical(categories, CATEGORIES_ID, "CATEGORIES")

    print ('-'*50)
    print ('[IAA]')
    print ("Num Annotators per Question: {}".format(num_annotators_per_question))
    if_cs_bins = list(range(0, num_annotators_per_question+2))
    if_cs_hist, _ = np.histogram(if_common_sense_iaa, bins=if_cs_bins)
    print (if_cs_hist, _)
    print (if_cs_hist / np.sum(if_cs_hist) * 100.)
    if_common_sense_iaa_cs_and_correct = np.asarray(if_common_sense_iaa_cs_and_correct)
    if_common_sense_iaa_cs_and_correct_acc = np.mean(if_common_sense_iaa_cs_and_correct)
    if_common_sense_iaa_not_cs_and_correct = np.asarray(if_common_sense_iaa_not_cs_and_correct)
    if_common_sense_iaa_not_cs_and_correct_acc = np.mean(if_common_sense_iaa_not_cs_and_correct)
    print (len(if_common_sense_iaa_cs_and_correct))
    print (if_common_sense_iaa_cs_and_correct_acc)
    print (len(if_common_sense_iaa_not_cs_and_correct))
    print (if_common_sense_iaa_not_cs_and_correct_acc)
    print ('-'*50)

    which_correct_bins = list(range(0, num_annotators_per_question+2))
    which_correct_hist, _ = np.histogram(which_correct_iaa, bins=which_correct_bins)
    print (which_correct_hist, _)
    print (which_correct_hist / np.sum(which_correct_hist) * 100.)
    which_correct_iaa_and_correct = np.asarray(which_correct_iaa_and_correct)
    which_correct_iaa_and_correct_acc = np.mean(which_correct_iaa_and_correct)
    print (len(which_correct_iaa_and_correct))
    print (which_correct_iaa_and_correct_acc)
    print ('-'*50)

    print (len(edu_level_iaa))
    edu_level_bins = list(range(0, len(EDUCATION_LEVEL_ID)+2))
    edu_level_hist, _ = np.histogram(edu_level_iaa, bins=edu_level_bins)
    print (edu_level_hist, _)
    print (edu_level_hist / np.sum(edu_level_hist) * 100.)
    print (float(sum(edu_level_iaa_counts)) / float(len(edu_level_iaa_counts))* 100.)
    pp.pprint(edu_level_iaa_perf)
    edu_level_iaa_perf_l = [edu_level_iaa_perf[c] for c in sorted(edu_level_iaa_perf)]
    edu_level_iaa_perf_l = np.asarray(edu_level_iaa_perf_l) / num_annotators_per_question
    edu_level_hist_l = np.asarray([x for x in edu_level_hist if x > 0])
    edu_level_hist_acc = edu_level_iaa_perf_l / edu_level_hist_l * 100.0
    print (edu_level_hist_acc)

    print ('-'*50)
    pp.pprint(cat_iaa)
    pp.pprint(cat_iaa_1)
    cat_counts_l = np.asarray([cat_iaa[cat]["cnt"] for cat in cat_iaa])
    cat_corrects_l = np.asarray([cat_iaa[cat]["correct"] for cat in cat_iaa])
    cat_corrects_l = cat_corrects_l / num_annotators_per_question
    cat_corrects_acc = cat_corrects_l / cat_counts_l * 100.0
    print (cat_corrects_acc)

    # close files
    if_cs_ids_f.close()
    if_not_cs_ids_f.close()

    # dump files
    json.dump(cat_iaa_ids, open(os.path.join("files", "{}_cat_ids.json".format(args.task)), "w"))
    if_cs_ids_json = open(os.path.join(args.out_dir, "{}_common_sense_iaa_ids.json".format(TASK_ABR[args.task])), "w")
    if_not_cs_ids_json = open(os.path.join(args.out_dir, "{}_not_common_sense_iaa_ids.json".format(TASK_ABR[args.task])), "w")
    json.dump(if_cs_ids_dict, if_cs_ids_json)
    json.dump(if_not_cs_ids_dict, if_not_cs_ids_json)

    # plotting top-k words histograms
    top_k_words_hist(all_cat_data_ids, ai2_ids2data, args, tags_prefix=['N', 'V'])

    # saving all the categorical ids file
    json.dump(all_cat_data_ids, open(os.path.join(args.out_dir, "{}_all_cat_ids.json".format(TASK_ABR[args.task])), "w"))
    
    # missing terms
    pp.pprint(missing_words_list)

    return None


def top_k_words_hist(d, ai2_ids2data, args, tags_prefix=None):
    print ('-'*50)
    print ("Saving various figures to {}".format(args.figs_dir))

    if args.start_k is None:
        start_k = 0
    else:
        start_k = max(0, args.start_k)
    if args.end_k is None:
        end_k = args.top_k_words
    else:
        end_k = args.end_k

    cat_keys = ["com", "edu", "cat"]
    for cat_key in cat_keys:
        figs_root = os.path.join(args.figs_dir, args.task, cat_key)
        if not os.path.exists(figs_root):
            os.makedirs(figs_root)

        for cat in sorted(d[cat_key]):
            cat_words_list = []
            for id_ in d[cat_key][cat]:
                sents = []
                goal = ai2_ids2data[id_]["goal"]
                if args.task == "physicaliqa":
                    sol1 = ai2_ids2data[id_]["sol1"]
                    sol2 = ai2_ids2data[id_]["sol2"]
                    sents = [goal, sol1, sol2]
                elif args.task == "physicalbinqa":
                    sol = ai2_ids2data[id_]["sol"]
                    sents = [goal, sol]

                # FIXME: currently only doing simple word tokenize, can do
                # ner/pos tag in the future as well, if more informative
                for sent in sents:
                    tokens = word_tokenize(sent)
                    tokens_and_tags = nltk.pos_tag(tokens)
                    if tags_prefix is not None:
                        tokens = [w for w, t in tokens_and_tags if t[0] in tags_prefix]
                    cat_words_list += tokens

            cat_words_dict = Counter(cat_words_list)
            top_k_words = cat_words_dict.most_common(len(cat_words_dict))
            top_k_words = top_k_words[start_k:end_k]
            values = [v for (w, v) in top_k_words]
            tokens = [w for (w, v) in top_k_words]

            # TODO: plotting
            fig = plt.figure(figsize=(20, 10))
            plt.bar(range(len(values)), values, align='center')
            plt.xticks(range(len(tokens)), tokens)
            plt.xticks(rotation=90, fontsize=14)
            title = CAT_NAMES[cat_key] + ": " + CAT_ID_DICTS[cat_key][cat]
            plt.title(title)
            print (title)
            save_name = cat_key + "_" + str(cat) + ".png"
            save_path = os.path.join(figs_root, save_name)
            fig.savefig(save_path, bbox_inches="tight")
            plt.close(fig)

    return None


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
    num_annotators_per_question = 0
    len_data = len(mcq_qualt_dict)
    assert len(mcq_qualt_dict) == len(bin_qualt_dict)

    for key in mcq_qualt_dict:
        annots = mcq_qualt_dict[key]["annotations"]["1. choice"]
        num_annots = len(annots)
        num_annotators_per_question = max(num_annotators_per_question, num_annots)

    if mode == "joint":
        num_iaas = num_annotators_per_question * 2
    else:
        num_iaas = num_annotators_per_question

    print ()
    print ('-'*50)
    print ("Number of Annotators: {}".format(num_annotators_per_question))
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

    if mode == "mcq" or mode == "joint":
        
        for i in range(len_data):
            qualt_id = qualt_ids[i]
            mcq_annots = mcq_qualt_dict[qualt_id]
            res_choice = mcq_annots["annotations"]["1. choice"]
            res_if_com = mcq_annots["annotations"]["4. if common sense"]
            if len(res_choice) < num_annotators_per_question:
                continue
            res_choice = np.asarray(res_choice)
            res_if_com = np.asarray(res_if_com)
            choices[i, :num_annotators_per_question] = res_choice
            if_cs[i, :num_annotators_per_question] = res_if_com

        pass

    if mode == "bin" or mode == "joint":
        
        for i in range(len_data):
            qualt_id = qualt_ids[i]
            bin_annots = bin_qualt_dict[qualt_id]
            res_choice = bin_annots["annotations"]["1. choice"]
            res_if_com = bin_annots["annotations"]["4. if common sense"]
            if len(res_choice) < num_annotators_per_question:
                continue
            res_choice = np.asarray(res_choice)
            res_if_com = np.asarray(res_if_com)
            if mode == "bin":
                choices[i, :num_annotators_per_question] = res_choice
                if_cs[i, :num_annotators_per_question] = res_if_com
            elif mode == "joint":
                choices[i, num_annotators_per_question:] = res_choice
                if_cs[i, num_annotators_per_question:] = res_if_com

                mcq_gt = mcq_qualt_dict[qualt_id]["gt_label"]
                bin_gt = bin_qualt_dict[qualt_id]["gt_label"]
                if sum(if_cs[i]) == num_iaas * 1:
                    if sum(choices[i, :num_annotators_per_question]) == num_annotators_per_question * mcq_gt:
                        joint_cs_correct_mcq += 1
                    if sum(choices[i, num_annotators_per_question:]) == num_annotators_per_question * bin_gt:
                        joint_cs_correct_bin += 1
                    joint_cs_cnt += 1
                elif sum(if_cs[i]) == num_iaas * 0:
                    if sum(choices[i, :num_annotators_per_question]) == num_annotators_per_question * mcq_gt:
                        joint_not_cs_correct_mcq += 1
                    if sum(choices[i, num_annotators_per_question:]) == num_annotators_per_question * bin_gt:
                        joint_not_cs_correct_bin += 1
                    joint_not_cs_cnt += 1

        pass

    choices = np.transpose(choices).astype(np.int32)
    if_cs = np.transpose(if_cs).astype(np.int32)

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
    # print ("alpha: {}".format(rating_choices.alpha()))
    # print ("scotts: {}".format(rating_choices.pi()))
    print ("If Common Sense:")
    print ("kappa: {}".format(rating_if_cs.kappa()))
    print ("fleiss: {}".format(rating_if_cs.multi_kappa()))

    if mode == "joint":
        print ('.'*50)
        joint_cs_correct_mcq_acc = float(joint_cs_correct_mcq) / float(joint_cs_cnt) * 100.
        joint_not_cs_correct_mcq_acc = float(joint_not_cs_correct_mcq) / float(joint_not_cs_cnt) * 100.
        joint_cs_correct_bin_acc = float(joint_cs_correct_bin) / float(joint_cs_cnt) * 100.
        joint_not_cs_correct_bin_acc = float(joint_not_cs_correct_bin) / float(joint_not_cs_cnt) * 100.
        print ("Joint CS MCQ Correct =     {}/{} = {}%".format(joint_cs_correct_mcq, joint_cs_cnt, joint_cs_correct_mcq_acc))
        print ("Joint Not CS MCQ Correct = {}/{} = {}%".format(joint_not_cs_correct_mcq, joint_not_cs_cnt, joint_not_cs_correct_mcq_acc))
        print ("Joint CS BIN Correct =     {}/{} = {}%".format(joint_cs_correct_bin, joint_cs_cnt, joint_cs_correct_bin_acc))
        print ("Joint Not CS BIN Correct = {}/{} = {}%".format(joint_not_cs_correct_bin, joint_not_cs_cnt, joint_cs_correct_bin_acc))

    print ('-'*50)

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

    # looping over the given qualtrics csv files
    # for qualtrics_csv in args.input_csv_files:
    #     qualt_sorted_dict = read_qualtric_raw_csv(qualtrics_csv,
    #         qualt_sorted_dict, args, pp)
    
    # print ('-'*50)
    # print ("[INFO] Showing one examplar processed data instance ...")
    # exp_key = sorted(list(qualt_sorted_dict.keys()))[0]
    # print ("Qualtric ID: {}".format(exp_key))
    # pp.pprint(qualt_sorted_dict[exp_key])
    # print ('-'*50)

    # get some quick statistics
    # simple_analysis(qualt_sorted_dict, args, pp)

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