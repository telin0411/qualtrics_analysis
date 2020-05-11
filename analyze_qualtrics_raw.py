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


TASK_ABR = {
    "physicaliqa": "piqa",
    "physicalbinqa": "binpiqa",
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
    parser.add_argument('--samples_csv', type=str, default=None, required=True,
                        help='sampled questions csv file')
    parser.add_argument('--data_jsonl', type=str, default=None,
                        help='original questions jsonl file')
    parser.add_argument('--data_lst', type=str, default=None,
                        help='original questions label file')
    parser.add_argument('--input_csv_files', nargs="+", default=None,
                        help='the input qualtrics csv files')
    parser.add_argument('--num_questions_each', type=int, default=30)
    parser.add_argument('--num_total_blocks', type=int, default=24)
    parser.add_argument('--start_block', type=int, default=None)
    parser.add_argument('--end_block', type=int, default=None)
    parser.add_argument('--out_dir', type=str, default='outputs',
                        help='dir to save output files')
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
    return qualt_sorted_dict, qualt_id_ai2_id_mapping

 
# reading the qualtrics csv files
def read_qualtric_raw_csv(qualtrics_csv, qualt_sorted_dict, args, pp):
    print ('-'*50)
    print ("[INFO] Procssing {} ...".format(qualtrics_csv))

    qualtrics_csv_file = open(qualtrics_csv, "r")
    csv_reader = csv.DictReader(qualtrics_csv_file)

    if args.start_block is None:
        start_block = 2
    if args.end_block is None:
        end_block = args.num_total_blocks + 2
    
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

    which_correct_iaa = []
    which_correct_iaa_and_correct = []

    if_common_sense_iaa = []
    if_common_sense_iaa_cs_and_correct = []
    if_common_sense_iaa_not_cs_and_correct = []

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

    for qualt_id in qualt_sorted_dict:
        curr_annots = qualt_sorted_dict[qualt_id]["annotations"]
        if len(curr_annots) == 0: # not yet annotated!
            continue

        gt = qualt_sorted_dict[qualt_id]["gt_label"]
        id_curr = qualt_sorted_dict[qualt_id]["id"]

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
                        cat_iaa_ids[ctrg_num].append(id_curr)
                        for choice in choices:
                            if choice == gt:
                                cat_iaa[ctrg_num]["correct"] += 1
            pass

    ###########################################################################

    # post processing
    confidences = np.asarray(confidences)
    agreements = np.asarray(agreements)
    if_common_sense = np.asarray(if_common_sense)
    edu_level = np.asarray(edu_level)
    clearness = np.asarray(clearness)
    categories = np.asarray(categories)

    # results
    human_acc = float(correct_cnt) / float(total_cnt) * 100.0
    mean_confidences = np.mean(confidences)
    mean_agreements = np.mean(agreements)
    mean_if_cs = np.mean(if_common_sense) * 100.0

    print ("[Analysis] Human Accuracy: {:.2f} %".format(human_acc))
    print ("[Analysis] Mean Confidence: {:.2f} %".format(mean_confidences))
    print ("[Analysis] Mean Others Agree: {:.2f} %".format(mean_agreements))
    print ("[Analysis] Mean If Common Sense: {:.2f} %".format(mean_if_cs))

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

    print ('-'*50)
    pp.pprint(cat_iaa)

    # close files
    if_cs_ids_f.close()
    if_not_cs_ids_f.close()

    # dump files
    json.dump(cat_iaa_ids, open(os.path.join("files", "{}_cat_ids.json".format(args.task)), "w"))
    if_cs_ids_json = open(os.path.join(args.out_dir, "{}_common_sense_iaa_ids.json".format(TASK_ABR[args.task])), "w")
    if_not_cs_ids_json = open(os.path.join(args.out_dir, "{}_not_common_sense_iaa_ids.json".format(TASK_ABR[args.task])), "w")
    json.dump(if_cs_ids_dict, if_cs_ids_json)
    json.dump(if_not_cs_ids_dict, if_not_cs_ids_json)

    return None


# the main function
def analyze_pipeline(args):
    pp = pprint.PrettyPrinter(indent=2)

    # the task
    print ('-'*50)
    print ("[INFO] Task: {}".format(args.task))

    # get the qualtrics question id dict and the qualtrics-id to ai2-id mappings
    qualt_sorted_dict, qualt_id_ai2_id_mapping = label_samples(args)

    # looping over the given qualtrics csv files
    for qualtrics_csv in args.input_csv_files:
        qualt_sorted_dict = read_qualtric_raw_csv(qualtrics_csv,
            qualt_sorted_dict, args, pp)
    
    # show one example processed instance
    # entries are:
    # { "annotations": {
    #         "1. choice": ,
    #         "2. confidence": ,
    #         "3. others agreement": ,
    #         "4. if common sense": ,
    #         "5. education level": ,
    #         "6. clearness": ,
    #         "7. category": ,
    #     }
    # }
    print ('-'*50)
    print ("[INFO] Showing one examplar processed data instance ...")
    exp_key = sorted(list(qualt_sorted_dict.keys()))[0]
    print ("Qualtric ID: {}".format(exp_key))
    pp.pprint(qualt_sorted_dict[exp_key])
    print ('-'*50)

    # get some quick statistics
    simple_analysis(qualt_sorted_dict, args, pp)

    # TODO: add models performances here
    pass

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
