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


# arguments
def get_parser():
    def str2bool(v):
        v = v.lower()
        assert v == 'true' or v == 'false'
        return v.lower() == 'true'

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     conflict_handler='resolve')
    parser.add_argument('--input_legacy_file', default=None,
                        help='the input qualtrics csv file in legacy format')
    parser.add_argument('--out_file_path', type=str, default=None,
                        help='dir to save output files')
    parser.add_argument('--num_blocks', type=int, default=24,
                        help='number of blocks')
    parser.add_argument('--verbose', type=str2bool, default=False,
                        help='if verbose')
    return parser


def transform_legacy(args):
    print ("Legacy File: {}".format(args.input_legacy_file))

    legacy_file = open(args.input_legacy_file, "r")
    csv_reader = csv.DictReader(legacy_file)

    new_rows = []

    for row in csv_reader:

        assert "\ufeffV1" in row or "ResponseID" in row
        assert "V6" in row or "RecipientLastName" in row
        if "\ufeffV1" in row:
            if row["\ufeffV1"] == "ResponseID":
                continue
        elif "ResponseID" in row:
            if row["ResponseID"] == "ResponseID":
                continue
        if "V5" in row:
            if len(row["V5"]) == 0: # empty email address
                continue
        elif "RecipientEmail" in row:
            if len(row["RecipientEmail"]) == 0: # empty email address
                continue

        email_address = row["V5"] if "V5" in row else row["RecipientEmail"]
        new_row = {}
        new_row["RecipientEmail"] = email_address
        new_row["DistributionChannel"] = "email"

        for key in row:
            if "Q" in key and "." in key and "(" in key and ")" in key:
                loop_num = key.split("(")[-1].split(")")[0]
                loop_num = int(loop_num)
                qualt_id = key.split("(")[0]
                new_key = "{}_{}".format(loop_num, qualt_id)
                if ".9" in qualt_id and "TEXT" not in qualt_id:
                    new_key = new_key[:-2]
                    # print (key, new_key, row[key])
                    if new_key not in new_row:
                        new_row[new_key] = ""
                    if len(new_row[new_key]) > 0:
                        if len(row[key]) > 0:
                            new_row[new_key] += "," + row[key]
                    else:
                        new_row[new_key] = row[key]
                elif ".1" in qualt_id:
                    pass # don't add
                else:
                    new_row[new_key] = row[key]
        
        new_rows.append(new_row)

    print ("Num annotations:", len(new_rows))

    # 1_Q2.2,1_Q2.3_1,1_Q2.4_1,1_Q2.5,1_Q2.6,1_Q2.7,1_Q2.8,1_Q2.9,1_Q2.9_7_TEXT
    fileds = ["RecipientEmail", "DistributionChannel"]
    for block_num in range(2, args.num_blocks+2):
        for loop_num in range(1, 31):
            for ques_num in range(2, 10):
                additional_id = None

                qualt_id = "{}_Q{}.{}".format(loop_num, block_num, ques_num)
                if ques_num == 9:
                    additional_id = qualt_id + "_7_TEXT"
                if ques_num == 3 or ques_num == 4:
                    qualt_id += "_1"
                assert qualt_id in new_rows[0], "{}".format(qualt_id)
                fileds.append(qualt_id)

                if additional_id is not None:
                    assert additional_id in new_rows[0], "{}".format(additional_id)
                    fileds.append(additional_id)
    
    wr = csv.DictWriter(open(args.out_file_path, "w"), fieldnames=fileds)
    wr.writeheader()
    for new_row in new_rows:
        wr.writerow(new_row)

    return None


# execution
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # main analysis pipeline
    transform_legacy(args)
