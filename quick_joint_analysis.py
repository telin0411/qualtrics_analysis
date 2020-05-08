import json
import csv
import os
import numpy as np
import sys
import pandas as pd



piqa1 = "files/PIQASurveyTeLinToDistributePart1.csv"
piqa2 = "files/PIQASurveyTeLinToDistributePart2.csv"
binpiqa1 = "files/BINPIQASurveyTeLinToDistributePart1.csv"
binpiqa2 = "files/BINPIQASurveyTeLinToDistributePart2.csv"


df_piqa1 = pd.read_csv(piqa1)
df_piqa2 = pd.read_csv(piqa2)
df_binpiqa1 = pd.read_csv(binpiqa1)
df_binpiqa2 = pd.read_csv(binpiqa2)
# print (df_piqa1)
# print (len(df_piqa1))


frames = [
    df_piqa1,
    df_piqa2,
    df_binpiqa1,
    df_binpiqa2,
]


df = pd.concat(frames)
# print (df)
# print (len(df))


piqa_cs = json.load(open("outputs/piqa_common_sense_iaa_ids.json", "r"))
piqa_not_cs = json.load(open("outputs/piqa_not_common_sense_iaa_ids.json", "r"))
binpiqa_cs = json.load(open("outputs/binpiqa_common_sense_iaa_ids.json", "r"))
binpiqa_not_cs = json.load(open("outputs/binpiqa_not_common_sense_iaa_ids.json", "r"))


joint_cs_ids_dict = {}
joint_not_cs_ids_dict = {}

for id_ in piqa_cs:
    if id_ in binpiqa_cs:
        joint_cs_ids_dict[id_] = [piqa_cs[id_]]

for id_ in binpiqa_cs:
    if id_ in piqa_cs:
        assert id_ in joint_cs_ids_dict
        joint_cs_ids_dict[id_].append(binpiqa_cs[id_])

for id_ in piqa_not_cs:
    if id_ in binpiqa_not_cs:
        joint_not_cs_ids_dict[id_] = [piqa_not_cs[id_]]

for id_ in binpiqa_not_cs:
    if id_ in piqa_not_cs:
        assert id_ in joint_not_cs_ids_dict
        joint_not_cs_ids_dict[id_].append(binpiqa_not_cs[id_])

print (len(joint_cs_ids_dict))
print (len(joint_not_cs_ids_dict))

piqa_cs_correct_cnts = []
binpiqa_cs_correct_cnts = []
for id_ in joint_cs_ids_dict:
    piqa_res, binpiqa_res = joint_cs_ids_dict[id_]
    piqa_gt = piqa_res["gt"]
    for choice in piqa_res["human_preds"]:
        if choice == piqa_gt:
            piqa_cs_correct_cnts.append(1)
        else:
            piqa_cs_correct_cnts.append(0)
    binpiqa_gt = binpiqa_res["gt"]
    for choice in binpiqa_res["human_preds"]:
        if choice == binpiqa_gt:
            binpiqa_cs_correct_cnts.append(1)
        else:
            binpiqa_cs_correct_cnts.append(0)

piqa_not_cs_correct_cnts = []
binpiqa_not_cs_correct_cnts = []
for id_ in joint_not_cs_ids_dict:
    piqa_res, binpiqa_res = joint_not_cs_ids_dict[id_]
    piqa_gt = piqa_res["gt"]
    for choice in piqa_res["human_preds"]:
        if choice == piqa_gt:
            piqa_not_cs_correct_cnts.append(1)
        else:
            piqa_not_cs_correct_cnts.append(0)
    binpiqa_gt = binpiqa_res["gt"]
    for choice in binpiqa_res["human_preds"]:
        if choice == binpiqa_gt:
            binpiqa_not_cs_correct_cnts.append(1)
        else:
            binpiqa_not_cs_correct_cnts.append(0)

piqa_cs_correct_cnts = np.asarray(piqa_cs_correct_cnts)
binpiqa_cs_correct_cnts = np.asarray(binpiqa_cs_correct_cnts)
print ("PIQA CS:         {:.4f} %".format(np.mean(piqa_cs_correct_cnts)))
print ("BIN-PIQA CS:     {:.4f} %".format(np.mean(binpiqa_cs_correct_cnts)))
json.dump(joint_cs_ids_dict, open("outputs/joint_piqq_binpiqa_common_sense_iaa_ids.json", "w"))

piqa_not_cs_correct_cnts = np.asarray(piqa_not_cs_correct_cnts)
binpiqa_not_cs_correct_cnts = np.asarray(binpiqa_not_cs_correct_cnts)
print ("PIQA Not CS:     {:.4f} %".format(np.mean(piqa_not_cs_correct_cnts)))
print ("BIN-PIQA Not CS: {:.4f} %".format(np.mean(binpiqa_not_cs_correct_cnts)))
json.dump(joint_not_cs_ids_dict, open("outputs/joint_piqq_binpiqa_not_common_sense_iaa_ids.json", "w"))
