import os
import subprocess
import codecs
from collections import defaultdict
import jsonlines
import spacy
import csv

nlp = spacy.load("en_core_web_sm")

def convert_pred_to_bottom_up(tag):
    converted = []
    if len(tag.strip()) > 1:
        tag = tag.split("-")
        prefix = tag[0]
        suffix = tag[1]
        if suffix in ["deontic", "buletic", "teleological", "intentional", "buletic_teleological"]:
            converted.append(f"{prefix}-priority")
        elif suffix in ["epistemic", "ability", "circumstantial", "opportunity", "ability_circumstantial",
                       "epistemic_circumstantial"]:
            converted.append(f"{prefix}-plausibility")
        else:
            converted.append(f"{prefix}-{suffix}")
    else:
        converted.append(tag)
    return converted[0]

def convert_tags_to_mnm(tag):
    converted = []
    if len(tag.strip()) > 1:
        tag = tag.split("-")
        prefix = tag[0]
        suffix = tag[1]
        converted.append(f"{prefix}-modal")
    else:
        converted.append(tag)
    return converted[0]

def convert_to_eval_format(filename, outfilename, bottom_up=False):
    outfile = codecs.open(outfilename, 'w')
    with jsonlines.open(filename) as reader:
        for obj in reader:
            pred = obj["tags"]
            gold = obj["gold_labels"]
            words = obj["words"]
            for p, g, w in zip(pred, gold, words):
                if bottom_up:
                    #g = convert_pred_to_bottom_up(g)
                    #p = convert_pred_to_bottom_up(p)
                    g = convert_tags_to_mnm(g)
                    p = convert_tags_to_mnm(p)
                    outfile.write(w + '\t' + g + '\t' + p + '\n')
                else:
                    if w not in ['T_S', 'T_E', 'R_S', 'R_E', 'L_S', 'L_E', 'Y_S', 'Y_E', 'X_S', 'X_E', 'V_S', 'V_E']:
                        outfile.write(w+'\t'+g+'\t'+p+'\n')
            outfile.write('\n')
    outfile.close()

for exp in ['joint_head']:
    for split in [0, 1, 2, 3, 4]:
        try:
            command = "allennlp train experiments/head/"+exp+str(split)+".jsonnet --include-package my_library -s probing_exp/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError:
            command = "allennlp train experiments/head/"+exp+str(split)+".jsonnet --include-package my_library -s probing_exp/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        try:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz /home/nlp/pyatkiv/workspace/Modality/data/"+str(split)+"/dev_prejacent_five_head.txt --include-package my_library --use-dataset-reader --cuda-device 0 --output-file probing_predictions2/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz /home/nlp/pyatkiv/workspace/Modality/data/"+str(split)+"/dev_prejacent_five_head.txt --include-package my_library --use-dataset-reader --cuda-device 0 --output-file probing_predictions2/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        try:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz /home/nlp/pyatkiv/workspace/Modality/data/full/test_prejacent_five_head.txt --include-package my_library --use-dataset-reader --cuda-device 0 --output-file probing_predictions2/test_"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz /home/nlp/pyatkiv/workspace/Modality/data/full/test_prejacent_five_head.txt --include-package my_library --use-dataset-reader --cuda-device 0 --output-file probing_predictions2/test_"+exp+str(split)
            subprocess.run(command, check=True, shell=True)

        pred_file = "probing_predictions2/"+exp+str(split)
        outfile = "probing_predictions2/readable"+exp+str(split)
        convert_to_eval_format(
            pred_file,
            outfile,
            bottom_up=False)
        pred_file = "probing_predictions2/test_"+exp+str(split)
        outfile = "probing_predictions2/readable_test_"+exp+str(split)
        convert_to_eval_format(
            pred_file,
            outfile,
            bottom_up=False)
