import os
import subprocess
import codecs
from collections import defaultdict
import jsonlines
import spacy
import csv

nlp = spacy.load("en_core_web_sm")

def get_char_to_orig_token_map(sentence):
    #char : orig_token_idx
    char_tok_map = defaultdict(lambda : 0)
    char_idx = 0
    for word_idx, word in enumerate(sentence.split(' ')):
        for i in range(len(word)):
            char_tok_map[char_idx+i] = word_idx
        char_idx += len(word)+1
    return char_tok_map

def get_sent_map(sent_file):
    # sent_id : [sent, labels]
    sent_map = defaultdict(lambda : [])
    infile = codecs.open(sent_file, 'r')
    for line_numb, line in enumerate(infile.readlines()):
        words = []
        labels = []
        for token in line.split():
            word, label = token.split('###')
            words.append(word)
            labels.append(label)
        sent_map[line_numb].append(words)
        sent_map[line_numb].append(labels)
    return sent_map


def convert_prediction(sent_file, pred_file, outfile):
    sent_map = get_sent_map(sent_file)
    prediction_file = jsonlines.open(pred_file)
    outfile = csv.writer(open(outfile, 'w'), delimiter='\t')
    outfile.writerow(["sentence_id", "token", "lemma", "predicted_modal", "gold_modal", "coarse_pos", "fine_pos", "probabilites"])
    # sent_id : [{gold_label, predicted_label, indices, probs}]
    pred_dict = defaultdict(lambda : [])
    for row in prediction_file:
        sent_id = row['sent_id']
        gold_label = row["gold_label"]
        predicted_label = row["predicted_label"]
        indices = row["indices"]
        probs = row["probs"]
        pred_dict[sent_id].append({"gold_label": gold_label, "predicted_label": predicted_label, 'indices': indices, "probs": probs})
    for sent_id, value in pred_dict.items():
        sent_entry = sent_map[sent_id]
        sentence = ' '.join(sent_entry[0])
        char_tok_map = get_char_to_orig_token_map(sentence)
        doc = nlp(sentence)
        coarse_pos_list = []
        fine_pos_list = []
        lemma_list = []
        token_list = []
        orig_char_list = []
        for token in doc:
            orig_char_list.append(token.idx)
            coarse_pos_list.append(token.pos_)
            fine_pos_list.append(token.tag_)
            lemma_list.append(token.lemma_)
            token_list.append(token.text)
        tok_tok_map = defaultdict(lambda : 0)
        for tok_id, char in enumerate(orig_char_list):
            tok_tok_map[char_tok_map[char]] = tok_id

        out_gold_modal = ['O']*len(token_list)
        out_pred_modal = ['O']*len(token_list)
        out_probs = ['O']*len(token_list)

        for entry in value:
            gold_label = entry['gold_label']
            predicted_label = entry["predicted_label"]
            probs = [str(prob) for prob in entry["probs"]]
            indices = [int(ind) for ind in entry["indices"].split()]
            start_index = tok_tok_map[indices[0]]
            end_index = tok_tok_map[indices[-1]]
            if start_index==end_index:
                out_gold_modal[start_index] = 'S-'+gold_label
                out_pred_modal[start_index] = 'S-'+predicted_label
                out_probs[start_index] = ' '.join(probs)
            else:
                out_gold_modal[start_index] = 'B-'+gold_label
                out_pred_modal[start_index] = 'B-'+predicted_label
                out_probs[start_index] = ' '.join(probs)
                out_gold_modal[end_index] = 'E-'+gold_label
                out_pred_modal[end_index] = 'E-'+predicted_label
                out_probs[end_index] = ' '.join(probs)
                if end_index-start_index>1:
                    for i in range(start_index+1, end_index):
                        out_gold_modal[i] = 'I-' + gold_label
                        out_pred_modal[i] = 'I-' + predicted_label
                        out_probs[i] = ' '.join(probs)

        outfile.writerow(["", "", "", "", "", "", "",
                          ""])
        for i in range(len(token_list)):
            outfile.writerow([sent_id, token_list[i], lemma_list[i], out_pred_modal[i], out_gold_modal[i], coarse_pos_list[i], fine_pos_list[i], out_probs[i]])

for exp in ['full_classifier']:
    for split in [0, 1, 2, 3, 4]:
        try:
            command = "allennlp train experiments/probing/"+exp+str(split)+".jsonnet --include-package my_library -s probing_exp/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError:
            command = "allennlp train experiments/probing/"+exp+str(split)+".jsonnet --include-package my_library -s probing_exp/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        try:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz data/"+str(split)+"/dev_binary_space.txt --include-package my_library --use-dataset-reader --cuda-device 2 --output-file probing_predictions/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz data/"+str(split)+"/dev_binary_space.txt --include-package my_library --use-dataset-reader --cuda-device 2 --output-file probing_predictions/"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        try:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz data/full/test_binary_space.txt --include-package my_library --use-dataset-reader --cuda-device 2 --output-file probing_predictions/test_"+exp+str(split)
            subprocess.run(command, check=True, shell=True)
        except subprocess.CalledProcessError:
            command = "allennlp predict probing_exp/"+exp+str(split)+"/model.tar.gz data/full/test_binary_space.txt --include-package my_library --use-dataset-reader --cuda-device 2 --output-file probing_predictions/test_"+exp+str(split)
            subprocess.run(command, check=True, shell=True)

        sent_file = "data/"+str(split)+"/dev_binary_space.txt"
        pred_file = "probing_predictions/"+exp+str(split)
        outfile = "probing_predictions/readable"+exp+str(split)
        convert_prediction(sent_file, pred_file, outfile)

        sent_file = "data/full/test_binary_space.txt"
        pred_file = "probing_predictions/test_"+exp+str(split)
        outfile = "probing_predictions/readable_test_"+exp+str(split)
        convert_prediction(sent_file, pred_file, outfile)

