import argparse
import codecs
import csv

def convert_to_conll(input_file, output_file):
    input_file = codecs.open(input_file, 'r')
    output_file = codecs.open(output_file, 'w')
    output_file.write('sentence_id\ttoken\tgold_modal\n')
    for row_numb, row in enumerate(input_file.readlines()):
        row = row.split()
        for token in row:
            print(token)
            output_file.write(str(row_numb)+'\t'+'\t'.join(token.split('###'))+'\n')
        output_file.write('\n')
    output_file.close()

def convert_to_space(input_file, output_file):
    input_file = csv.DictReader(open(input_file), delimiter='\t')
    output_file = codecs.open(output_file, 'w')
    sentence = []
    prev_id = '0'
    for row in input_file:
        if row['sentence_id']!= prev_id:
            output_file.write(' '.join(sentence)+'\n')
            sentence = []
            prev_id = row['sentence_id']
        else:
            sentence.append(row['token']+'###'+row['gold_modal'])
            prev_id = row['sentence_id']



def main(args):
    if args.target_format=='conll':
        convert_to_conll(args.input, args.output)
    elif args.target_format=='space':
        convert_to_space(args.input, args.output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="input file")
    parser.add_argument("-o", "--output", required=True, type=str,
                        help="output file")
    parser.add_argument("--target_format", required=True, type=str,
                        help="choose either conll or space")
    main(parser.parse_args())