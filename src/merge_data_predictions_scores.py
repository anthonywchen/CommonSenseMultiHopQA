"""
	Merges the data files, prediction files, and scoring files (BLEU, ROUGE, etc) into one JSON file for 
	easy portability and manipulation
"""

import argparse
from json import dumps
from jsonlines import Reader
from os.path import join

def load_text_file(filename):
	lines = []
	with open(filename, 'r') as f:
		for l in f:
			lines.append(l.strip())

	return lines

def load_json_file(filename):
	lines = []
	for l in Reader(open(filename)):
		lines.append(l)
	return lines

def main(data_file, prediction_file, bleu1_file, bleu4_file, rouge_file, meteor_file, cider_file, bert_score_file, paraphrase_score_file):
	# Load contents of the argument files
	data_lines = load_json_file(data_file)
	prediction_lines = load_text_file(prediction_file)
	bleu1_lines = load_text_file(bleu1_file)
	bleu4_lines = load_text_file(bleu4_file)
	rouge_lines = load_text_file(rouge_file)
	meteor_lines = load_text_file(meteor_file)
	cider_lines = load_text_file(cider_file)
	bert_score_lines = load_text_file(bert_score_file)
	paraphrase_score_lines = load_text_file(paraphrase_score_file)

	assert len(data_lines) == len(prediction_lines)
	assert len(data_lines) == len(bleu1_lines)
	assert len(data_lines) == len(bleu1_lines)
	assert len(data_lines) == len(bleu4_lines)
	assert len(data_lines) == len(rouge_lines)
	assert len(data_lines) == len(meteor_lines)
	assert len(data_lines) == len(cider_lines)
	assert len(data_lines) == len(bert_score_lines)
	assert len(data_lines) == len(paraphrase_score_lines)

	#############################
	### For each data point, merge the predictions and scores with the data dictionary and write 
	### to a file in the same directory as the predictions
	#############################
	output_file = join('/'.join(prediction_file.split('/')[:-1]), data_file.split('/')[-1] + '.merged')

	with open(output_file, 'w') as writer:
		# Iterate through data points
		for data_num in range(len(data_lines)):
			data_dict = data_lines[data_num]
			data_dict['pred'] = prediction_lines[data_num]
			data_dict['bleu1'] = bleu1_lines[data_num]
			data_dict['bleu4'] = bleu4_lines[data_num]
			data_dict['rouge'] = rouge_lines[data_num]
			data_dict['meteor'] = meteor_lines[data_num]
			data_dict['cider'] = cider_lines[data_num]
			data_dict['bert-score'] = bert_score_lines[data_num]
			data_dict['paraphrase-score'] = paraphrase_score_lines[data_num]

			# Write the updated data dict to file
			writer.write(dumps(data_dict) + '\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_file', type=str, required=True, help='path to processed data file')
	parser.add_argument('--prediction_file', type=str, required=True, help='path to predicted answers')

	parser.add_argument('--bleu1_file', type=str, required=True, default=None, help='path to file of BLEU1 scores')
	parser.add_argument('--bleu4_file', type=str, required=True, default=None, help='path to file of BLEU4 scores')
	parser.add_argument('--rouge_file', type=str, required=True, default=None, help='path to file of ROUGE-L scores')
	parser.add_argument('--meteor_file', type=str, required=True, default=None, help='path to file of METEOR scores')
	parser.add_argument('--cider_file', type=str, required=True, default=None, help='path to file of CIDEr scores')
	parser.add_argument('--bert_score_file', type=str, required=True, default=None, help='path to file of BERTScore scores')
	parser.add_argument('--paraphrase_score_file', type=str, required=True, default=None, help='path to file of paraphrasing scores')

	args = parser.parse_args()
	main(args.data_file, args.prediction_file, args.bleu1_file, args.bleu4_file, args.rouge_file, \
		 args.meteor_file, args.cider_file, args.bert_score_file, args.paraphrase_score_file)