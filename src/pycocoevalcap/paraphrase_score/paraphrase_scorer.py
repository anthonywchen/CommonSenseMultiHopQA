from allennlp.predictors.predictor import Predictor
import argparse
import sys

def compute_reference_scores(paraphrase_predictor, candidates_file, references_file):
	""" Computes the BERTScore with respect to a single references file """

	# Load in candidates 
	cands = []
	with open(candidates_file) as f:
		for line in f:
			cands.append(line.strip())

	# Load in references
	refs = []
	with open(references_file) as f:
		for line in f:
			refs.append(line.strip()) 

	# Check that number of candidates matches number of references
	assert len(cands) == len(refs)

	# Compute paraphrasing score
	scores = []
	for c, r in zip(cands, refs):
		input_dict = {'sentence1': c, 'sentence2': r}
		output_dict = paraphrase_predictor.predict_json(input_dict)
		scores.append(output_dict['class_probabilities'])
	return scores

def paraphrase_scorer(paraphrase_predictor, candidates_file, references_file1, references_file2=None):
	""" Computes the BERTScore with respect to each reference file 
		and merges the scores from the two reference files by doing a argmax operation.
		The scores are then written to file under the name ```<candidates_file>.bert_score```
	"""
	# Get paraphrasing score with respect to the first reference file
	cands_refs1_scores = compute_reference_scores(paraphrase_predictor, candidates_file, references_file1)

	# Get paraphrasing score with respect to second reference file if second reference file exists
	if references_file2:
		cands_refs2_scores = compute_reference_scores(paraphrase_predictor, candidates_file, references_file2)
		assert len(cands_refs1_scores) == len(cands_refs2_scores)
		# Argmax over the scores from the two references file
		cands_scores = []
		print(cands_refs1_scores[:10])
		print(cands_refs2_scores[:10])
		for s1, s2 in zip(cands_refs1_scores, cands_refs2_scores):
		    cands_scores.append(max(s1, s2))
	else:
		cands_scores = cands_refs1_scores

	# Write the scores to file
	with open(candidates_file + '-paraphrase_score.txt', 'w') as f:
		for c in cands_scores:
			f.write(str(c) + '\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--paraphrase_identification_repo', type=str, required=True, help='path to Paraphrase-Identification repository')
	parser.add_argument('--paraphrase_archive', type=str, required=True, help='path to trained paraphrase omdel archive in Paraphrase-Identification repository')
	parser.add_argument('--candidates_file', type=str, required=True, help='path to file with generated answers')
	parser.add_argument('--references_file1', type=str, required=True, help='path to file of reference answers')
	parser.add_argument('--references_file2', type=str, required=False, default=None, help='path to file of (second) reference answers')
	parser.add_argument('--cuda_device', type=int, required=False, default=-1, help='CUDA device to use. -1 for CPU')

	args = parser.parse_args()

	# Load the pretrained paraphrase predictor first
	sys.path.append(args.paraphrase_identification_repo)
	import src
	paraphrase_predictor = Predictor.from_path(args.paraphrase_archive, 'paraphrase')
	if args.cuda_device >= 0:
		paraphrase_predictor._model.to(args.cuda_device)

	paraphrase_scorer(paraphrase_predictor, args.candidates_file, args.references_file1, args.references_file2)
