import argparse
from bert_score import score

def compute_reference_scores(candidates_file, references_file):
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

	# Compute BERTScore
	_, _, bert_scores = score(cands, refs, bert="bert-base-uncased", verbose=True)

	return bert_scores


def bert_scorer(candidates_file, references_file1, references_file2=None):
	""" Computes the BERTScore with respect to each reference file 
		and merges the scores from the two reference files by doing a argmax operation.
		The scores are then written to file under the name ```<candidates_file>.bert_score```
	"""

	# Get BERTScore with respect to first reference file
	cands_refs1_scores = compute_reference_scores(candidates_file, references_file1)

	# Get BERTScore with respect to second reference file if second reference file exists
	if references_file2:
		cands_refs2_scores = compute_reference_scores(candidates_file, references_file2)
		# Argmax over the scores from the two references file
		assert len(cands_refs1_scores) == len(cands_refs2_scores)
		cands_scores = []
		for s1, s2 in zip(cands_refs1_scores, cands_refs2_scores):
		    cands_scores.append(max(s1, s2))
	else:
		cands_scores = cands_refs1_scores

	# Write the scores to file
	with open(candidates_file + '-bert_score.txt', 'w') as f:
		for c in cands_scores:
			f.write(str(c.item()) + '\n')

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--candidates_file', type=str, required=True, help='path to file with generated answers')
	parser.add_argument('--references_file1', type=str, required=True, help='path to file of reference answers')
	parser.add_argument('--references_file2', type=str, required=False, default=None, help='path to file of (second) reference answers')

	args = parser.parse_args()

	bert_scorer(args.candidates_file, args.references_file1, args.references_file2)
