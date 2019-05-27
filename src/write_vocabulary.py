import argparse
import tensorflow as tf

from read_data import *

flags = tf.app.flags
flags.DEFINE_string("processed_dataset_train", None, 'process dataset train')
flags.DEFINE_string("processed_dataset_valid", None, 'process dataset valid')
flags.DEFINE_string("output_vocabulary", None, 'path to output vocabulary file')
flags.DEFINE_string("min_occurance", 0, 'minimum number of occurances of a word to be included in the vocabulary')

def main(config):
	# Read in vocabulary from training data
	train_data = load_processed_dataset(config, 'train')
	train_vocab_freq = train_data.get_word_lists()

	# Read in vocabulary from validation data if it exists
	if config.processed_dataset_valid != None: 
		valid_data = load_processed_dataset(config, 'valid')
		valid_vocab_freq = valid_data.get_word_lists()

		for key in valid_vocab_freq:
			if key in train_vocab_freq:
				train_vocab_freq[key] += valid_vocab_freq[key]
			else:
				train_vocab_freq[key] = valid_vocab_freq[key]

	# Write vocabulary line by line with SOS, EOS, and UNK tokens
	with open(config.output_vocabulary, 'wb') as writer:
		writer.write('<S>\n</S>\n<UNK>\n')
		for key in train_vocab_freq:
			writer.write(key.encode("ascii", "ignore") + '\n')

if __name__ == '__main__':
	config = flags.FLAGS
	main(config)