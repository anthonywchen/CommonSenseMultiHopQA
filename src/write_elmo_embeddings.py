""" Generates ELMo word representations from a vocabulary file """
import tensorflow as tf
import sys

sys.path.append('/home/tony/CommonSenseMultiHopQA/src/elmo/')
from elmo.model import dump_token_embeddings

flags = tf.app.flags
flags.DEFINE_string("elmo_options_file", "lm_data/nqa/elmo_2x4096_512_2048cnn_2xhighway_options.json", "ELMo options file")
flags.DEFINE_string("elmo_weight_file", "lm_data/nqa/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5", "ELMo weights file")
flags.DEFINE_string("hdf5_output_file", None, 'ELMo word representations written out in hdf5 format')
flags.DEFINE_string("vocab_file", None, 'vocab file')

def main(config):
	vocab_file = config.vocab_file
	elmo_options_file = config.elmo_options_file
	elmo_weight_file = config.elmo_weight_file
	hdf5_output_file = config.hdf5_output_file

	dump_token_embeddings(vocab_file, elmo_options_file, elmo_weight_file, hdf5_output_file)

if __name__ == '__main__':
	config = flags.FLAGS
	main(config)