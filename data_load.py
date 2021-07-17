# data_load.py
# author: Diego Magdaleno
# Copy of the data_load.py file in the Tensorflow implementation of
# dc_tts.
# Python 3.7
# Tensorflow 2.4.0


from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata


def load_vocab():
	char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
	idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
	return char2idx, idx2char


def text_normalize(text):
	text = "".join(char for char in unicodedata.normalize("NFD", text)
		if unicodedata.category(char) != "Mn"
	) # strip accents.
	text = text.lower()
	text = re.sub("[^{}]".format(hp.vocab), " ", text)
	text = re.sub("[ ]+", " ", text)
	return text


# Load data.
# @param: mode, "train" or "synthesize".
def load_data(mode="train"):
	# Load vocabulary.
	char2idx, idx2char = load_vocab()

	if mode == "train":
		if "LJ" in hp.data: # LJ Speech Dataset.
			# Parse.
			fpaths, text_lengths, texts = [], [], []
			transcript = os.path.join(hp.data, "transcript.csv")
			lines = codecs.open(transcript, "r", "utf-8").readlines()
			for line in lines:
				fname, _, text = line.strip().split("|")

				fpath = os.path.join(hp.data, "wavs", fname + ".wav")
				fpaths.append(fpath)

				text = text_normalize(text) + "E" # E: EOS
				text = [char2idx[char] for char in text]
				text_lengths.append(len(text))
				texts.append(np.array(text, np.int32).tostring())

			return fpaths, text_lengths, texts
		else: # Nick Offerman or Winslet audiobook(s) dataset.
			# Parse.
			fpaths, text_lengths, texts = [], [], []
			transcript = os.path.join(hp.data, "transcript.csv")
			lines = codecs.open(transcript, "r", "utf-8").readlines()
			for line in lines:
				fname, _, text, is_inside_quotes, duration = line.strip().split("|")
				duration = float(duration)
				if duration > 10.0:
					continue

				fpath = os.path.join(hp.data, fname)
				fpaths.append(fpath)

				text += "E" # E: EOS
				text = [char2idx[char] for char in text]
				text_lengths.append(len(text))
				text.append(np.array(text, np.int32).tostring())

			return fpaths, text_lengths, texts
	else: # Synthesize on unseen test text.
		# Parse.
		lines = codecs.open(hp.test_data, "r", "utf-8").readlines()[1:]
		sents = [text_normalize(line.split(" ", 1)[-1]).strip() +"E"
			for line in lines
		] # Text normalization, E: EOS
		texts = np.zeros((len(sents), hp.max_N), np.int32)
		for i, sent in enumerate(sents):
			texts[i, :len(sent)] = [char2idx[char] for char in sent]
		return texts


# Load training data and put them in queues.
def get_batch():
	# available_devices = tf.config.list_physical_devices()
	# with tf.device(available_devices[-1].name):
	# Load data.
	fpaths, text_lengths, texts = load_data() # list
	maxlen, minlen = max(text_lengths), min(text_lengths)

	# Calculate total batch count.
	num_batch = len(fpaths) // hp.B

	# Create Queues.
	'''
	# Deprecated from TF 1.0
	fpath, text_length, text = tf.train.slice_input_producer(
		[fpaths, text_lengths, texts], shuffle=True
	)
	'''
	fpath, text_length, text = tf.data.Dataset.from_tensor_slices(
		tuple([fpaths, text_lengths, texts])
	).shuffle(256)

	# Parse.
	text = tf.io.decode_raw(text, tf.int32) # (None, )

	if hp.prepro:
		def _load_spectrograms(fpath):
			fname = os.path.basename(fpath)
			mel = "mels/{}".format(fname.replace("wav", "npy"))
			mag = "mags/{}".format(fname.replace("wav", "npy"))
			return fname, np.load(mel), np.load(mag)

		fname, mel, mag = tf.py_function(
			_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32]
		)
	else:
		fname, mel, mag = tf.py_function(
			load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32]
		) # (None, n_mels)

	# Add shape information.
	fname.set_shape(())
	text.set_shape((None,))
	mel.set_shape((None, hp.n_mels))
	mag.set_shape((None, hp.n_fft // 2 + 1))

	# Batching.
	_, (texts, mels, mags, fnames) = tf.bucket_by_sequence_length(
		input_length=text_length,
		tensors=[text, mel, mag, fname],
		batch_size=hp.B,
		bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
		num_threads=8,
		capacity=hp.B * 4,
		dynamic_pad=True
	)

	return texts, mels, mags, fnames, num_batch