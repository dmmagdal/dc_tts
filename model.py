# model.py
# author: Diego Magdaleno
# Convert the Graph from train.py to an OOP implementation from the
# Tensorflow implementation of dc_tts
# Python 3.7
# Tensorflow 2.4.0


import os
import json
import sys
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from layers import *
from utils import *
from hyperparams import Hyperparams as hp
from data_load import get_batch, load_vocab


class Graph:
	def __init__(self, num=1, mode="train"):
		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		# Set flag.
		training = True if mode == "train" else False

		# Graph.
		# Data feeding:
		# L: Text. (B, N), int32
		# mels: Reduced mel spectrogram. (B, T/r, n_mels), float32
		# mags: Magnitude. (B, T, n_fft // 2 + 1), float32
		if mode == "train":
			self.L, self.mels, self.mags, self.fnames, self.num_batch = get_batch()
			self.prev_max_attention = tf.ones(shape=(hp.B,), dtype=tf.int32)
			self.gts = tf.convert_to_tensor(guided_attention())
		else: # synthesize.
			# self.L = tf.placeholder(tf.int32, shape=(None, None))
			# self.mels = tf.placeholder(tf.float32, shape=(None, None, hp.n_mels))
			# self.prev_max_attention = tf.placeholder(tf.int32, shape=(None,))
			self.L = tf.keras.Input(shape=(None, None), dtype=tf.int32)
			self.mels = tf.keras.Input(
				shape=(None, None, hp.n_mels), dtype=tf.float32
			)
			self.prev_max_attention = tf.keras.Input(shape=(None,), dtype=tf.int32)

		if num == 1 or not training:
			# Get S or decoder inputs. (B, T//r, n_mels)
			self.S = tf.concat(
				(tf.zeros_like(self.mels[:, :1, :]), self.mels[:, :-1, :]), 1
			)

			# Networks.
			self.K, self.V = TextEncoder(hp, training=training)

			self.Q = AudioEncoder(hp, training=training)

			self.R, self.alignments, self.max_attentions = Attention(hp,
				monotonic_attention=(not training), 
				prev_max_attention=self.prev_max_attention
			)

			self.Y_logits, self.Y = AudioDecoder(hp, training=training)

		else:
			self.Z_logits, self.Z = SSRN(hp, training=training)

		if not training:
			self.Z_logits, self.Z = SSRN(hp, training=training)

		self.global_step = tf.Variable(0, trainable=False)

		if training:
			if num == 1: #Text2Mel.
				# Mel L1 loss.
				self.loss_mels = tf.reduce_mean(tf.abs(self.Y - self.mels))

				# Mel binary divergence loss.
				self.loss_bd1 = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Y_logits,
						labels=self.mels
					)
				)

				# Guided attention loss.
				self.A = tf.pad(self.alignments, 
					[(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT",
					constant_values=-1.0
				)[:, :hp.max_N, :hp.max_T]
				self.attention_masks = tf.to_float(tf.not_equal(self.A, -1))
				self.loss_att = tf.reduce_sum(
					tf.abs(self.A * self.gts) * self.attention_masks
				)
				self.mask_sum = tf.reduce_sum(self.attention_masks)
				self.loss_att /= self.mask_sum

				# Total loss.
				self.loss = self.loss_mels + self.loss_bd1 + self.loss_att
				
			else: # SSRN.
				# Mag L1 loss.
				self.loss_mags = tf.reduce_mean(tf.abs(self.Z - self.mags))

				# Mag binary divergence loss.
				self.loss_bd2 = tf.reduce_mean(
					tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Z_logits),
					labels=self.mags
				)

				# Total loss.
				self.loss = self.loss_mags + self.loss_bd2

			# Training scheme.
			self.lr = learning_rate_decay(hp.lr, self.global_step)
			self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)

			# Gradient clipping.
			self.gvs = self.optimizer.compute_gradients(self.loss)
			self.clipped = []
			for grad, var in self.gvs:
				grad = tf.clip_by_value(grad, -1.0, 1.0)
				self.clipped.append((grad, var))
				self.train_op = self.optimizer.apply_gradients(self.clipped,
					global_step=self.global_step
				)



		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp)
		self.audioDec = AudioDecoder(hp)
		self.ssrn = SSRN(hp)


	def create_model(self):
		# Initialize all model inputs.
		text = tf.keras.Input(shape=(None,), dtype=tf.int32,
			batch_size=hp
		) # (N)
		mels = tf.keras.Input(shape=(None, 80), dtype=tf.float32) # (T/r, n_mels)
		pma = tf.keras.Input(shape=(), dtype=tf.int32)
		s = tf.keras.Input(shape=(None, 80), dtype=tf.float32) # same shape as mel
		
		# Model layers.
		training = False
		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp, monotonic_attention=(not training), 
			prev_max_attention=pma
		)
		self.audioDec = AudioDecoder(hp)
		self.ssrn = SSRN(hp)
		
		# Create and compile Text2Mel model.
		k, v = self.textEnc(text, training=False)
		q = self.audioEnc(s, training=False)
		r, alignments, max_attentions = self.attention((q, k, v), training=False)
		y_logits, y = self.audioDec(r, training=False)
		self.text2mel_model = tf.keras.Model(inputs=[text, s], outputs=[y_logits, y], name="text2mel")
		self.text2mel_model.compile(optimizer="adam", loss=self.text2mel_loss)
		self.text2mel_model.summary()

		# Create and compile SSRN model.
		z_logits, z = self.ssrn(mels, training=False)
		self.ssrn_model = tf.keras.Model(inputs=[mels], outputs=[z_logits, z], name="ssrn")
		self.ssrn_model.compile(optimizer="adam", loss=self.ssrn_loss)
		self.ssrn_model.summary()


	def save_model(self, folder_path):
		pass


	def load_model(self, folder_path):
		pass


	def text2mel_loss(self):
		pass


	def ssrn_loss(self):
		pass


	def train_model(self, dataset_path, model=1):
		pass




def text2mel_loss():
	pass


class Text2Mel(Model):
	def __init__(self, input_hp=None):
		super(Text2Mel, self).__init__()

		if input_hp is not None:
			self.hp = input_hp
		else:
			self.hp = hp

		self.prev_max_attention = tf.zeros(shape=(self.hp.B,), dtype=tf.int32)

		self.gts = tf.convert_to_tensor(guided_attention())

		self.textEnc = TextEncoder(self.hp)
		self.audioEnc = AudioEncoder(self.hp)
		self.attention = Attention(self.hp)
		self.audioDec = AudioDecoder(self.hp)


	def call(self, text, s, training=False):
		key, value = self.textEnc(text, training=training)
		query = self.audioEnc(s, training=training)
		r, alignments, max_attentions = self.attention((query, key, value), 
			self.prev_max_attention,
			training=training
		)
		y_logits, y = self.audioDec(r, training=training)

		self.prev_max_attention = max_attentions
		return y, y_logits, alignments, max_attentions


	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		text, s = data

		with tf.GradientTape() as tape:
			# Feed forward in training mode.
			y_pred, y_pred_logits, alignments, max_attentions = self(text, s,
				training=True
			)

			# Mel L1 loss.
			loss_mels = tf.reduce_mean(tf.abs(y - mels))

			# Mel binary divergence loss.
			loss_bd1 = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits,
					labels=mels
				)
			)

			# Guided attention loss.
			A = tf.pad(alignments, 
				[(0, 0), (0, self.hp.max_N), (0, self.hp.max_T)], mode="CONSTANT",
				constant_values=-1.0
			)[:, :self.hp.max_N, :self.hp.max_T]
			attention_masks = tf.to_float(tf.not_equal(A, -1))
			loss_att = tf.reduce_sum(
				tf.abs(A * self.gts) * attention_masks
			)
			mask_sum = tf.reduce_sum(attention_masks)
			loss_att /= mask_sum

			# Total loss.
			loss = loss_mels + loss_bd1 + loss_att

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metrics that tracks the loss).
		self.compile_metrics.update_state(mels, y_pred)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}


class GraphModel:
	def __init__(self):
		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		# Model layers.
		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp)
		self.audioDec = AudioDecoder(hp)
		self.ssrn = SSRN(hp)

		# Instantiate models.
		self.create_model()


	def create_model(self):
		# Initialize all model inputs.
		text = tf.keras.Input(shape=(None,), dtype=tf.int32,
			batch_size=hp.B
		) # (N)
		mels = tf.keras.Input(shape=(None, 80), dtype=tf.float32,
			batch_size=hp.B
		) # (T/r, n_mels)
		self.prev_max_attention = tf.zeros(shape=(hp.B,), dtype=tf.int32)
		s = tf.keras.Input(shape=(None, 80), dtype=tf.float32, 
			batch_size=hp.B
		) # same shape as mel
		
		# Model layers.
		self.textEnc = TextEncoder(hp)
		self.audioEnc = AudioEncoder(hp)
		self.attention = Attention(hp)
		self.audioDec = AudioDecoder(hp)
		self.ssrn = SSRN(hp)
		
		# Create and compile Text2Mel model.
		k, v = self.textEnc(text, training=False)
		q = self.audioEnc(s, training=False)
		r, alignments, max_attentions = self.attention((q, k, v),
			self.prev_max_attention, training=False
		)
		y_logits, y = self.audioDec(r, training=False)
		self.text2mel_model = tf.keras.Model(
			inputs=[text, s], 
			outputs=[y_logits, y, alignments, max_attentions], name="text2mel"
		)
		self.text2mel_model.compile(optimizer="adam", loss=self.text2mel_loss)
		self.text2mel_model.summary()

		# Create and compile SSRN model.
		z_logits, z = self.ssrn(mels, training=False)
		self.ssrn_model = tf.keras.Model(inputs=[mels], outputs=[z_logits, z], name="ssrn")
		self.ssrn_model.compile(optimizer="adam", loss=self.ssrn_loss)
		self.ssrn_model.summary()


	def save_model(self, path_to_model_folder):
		pass


	def load_model(self, path_to_model_folder):
		# Check for the existance of the path specified along with the
		# hyperparameters json file and the model's h5 model. Print an
		# error message and return the function if any of the files or
		# the folder don't exist.
		text2mel_folder = path_to_model_folder + "/text2mel"
		ssrn_folder = path_to_model_folder + "/ssrn_model"
		
		text2mel_hparams_file = text2mel_folder +"/hparams.json"
		ssrn_hparams_file = ssrn_folder "/hparams.json"
		text2mel_model_file = text2mel_folder + "/model.h5"
		ssrn_model_file = ssrn_folder "/model.h5"
		if not os.path.exists(path_to_model_folder):
			print("Error: Path to folder does not exist.")
			return
		elif not os.path.exists(text2mel_hparams_file):
			print("Error: Hyperparameter file in path to text2mel folder does not exist.")
			return
		elif not os.path.exists(ssrn_hparams_file):
			print("Error: Hyperparameter file in path to ssrn folder does not exist.")
			return
		elif not os.path.exists(text2mel_model_file):
			print("Error: Model h5 file in path to text2mel folder does not exist.")
			return
		elif not os.path.exists(ssrn_model_file):
			print("Error: Model h5 file in path to ssrn folder does not exist.")
			return

		# Load the hyperparameters and model from file.
		'''
		with open(hparams_file, "r") as json_file:
			hparams = json.load(json_file)
		self.n_heads = hparams["n_heads"]
		self.n_layers = hparams["n_layers"]
		self.vocab_size = hparams["vocab_size"]
		self.embedding_size = hparams["embedding_size"]
		self.ff_dim = hparams["ff_dim"]
		self.dropout_rate = hparams["dropout_rate"]
		self.optimizer = "adam" if hparams["optimizer"] == "" else hparams["optimizer"]
		self.loss = "sparse_categorical_crossentropy" if hparams["loss"] == "" else hparams["loss"]
		self.metrics = "accuracy" if hparams["metrics"] == "" else hparams["accuracy"]
		self.gpt_model = load_model(h5_model_file, 
									custom_objects={"TokenAndPositionEmbedding": TokenAndPositionEmbedding,
													"DecoderBlock": DecoderBlock})
		'''

		# Return the function.
		return


	def text2mel_loss(self, inputs, outputs, training):
		text, s = inputs
		mels = outputs
		y_logits, y, alignments, max_attentions = self.text2mel_model(
			(text, s), training=training
		)

		# Mel L1 loss.
		loss_mels = tf.reduce_mean(tf.abs(y - mels))

		# Mel binary divergence loss.
		loss_bd1 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logits,
				labels=mels
			)
		)

		# Guided attention loss.
		A = tf.pad(alignments, 
			[(0, 0), (0, hp.max_N), (0, hp.max_T)], mode="CONSTANT",
			constant_values=-1.0
		)[:, :hp.max_N, :hp.max_T]
		attention_masks = tf.to_float(tf.not_equal(A, -1))
		loss_att = tf.reduce_sum(
			tf.abs(A * self.gts) * attention_masks
		)
		mask_sum = tf.reduce_sum(attention_masks)
		loss_att /= mask_sum

		# Total loss.
		loss = loss_mels + loss_bd1 + loss_att
		return loss


	def ssrn_loss(self, inputs, outputs, training):
		mels = inputs
		mags = outputs
		z_logits, z = sef.ssrn_model(mels, training=training)

		# Mag L1 loss.
		loss_mags = tf.reduce_mean(tf.abs(z - mags))

		# Mag binary divergence loss.
		self.loss_bd2 = tf.reduce_mean(
			tf.nn.sigmoid_cross_entropy_with_logits(logits=z_logits),
			labels=mags
		)

		# Total loss.
		loss = loss_mags + loss_bd2
		return loss


	def text2mel_grad(self, inputs, outputs):
		with tf.GradientTape() as tape:
			loss = self.text2mel_loss(inputs, outputs, training=True)
		return loss, tape.gradient(loss, self.text2mel_model.trainable_variables)


	def ssrn_grad(self, inputs, outputs):
		with tf.GradientTape() as tape:
			loss = self.ssrn_loss(inputs, outputs, training=True)
		return loss, tape.gradient(loss, self.ssrn_model.trainable_variables)


	def train_model(self, dataset_path, model=1):
		# Unpack dataset.
		dataset = get_batch()

		# Reset prev_max_attention.
		self.prev_max_attention = tf.zeros(shape=(hp.B,), dtype=tf.int32)
		pass


	def inference(self, inputs, model=1):
		# Unpack dataset.

		# Reset prev_max_attention.
		self.prev_max_attention = tf.zeros(shape=(hp.B,), dtype=tf.int32)
		pass
