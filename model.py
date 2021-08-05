# model.py
# author: Diego Magdaleno
# Convert the Graph from train.py to an OOP implementation from the
# Tensorflow implementation of dc_tts
# Python 3.7
# Tensorflow 2.4.0


import os
import json
import sys
import math
from tqdm import tqdm
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.models import load_model
from layers import *
from utils import *
from hyperparams import Hyperparams as hp
from data_load import get_batch, load_vocab



#'''
# Configuration code for allowing GPU usage on Tensorflow 2. Comment
# out when running on Tensorflow 1 on CPU.
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
session = tf.compat.v1.Session(config=config)
#'''


# Function that converts the number of trainingsteps or iterations to
# epochs (rounded up).
def iterations_to_epochs(num_iterations, num_batch):
	# Formula:
	# 1 epoch = num_batch * batch_size (1 pass through the data).
	# 1 batch = 1 step (data of batch_size is passed through on each
	# step).
	# num_iterations steps * (1 batch/1 step) * (1 epoch/num_batch batch)
	# Formula is really num_iterations/num_batch == num_epochs
	epochs = num_iterations / num_batch

	return math.ceil(epochs)


class Graph:
	def __init__(self, graph_name="", input_hp=None):
		# Set graph name.
		if graph_name == "":
			graph_name = "graph"
		self.graph_name = graph_name

		# Set hyperparameters.
		if input_hp:
			self.hp = input_hp
		else:
			self.hp = hp

		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		# Initialize models.
		self.text2mel_model = Text2MelModel(self.hp)
		self.ssrn_model = SSRNModel(self.hp)

		# Initialize optimizers.
		text2mel_optimizer = keras.optimizers.Adam(lr=self.hp.lr)
		ssrn_optimizer = keras.optimizers.Adam(lr=self.hp.lr)

		# Compile models.
		self.text2mel_model.compile(
			optimizer=text2mel_optimizer, metrics=["accuracy", "mae"]
		)
		self.ssrn_model.compile(
			optimizer=ssrn_optimizer, metrics=["accuracy", "mae"]
		)

		# Build models (specify input shape(s)).
		# Text2Mel input shape = [text_shape, s_shape]
		# text_shape = (batch_size, None,)
		# s_shape = (batch_size, None, n_mels)
		# SSRN input_shape = mel_shape
		# mel_shape = (batch_size, None, n_mels)
		self.text2mel_model.build(
			input_shape=[(None, None,), (None, None, self.hp.n_mels)]
		)
		self.ssrn_model.build(
			input_shape=(None, None, self.hp.n_mels)
		)

		# Print summary of models.
		self.text2mel_model.summary()
		self.ssrn_model.summary()


	def call(self, inputs, training=False):
		pass


	def inference(self, text):
		# Unpackage the data.
		pass


	def train(self, data_batch, model=0, epochs=1, num_iterations=None):
		# Unpackage the data and the number of batches.
		data, num_batch = data_batch

		# Determine which model(s) to train based on the value passed
		# in.
		train_text2mel = True
		train_ssrn = True
		if model == 1:
			train_ssrn = False
		elif model == 2:
			train_text2mel = False
		elif model not in [0, 1, 2]:
			print("Error: Select which model to train: " +\
				"[0] Text2Mel & SSRN , [1] Text2Mel only, [2] SSRN only."
			)
			return

		# Initialize callbacks.
		early_stop = keras.callbacks.EarlyStopping(
			monitor="mae", patience=3
		)
		text2mel_checkpoint = keras.callbacks.ModelCheckpoint(
			"./" + self.graph_name + "/text2mel/checkpoints/text2mel_chkpt", 
			monitor="mae", 
			save_best_only=True
		)
		ssrn_checkpoint = keras.callbacks.ModelCheckpoint(
			"./" + self.graph_name + "/ssrn/checkpoints/ssrn_chkpt", 
			monitor="mae", 
			save_best_only=True
		)

		# Calculate the number of epochs to train for if a value was
		# passed in for the number of iterations.
		if num_iterations:
			epochs = iterations_to_epochs(num_iterations, num_batch)

		# Train the model(s).
		if train_text2mel:
			print("Training {} Text2Mel...".format(self.graph_name))
			self.text2mel_model.fit(
				data,
				epochs=epochs, 
				callbacks=[early_stop, text2mel_checkpoint]
			)
			print("Finished training {} Text2Mel.".format(self.graph_name))
		if train_ssrn:
			print("Training {} SSRN...".format(self.graph_name))
			self.ssrn_model.fit(
				data,
				epochs=epochs, 
				callbacks=[early_stop, ssrn_checkpoint]
			)
			print("Finished training {} SSRN.".format(self.graph_name))

		return


	def save(self, save_path=".", h5=False):
		if h5:
			# Check if path exists.
			if not os.path.exists(save_path):
				os.makedirs(save_path)

			# Save path strings.
			graph_path = self.graph_name + "/"
			if not save_path.endswith("/"):
				graph_path = save_path + "/" + graph_path
			else:
				graph_path = save_path + graph_path
			h5_text2mel = graph_path + "text2mel/text2mel.h5"
			h5_ssrn = graph_path + "ssrn/ssrn.h5"
			

			# Save the models.
			self.text2mel_model.save(h5_text2mel)
			self.ssrn_model.save(h5_ssrn)
		else:
			# Save path strings.
			graph_path = self.graph_name + "/"
			if not save_path.endswith("/"):
				graph_path = save_path + "/" + graph_path
			else:
				graph_path = save_path + graph_path
			text2mel_save = graph_path + "text2mel"
			ssrn_save = graph_path + "ssrn"

			# Save the models.
			self.text2mel_model.save(text2mel_save)
			self.ssrn_model.save(ssrn_save)

		# Save the hyperparameters.
		hparam_path = graph_path + "hparams.json"
		hparams = {"graph_name": self.graph_name}
		hp_attrs = [dir(self.hp)]
		for attr in hp_attrs:
			if "__" not in attr:
				hparams.update({attr: getattr(self.hp, attr)})
		with open(hparam_path, "w+") as file:
			json.dump(hparams, file, indent=4)

		# Print summary of models loaded.
		self.text2mel_model.summary()
		self.ssrn_model.summary()
		return


	def load(self, save_path=".", h5=False):
		# Save path strings.
		graph_path = self.graph_name + "/"
		if not save_path.endswith("/"):
			graph_path = save_path + "/" + graph_path
		else:
			graph_path = save_path + graph_path
		hparam_path = graph_path + "hparams.json"

		# Check if paths exist.
		if not os.path.exists(save_path):
			print("Error: Could not detect path {}.".format(save_path))
			return
		elif not os.path.exists(graph_path):
			print("Error: Could not detect path {}.".format(graph_path))
			return
		elif not os.path.exists(hparam_path):
			print("Error: Could not detect file {}.".format(hparam_path))
			return

		if h5:
			# Model file save paths.
			h5_text2mel = graph_path + "text2mel/text2mel.h5"
			h5_ssrn = graph_path + "ssrn/ssrn.h5"
			
			# Check if model files exist.
			if not os.path.exists(h5_text2mel):
				print("Error: Could not detect file {}.".format(h5_text2mel))
				return
			if not os.path.exists(h5_ssrn):
				print("Error: Could not detect file {}.".format(h5_ssrn))
				return

			# Load the models.
			self.text2mel_model = load_model(h5_text2mel)
			self.ssrn_model = load_model(h5_ssrn)
		else:
			# Model file save paths.
			text2mel_save = graph_path + "text2mel"
			ssrn_save = graph_path + "ssrn"

			# Check if model files exist.
			if not os.path.exists(text2mel_save):
				print("Error: Could not detect path {}.".format(text2mel_save))
				return
			if not os.path.exists(ssrn_save):
				print("Error: Could not detect path {}.".format(ssrn_save))
				return
			
			# Load the models.
			self.text2mel_model = load_model(text2mel_save)
			self.ssrn_model = load_model(ssrn_save)

		# Load the hyperparameters.
		with open(hparam_path, "r") as file:
			hparams = json.load(file)
		self.graph_name = hparams["graph_name"]
		for attr in hparams:
			if attr == "graph_name":
				continue
			setattr(self.hp, attr, hparams[attr])

		return


	def rename(self, new_name):
		self.graph_name = new_name


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


	def call(self, inputs, training=False):
		text, s = inputs
		key, value = self.textEnc(text, training=training)
		query = self.audioEnc(s, training=training)
		r, alignments, max_attentions = self.attention((query, key, value), 
			self.prev_max_attention,
			training=training
		)
		y_logits, y = self.audioDec(r, training=training)

		self.prev_max_attention = max_attentions
		return y, y_logits, alignments, max_attentions


	@tf.function
	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		fnames, texts, mels, mags = data

		with tf.GradientTape() as tape:
			# Compute s.
			s = tf.concat(
				(tf.zeros_like(mels[:, :1, :]), mels[:, :-1, :]), 1
			)

			# Feed forward in training mode.
			y_pred, y_pred_logits, alignments, max_attentions = self((texts, s),
				training=True
			)

			# Mel L1 loss.
			loss_mels = tf.reduce_mean(tf.abs(y_pred - mels))

			# Mel binary divergence loss.
			loss_bd1 = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					logits=y_pred_logits,
					labels=mels
				)
			)

			# Guided attention loss.
			A = tf.pad(alignments, 
				[(0, 0), (0, self.hp.max_N), (0, self.hp.max_T)], mode="CONSTANT",
				constant_values=-1.0
			)[:, :self.hp.max_N, :self.hp.max_T]
			#attention_masks = tf.to_float(tf.not_equal(A, -1))
			attention_masks = tf.cast(tf.not_equal(A, -1), tf.float32)
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
		self.compiled_metrics.update_state(mels, y_pred)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}


class SSRNModel(Model):
	def __init__(self, input_hp=None):
		super(SSRNModel, self).__init__()

		if input_hp is not None:
			self.hp = input_hp
		else:
			self.hp = hp

		self.ssrn = SSRN(self.hp, input_shape=(None, hp.n_mels))


	def call(self, inputs, training=False):
		mel = inputs

		z_logits, z = self.ssrn(mel, training=training)

		return z, z_logits


	@tf.function
	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		fnames, texts, mels, mags = data

		with tf.GradientTape() as tape:
			# Feed forward in training mode.
			z_pred, z_pred_logits = self(mels, training=True)

			# Mag L1 loss.
			loss_mags = tf.reduce_mean(tf.abs(z_pred - mags))

			# Mag binary divergence loss.
			loss_bd2 = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					logits=z_pred_logits,
					labels=mags
				)
			)

			# Total loss.
			loss = loss_mags + loss_bd2

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metrics that tracks the loss).
		self.compiled_metrics.update_state(mags, z_pred)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}


class GraphModel:
	def __init__(self, input_hp=None):
		# Load vocabulary.
		self.char2idx, self.idx2char = load_vocab()

		# Hyperparameters.
		self.hp = hp if input_hp is None else input_hp

		# Guided attention step.
		self.gts = tf.convert_to_tensor(guided_attention())

		# Model layers.
		self.textEnc = TextEncoder(self.hp)
		self.audioEnc = AudioEncoder(self.hp)
		self.attention = Attention(self.hp)
		self.audioDec = AudioDecoder(self.hp)
		self.ssrn = SSRN(self.hp)

		# Instantiate models.
		self.create_model()


	def create_model(self):
		# Initialize all model inputs.
		text = tf.keras.Input(shape=(None,), dtype=tf.int32,
			batch_size=self.hp.B
		) # (N)
		mels = tf.keras.Input(shape=(None, 80), dtype=tf.float32,
			batch_size=self.hp.B
		) # (T/r, n_mels)
		self.prev_max_attention = tf.zeros(shape=(self.hp.B,), dtype=tf.int32)
		s = tf.keras.Input(shape=(None, 80), dtype=tf.float32, 
			batch_size=self.hp.B
		) # same shape as mel
		
		# Model layers.
		self.textEnc = TextEncoder(self.hp)
		self.audioEnc = AudioEncoder(self.hp)
		self.attention = Attention(self.hp)
		self.audioDec = AudioDecoder(self.hp)
		self.ssrn = SSRN(self.hp)

		# Optimizer.
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.hp.lr)
		
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
		self.text2mel_model.compile(
			optimizer=self.optimizer, loss=self.text2mel_loss
		)
		self.text2mel_model.summary()

		# Create and compile SSRN model.
		z_logits, z = self.ssrn(mels, training=False)
		self.ssrn_model = tf.keras.Model(inputs=[mels], outputs=[z_logits, z], 
			name="ssrn"
		)
		self.ssrn_model.compile(
			optimizer=self.optimizer, loss=self.ssrn_loss
		)
		self.ssrn_model.summary()


	def save_model(self, path_to_model_folder):
		text2mel_folder = path_to_model_folder + "/text2mel"
		ssrn_folder = path_to_model_folder + "/ssrn_model"

		self.text2mel_model.save(text2mel_folder)
		self.ssrn_model.save(ssrn_folder)
		pass


	def load_model(self, path_to_model_folder):
		# Check for the existance of the path specified along with the
		# hyperparameters json file and the model's h5 model. Print an
		# error message and return the function if any of the files or
		# the folder don't exist.
		text2mel_folder = path_to_model_folder + "/text2mel"
		ssrn_folder = path_to_model_folder + "/ssrn_model"
		
		text2mel_hparams_file = text2mel_folder +"/hparams.json"
		ssrn_hparams_file = ssrn_folder + "/hparams.json"
		text2mel_model_file = text2mel_folder + "/model.h5"
		ssrn_model_file = ssrn_folder + "/model.h5"
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
		with open(text2mel_hparams_file, "r") as json_file:
			hparams = json.load(json_file)
		self.hp = hparams["hp"]
		self.optimizer = "adam" if hparams["optimizer"] == "" else hparams["optimizer"]
		self.loss = "sparse_categorical_crossentropy" if hparams["loss"] == "" else hparams["loss"]
		self.metrics = "accuracy" if hparams["metrics"] == "" else hparams["accuracy"]
		self.text2mel_model = load_model(text2mel_model_file, 
			custom_objects={
				"TextEncoder": TextEncoder,
				"AudioEncoder": AudioEncoder, 
				"Attention": Attention,
				"AudioDecoder": AudioDecoder}
		)
		self.ssrn_model = load_model(ssrn_model_file,
			custom_objects={
				"SSRN": SSRN
			}
		)
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
		self.prev_max_attention = tf.zeros(shape=(self.hp.B,), dtype=tf.int32)

		# Begin custom training loop.
		global_step = 0
		epoch = 0
		while global_step <= hp.num_iterations:
			#print("\nStart of Epoch {} of {}\n".format(iteration + 1, self.hp.num_iterations))
			print("\nStart of Epoch {}".format(epoch))

			# Iterate over batches of the dataset (batch_size = 1 for
			# now).
			for step, (fname, text, mel, mag) in enumerate(dataset):
				print(step)
				print(mel.get_shape())
				# Compute s from mel.
				s = tf.concat(
					(tf.zeros_like(mel[:, :1, :]), mel[:, :-1, :]), 1
				)

				# Open a GradientTape to record the operations run
				# during a forward pass, which enables
				# auto-differentiation.
				with tf.GradientTape() as tape:
					# Run a forward pass of the text2mel model. The
					# operations that model applies to its inputs are
					# going to be recorded on the GradientTape.
					y_logits, y, alignments, max_attentions = self.text2mel_model(
						#(text, mel, self.prev_max_attention, s), training=True
						(text, s), training=True
					)

					# Set prev_max_attention to the newly output
					# max_attention.
					self.prev_max_attention = max_attentions

					# Compute the loss value for this minibatch.
					loss_value = self.text2mel_loss((text, s), mel, True)

				# Use the gradient tape to automatically retrieve the
				# gradients of the trainable variables with respect to
				# the loss.
				grads = tape.gradient(
					loss_value, self.text2mel_model.trainable_weights
				)

				# Run one step of gradient descent by updating the
				# value of the variables to minimize the loss.
				self.optimizer.apply_gradients(
					zip(grads, self.text2mel_model.trainable_weights)
				)

				# Log every 1000 batches.
				if step % 1000 == 0:
					print(
						"Training loss (for one batch) at step {}: {}".format(
							step, float(loss_value)
						)
					)

				global_step += 1
			epoch += 1

		return


	def inference(self, inputs, model=1):
		# Unpack dataset.

		# Reset prev_max_attention.
		self.prev_max_attention = tf.zeros(shape=(self.hp.B,), dtype=tf.int32)
		pass


class Text2MelModel(Model):
	def __init__(self, input_hp=None):
		super(Text2MelModel, self).__init__()

		if input_hp is not None:
			self.hp = input_hp
		else:
			self.hp = hp

		# https://stackoverflow.com/questions/64455531/
		# multi-input-modeling-with-model-sub-classing-api-in-
		# tf-keras
		self.textEnc = TextEncoder(self.hp, input_shape=(None,))
		self.audioEnc = AudioEncoder(self.hp)
		self.attention = layers.AdditiveAttention()
		self.audioDec = AudioDecoder(self.hp)


	def call(self, inputs, training=False):
		text, s = inputs
		key, value = self.textEnc(text, training=training)
		query = self.audioEnc(s, training=training)
		r, attention_scores = self.attention([query, key, value],
			training=training, return_attention_scores=True
		)
		y_logits, y = self.audioDec(r, training=training)

		return y, y_logits


	@tf.function
	def train_step(self, data):
		# Unpack data. Structure depends on the model and on what was
		# passed to fit().
		fnames, texts, mels, mags = data

		with tf.GradientTape() as tape:
			# Compute s.
			s = tf.concat(
				(tf.zeros_like(mels[:, :1, :]), mels[:, :-1, :]), 1
			)

			# Feed forward in training mode.
			y_pred, y_pred_logits, = self((texts, s), training=True)

			# Mel L1 loss.
			loss_mels = tf.reduce_mean(tf.abs(y_pred - mels))

			# Mel binary divergence loss.
			loss_bd1 = tf.reduce_mean(
				tf.nn.sigmoid_cross_entropy_with_logits(
					logits=y_pred_logits,
					labels=mels
				)
			)

			# Total loss.
			loss = loss_mels + loss_bd1

		# Compute gradients.
		trainable_vars = self.trainable_variables
		gradients = tape.gradient(loss, trainable_vars)

		# Update weights.
		self.optimizer.apply_gradients(zip(gradients, trainable_vars))

		# Update metrics (includes the metrics that tracks the loss).
		self.compiled_metrics.update_state(mels, y_pred)

		# Return a dict mapping metric names to current value.
		return {m.name: m.result() for m in self.metrics}