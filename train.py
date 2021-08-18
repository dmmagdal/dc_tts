# train.py
# author: Diego Magdaleno
# Instantiate and train a Graph object from model.py. Trains both the
# Text2Mel and SSRN models in the Graph.
# Python 3.7
# Tensorflow 2.4.0


import math
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Model
from hyperparams import Hyperparams as hp
from layers import *
from utils import *
from data_load import load_vocab, get_batch, load_data
from model import Text2MelModel, SSRNModel, Graph, SavePoint, TTSGraph
from datetime import datetime


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


# Pull data.
data, num_batch = get_batch()

# Initialize model callbacks.
early_stop = tf.keras.callbacks.EarlyStopping(monitor="mae", patience=3)
text2mel_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	"./tmp/text2mel_test_chkpt", monitor="mae", save_best_only=True,
	save_freq="epoch"
)
ssrn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	"./tmp/ssrn_test_chkpt", monitor="mae", save_best_only=True,
	save_freq="epoch"
)
#text2mel_savepoint = SavePoint("./tmp/text2mel_custom_savepoint")

# DC_TTS paper shows model was trained on the following number of
# iterations:
# 20-40K -> ~2 hours -> 1.74 +- 0.52 MOS
# 90-150K -> ~7 hours -> 2.63 +- 0.75 MOS
# 200-340K -> ~17 hours -> 2.71 +- 0.66 MOS
# 540-900K -> ~40 hours -> 2.61 +- 0.62 MOS
num_iterations = 20000
num_iterations = 340000
epochs = iterations_to_epochs(num_iterations, num_batch)
#epochs = 10

'''
# Initialize and compile models.
text2mel = Text2MelModel()
text2mel.compile(
	optimizer=tf.keras.optimizers.Adam(lr=hp.lr), 
	metrics=["accuracy", "mae"],
)
ssrn = SSRNModel()
ssrn.compile(
	optimizer=tf.keras.optimizers.Adam(lr=hp.lr), 
	metrics=["accuracy", "mae"],
)

# Build the models.
text2mel.build(
	input_shape=[(None, None,), (None, None, hp.n_mels)]
)
ssrn.build(
	input_shape=(None, None, hp.n_mels)
)

# Print summary of models.
text2mel.summary()
ssrn.summary()

# Train and save models.
start = datetime.now()
text2mel.fit(
	data, epochs=epochs,
	callbacks=[early_stop, text2mel_checkpoint],
)
text2mel.save("./text2mel_test")
print("Time to train Text2Mel {}".format(datetime.now() - start))
start2 = datetime.now()
ssrn.fit(
	data, epochs=epochs,
	callbacks=[early_stop, ssrn_checkpoint]
	)
ssrn.save("./ssrn_test")
print("Time to train SSRN {}".format(datetime.now() - start2))
print("Time to train all models {}".format(datetime.now() - start))
'''

#'''
# Initialize a graph object that contains both the Text2Mel and SSRN
# models for streamlined training and inference.
graph = TTSGraph("original_dc_tts_graph")
#graph = Graph("dc_tts_graph")
#graph = Graph()
#graph.train((data, num_batch))

graph.load()
graph.train((data, num_batch), num_iterations=num_iterations)
graph.save()

# Pull texts from the harvard sentences text file and synthesize on
# those texts.
text = load_data("synthesize")
graph.inference(text)
#'''