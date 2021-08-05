# test_data_load.py


import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import *
from model import Text2Mel, GraphModel, Text2MelModel, SSRNModel


gpus = tf.config.list_physical_devices("GPU")
if gpus:
	for gpu in gpus:
		tf.config.experimental.set_memory_growth(gpu, True)

#data1, data2, data3, = get_batch()
data3, num_batch = get_batch()

z = list(data3.as_numpy_iterator())
print(len(z))
print(z[0])

for step, (fname, text, mel, mag) in enumerate(data3):
	print("Step {}: {} {} {} {}".format(step, fname, text, mel, mag))
	print(mel.get_shape())
	print(step)
	if step % 10 == 0 and step > 0:
		break

early_stop = tf.keras.callbacks.EarlyStopping(monitor="mae", patience=3)
text2mel_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	"./tmp/text2mel_test_chkpt", monitor="mae", save_best_only=True
)
ssrn_checkpoint = tf.keras.callbacks.ModelCheckpoint(
	"./tmp/ssrn_test_chkpt", monitor="mae", save_best_only=True
)
#text2mel = Text2Mel()
text2mel = Text2MelModel()
text2mel.compile(
	optimizer=tf.keras.optimizers.Adam(lr=hp.lr), 
	metrics=["accuracy", "mae"],
)
text2mel.fit(
	data3, epochs=10,
	callbacks=[early_stop, text2mel_checkpoint]
)
text2mel.save("./text2mel_test")

ssrn = SSRNModel()
ssrn.compile(
	optimizer=tf.keras.optimizers.Adam(lr=hp.lr), 
	metrics=["accuracy", "mae"],
)
ssrn.fit(
	data3, epochs=10,
	callbacks=[early_stop, ssrn_checkpoint]
	)
ssrn.save("./ssrn_test")