# test_data_load.py


import numpy as np
import tensorflow as tf
from hyperparams import Hyperparams as hp
from data_load import *
from model import Text2Mel, GraphModel


#data1, data2, data3, = get_batch()
data3 = get_batch()

z = list(data3.as_numpy_iterator())
print(len(z))
print(z[0])

for step, (fname, text, mel, mag) in enumerate(data3):
	print("Step {}: {} {} {} {}".format(step, fname, text, mel, mag))
	print(mel.get_shape())
	print(step)
	if step % 10 == 0 and step > 0:
		break

#exit()
graph = GraphModel()
graph.train_model("")
graph.save_model(". q")


'''
text2mel = Text2Mel()
text2mel.compile(optimizer=tf.keras.optimizers.Adam(lr=hp.lr))
text2mel.fit(data3, batch_size=hp.B, epochs=hp.num_iterations)
'''