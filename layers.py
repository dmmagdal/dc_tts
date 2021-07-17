# layers.py
# author: Diego Magdaleno
# OOP implementation of the modules.py and networks.py files in the
# Tensorflow implementation of dc_tts.
# Python 3.7
# Tensorflow 2.4.0


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class EmbeddingLayer(layers.Layer):
	def __init__(self, vocab_size, num_inputs, **kwargs):
		super(EmbeddingLayer, self).__init__()
		self.vocab_size = vocab_size
		self.num_inputs = num_inputs

		self.embedding_table = layers.Embedding(vocab_size, num_inputs)


	def call(self, inputs):
		return self.embedding_table(inputs)


class NormalizeLayer(layers.Layer):
	def __init__(self, **kwargs):
		super(NormalizeLayer, self).__init__()

		self.norm = layers.LayerNormalization()


	def call(self, inputs):
		return self.norm(inputs)


class HighwayNet(layers.Layer):
	def __init__(self, num_units, **kwargs):
		super(HighwayNet, self).__init__()

		self.num_units = num_units
		self.H = layers.Dense(num_units, activation="relu")
		self.T = layers.Dense(num_units, activation="sigmoid",
			bias_initializer=tf.constant_initializer(-1.0)
		)


	def call(self, inputs):
		h_outputs = self.H(inputs)
		t_outputs = self.T(inputs)
		outputs = h_outputs * t_outputs * (1.0 - t_outputs)
		return outputs


class Conv1DLayer(layers.Layer):
	def __init__(self, filters=None, size=1, rate=1, padding="same", 
			dropout_rate=0.0, use_bias=True, activation=None, **kwargs):
		super(Conv1DLayer, self).__init__()

		self.filters = filters
		self.size = size
		self.rate = rate
		self.padding = padding
		self.dropout_rate = dropout_rate
		self.use_bias = use_bias
		self.activation = activation
		self.pad_inputs = False
		self.pad_len = 0


	def build(self, input_shape):
		if self.padding.lower() == "causal":
			# Pre-padding for causality.
			self.pad_len = (self.size - 1) * self.rate # padding size.
			self.pad_inputs = True
			self.padding = "valid"

		if self.filters is None:
			self.filters = input_shape[-1]

		self.conv = layers.Conv1D(filters=self.filters, kernel_size=self.size, 
			dilation_rate=self.rate, padding=self.padding, use_bias=self.use_bias,
			kernel_initializer=keras.initializers.VarianceScaling()
		)
		self.norm = NormalizeLayer()
		if self.activation is not None:
			self.act = layers.Activation(self.activation)
		self.dropout = layers.Dropout(rate=self.dropout_rate)


	def call(self, inputs, training=True):
		if self.pad_inputs:
			inputs = tf.pad(inputs, [[0, 0], [self.pad_len, 0], [0, 0]])

		conv_output = self.conv(inputs)
		norm_output = self.norm(conv_output)
		if self.activation is not None:
			norm_output = self.act(norm_output)
		return self.dropout(norm_output, training=training)


class HC(layers.Layer):
	def __init__(self, filters=None, size=1, rate=1, padding="same", 
			dropout_rate=0.0, use_bias=True, activation=None, **kwargs):
		super(HC, self).__init__()

		self.filters = filters
		self.size = size
		self.rate = rate
		self.padding = padding
		self.dropout_rate = dropout_rate
		self.use_bias = use_bias
		self.activation = activation
		self.pad_inputs = False
		self.pad_len = 0


	def build(self, input_shape):
		if self.padding.lower() == "causal":
			# Pre-padding for causality.
			self.pad_len = (self.size - 1) * self.rate # padding size.
			self.pad_inputs = True
			self.padding = "valid"

		if self.filters is None:
			self.filters = input_shape[-1]

		self.conv = layers.Conv1D(filters=2 * self.filters, kernel_size=self.size, 
			dilation_rate=self.rate, padding=self.padding, use_bias=self.use_bias,
			kernel_initializer=keras.initializers.VarianceScaling()
		)
		self.norm1 = NormalizeLayer()
		self.norm2 = NormalizeLayer()
		self.sigmoid = layers.Activation("sigmoid")

		if self.activation is not None:
			self.activation = layers.Activation(self.activation)
		self.dropout = layers.Dropout(rate=self.dropout_rate)


	def call(self, inputs, training=True):
		# Store copy of inputs in case of padding.
		_inputs = inputs

		# If padding is required, then pad inputs.
		if self.pad_inputs:
			inputs = tf.pad(inputs, [[0, 0], [self.pad_len, 0], [0, 0]])

		conv_output = self.conv(inputs)
		H1, H2 = tf.split(conv_output, 2, axis=-1)

		h1_norm = self.norm1(H1)
		h2_norm = self.norm2(H2)

		h1_sigmoid = self.sigmoid(h1_norm)
		if self.activation is not None:
			h2_activation = self.activation(h2_norm)
		else:
			h2_activation = h2_norm

		dropout_input = h1_sigmoid * h2_activation + (1.0 - h1_sigmoid) * _inputs

		return self.dropout(dropout_input, training=training)


class Conv1DTransposeLayer(layers.Layer):
	def __init__(self, filters=None, size=3, stride=2, padding="same", 
			dropout_rate=0.0, use_bias=True, activation=None, **kwargs):
		super(Conv1DTransposeLayer, self).__init__()

		self.filters = filters
		self.size = size
		self.stride = stride
		self.padding = padding
		self.dropout_rate = dropout_rate
		self.use_bias = use_bias
		self.activation = activation
		self.pad_inputs = False
		self.pad_len = 0


	def build(self, input_shape):
		if self.filters is None:
			self.filters = input_shape[-1]

		self.conv = layers.Conv2DTranspose(filters=self.filters, 
			kernel_size=(1, self.size), strides=(1, self.stride), 
			padding=self.padding, activation=None, use_bias=self.use_bias,
			kernel_initializer=keras.initializers.VarianceScaling()
		)
		self.norm = NormalizeLayer()
		if self.activation is not None:
			self.activation = layers.Activation(self.activation)
		self.dropout = layers.Dropout(rate=self.dropout_rate)


	def call(self, inputs, training=True):
		inputs = tf.expand_dims(inputs, 1)
		conv_output = self.conv(inputs)
		conv_output = tf.squeeze(conv_output, 1)
		norm_output = self.norm(conv_output)
		if self.activation is not None:
			norm_output = self.activation(norm_output)
		return self.dropout(norm_output, training=training)


class TextEncoder(layers.Layer):
	def __init__(self, hp, **kwargs):
		super(TextEncoder, self).__init__()

		self.embed_layer = EmbeddingLayer(
			vocab_size=len(hp.vocab), num_inputs=hp.e
		)
		self.conv1 = Conv1DLayer(filters=2 * hp.d, size=1, rate=1, 
			dropout_rate=hp.dropout_rate, activation="relu"
		)
		self.conv2 = Conv1DLayer(size=1, rate=1, dropout_rate=hp.dropout_rate)

		self.hc1 = []
		for _ in range(2):
			for j in range(4):
				self.hc1.append(
					HC(size=3, rate=3 ** j, dropout_rate=hp.dropout_rate,
						activation=None
					)
				)

		self.hc2 = []
		for _ in range(2):
			self.hc2.append(
				HC(size=3, rate=1, dropout_rate=hp.dropout_rate, activation=None)
			)

		self.hc3 = []
		for _ in range(2):
			self.hc3.append(
				HC(size=1, rate=1, dropout_rate=hp.dropout_rate, activation=None)
			)
			

	def call(self, inputs, training=True):
		embed_output = self.embed_layer(inputs)
		conv1_output = self.conv1(embed_output, training=training)
		conv2_output = self.conv2(conv1_output, training=training)

		hc1_output = conv2_output
		for hc in self.hc1:
			hc1_output = hc(hc1_output, training=training)

		hc2_output = hc1_output
		for hc in self.hc2:
			hc2_output = hc(hc2_output, training=training)

		hc3_output = hc2_output
		for hc in self.hc3:
			hc3_output = hc(hc3_output, training=training)

		key, value = tf.split(hc3_output, 2, -1)
		return key, value


class AudioEncoder(layers.Layer):
	def __init__(self, hp, **kwargs):
		super(AudioEncoder, self).__init__()

		self.conv1 = Conv1DLayer(filters=hp.d, size=1, rate=1, padding="causal", 
			dropout_rate=hp.dropout_rate, activation="relu"
		)
		self.conv2 = Conv1DLayer(size=1, rate=1, padding="causal", 
			dropout_rate=hp.dropout_rate, activation="relu"
		)
		self.conv3 = Conv1DLayer(size=1, rate=1, padding="causal",
			dropout_rate=hp.dropout_rate
		)

		self.hc1 = []
		for _ in range(2):
			for j in range(4):
				self.hc1.append(
					HC(size=3, rate=3 ** j, padding="causal", 
						dropout_rate=hp.dropout_rate
					)
				)

		self.hc2 = []
		for _ in range(2):
			self.hc2.append(
				HC(size=3, rate=3, padding="causal", 
					dropout_rate=hp.dropout_rate
				)
			)


	def call(self, inputs, training=True):
		conv1_output = self.conv1(inputs, training=training)
		conv2_output = self.conv2(conv1_output, training=training)
		conv3_output = self.conv3(conv2_output, training=training)

		hc1_output = conv3_output
		for hc in self.hc1:
			hc1_output = hc(hc1_output, training=training)

		hc2_output = hc1_output
		for hc in self.hc2:
			hc2_output = hc(hc2_output, training=training)

		return hc2_output


class Attention(layers.Layer):
	#def __init__(self, hp, monotonic_attention=False, prev_max_attention=None, 
	#		**kwargs):
	#def __init__(self, hp, prev_max_attention=None, **kwargs):
	def __init__(self, hp, **kwargs):
		super(Attention, self).__init__()

		self.hp = hp
		#self.monotonic_attention = monotonic_attention
		#self.prev_max_attention = prev_max_attention

		self.softmax = layers.Softmax()


	'''
	def build(self, input_shape):
		print(input_shape)
		batch_size = input_shape[0][0]
		self.prev_max_attention = tf.zeros(shape=(batch_size,), 
			dtype=tf.int32
		)
	'''


	def call(self, inputs, prev_max_attention, training=True):
		query, key, value = inputs
		attention = tf.matmul(query, key, transpose_b=True) * \
			tf.math.rsqrt(tf.cast(self.hp.d, dtype=tf.float32))

		#if self.monotonic_attention: # For inference/not training
		if not training:
			key_masks = tf.sequence_mask(prev_max_attention, self.hp.max_N)
			reverse_masks = tf.sequence_mask(
				self.hp.max_N - self.hp.attention_win_size - prev_max_attention,
				self.hp.max_N
			)[:, ::-1]
			masks = tf.logical_or(key_masks, reverse_masks)
			masks = tf.tile(tf.expand_dims(masks, 1), [1, self.hp.max_T, 1])
			paddings = tf.ones_like(attention) * (-2 ** 32 + 1) # (B, T/r, N)
			attention = tf.where(tf.equal(masks, False), attention, paddings)
		attention_output = self.softmax(attention, training=training) # (B, T/r, N)
		max_attentions = tf.argmax(attention_output, -1) # (B, T/r)
		result = tf.matmul(attention_output, value)
		result = tf.concat((result, query), -1)

		alignments = tf.transpose(attention_output, [0, 2, 1]) # (B, N, T/r)

		return result, alignments, max_attentions


class AudioDecoder(layers.Layer):
	def __init__(self, hp, **kwargs):
		super(AudioDecoder, self).__init__()

		self.hp = hp		

		self.conv1 = Conv1DLayer( filters=hp.d, size=1, rate=1, padding="causal", 
			dropout_rate=self.hp.dropout_rate
		)

		self.hc1 = []
		for j in range(4):
			self.hc1.append(
				HC(size=3, rate=3 ** j, padding="causal",
					dropout_rate=self.hp.dropout_rate
				)
			)

		self.hc2 = []
		for _ in range(2):
			self.hc2.append(
				HC(size=3, rate=1, padding="causal", 
					dropout_rate=self.hp.dropout_rate
				)
			)

		self.conv_layers = []
		for _ in range(3):
			self.conv_layers.append(
				Conv1DLayer(size=1, rate=1, padding="causal", 
					dropout_rate=self.hp.dropout_rate, activation="relu"
				)
			)

		self.final_conv = Conv1DLayer(filters=self.hp.n_mels, size=1, rate=1, 
			padding="causal", dropout_rate=self.hp.dropout_rate
		)
		self.sigmoid = layers.Activation("sigmoid")


	def call(self, inputs, training=True):
		conv1_output = self.conv1(inputs, training=training)

		hc1_output = conv1_output
		for hc in self.hc1:
			hc1_output = hc(hc1_output, training=training)

		hc2_output = hc1_output
		for hc in self.hc2:
			hc2_output = hc(hc2_output, training=training)

		conv_layer_output = hc2_output
		for conv in self.conv_layers:
			conv_layer_output = conv(conv_layer_output, training=training)

		logits = self.final_conv(conv_layer_output, training=training)
		output = self.sigmoid(logits, training=training)
		return logits, output


class SSRN(layers.Layer):
	def __init__(self, hp, **kwargs):
		super(SSRN, self).__init__()

		self.hp = hp

		self.conv1 = Conv1DLayer(filters=self.hp.c, size=1, rate=1, 
			dropout_rate=self.hp.dropout_rate
		)

		self.hc1 = []
		for j in range(2):
			self.hc1.append(
				HC(size=3, rate=3 ** j, dropout_rate=self.hp.dropout_rate)
			)

		self.convT_hc = []
		for _ in range(2):
			self.convT_hc.append(
				Conv1DTransposeLayer(dropout_rate=self.hp.dropout_rate)
			)

			for j in range(2):
				self.convT_hc.append(
					HC(size=3, rate=3 ** j, dropout_rate=self.hp.dropout_rate)
				)

		self.conv2 = Conv1DLayer(filters=2 * self.hp.c, size=1, rate=1,
			dropout_rate=self.hp.dropout_rate
		)

		self.hc2 = []
		for _ in range(2):
			self.hc2.append(
				HC(size=3, rate=1, dropout_rate=self.hp.dropout_rate)
			)

		self.conv3 = Conv1DLayer(filters=1 + self.hp.n_fft // 2, size=1, rate=1,
			dropout_rate=self.hp.dropout_rate
		)

		self.conv_layers = []
		for _ in range(2):
			self.conv_layers.append(
				Conv1DLayer(size=1, rate=1, dropout_rate=self.hp.dropout_rate,
					activation="relu"
				)
			)

		self.conv4 = Conv1DLayer(size=1, rate=1, 
			dropout_rate=self.hp.dropout_rate
		)

		self.sigmoid = layers.Activation("sigmoid")


	def call(self, inputs, training=True):
		conv1_output = self.conv1(inputs, training=training)

		hc1_output = conv1_output
		for hc in self.hc1:
			hc1_output = hc(hc1_output, training=training)

		convT_output = hc1_output
		for convT in self.convT_hc:
			convT_output = convT(convT_output, training=training)

		conv2_output = self.conv2(convT_output, training=training)

		hc2_output = conv2_output
		for hc in self.hc2:
			hc2_output = hc(hc2_output, training=training)

		conv3_output = self.conv3(hc2_output, training=training)

		conv_layer_output = conv3_output
		for conv_layer in self.conv_layers:
			conv_layer_output = conv_layer(conv_layer_output, training=training)

		logits = self.conv4(conv_layer_output, training=training)
		output = self.sigmoid(logits, training=training)
		return logits, output