# synthesize.py
# author: Diego Magdaleno
# Instantiate a Graph object from model.py.
# Python 3.7
# Tensorflow 2.4.0


from hyperparams import Hyperparams as hp
from model import TTSGraph


# Initialize a graph object that contains both the Text2Mel and SSRN
# models for streamlined training and inference.
graph = TTSGraph("original_dc_tts_graph")
graph.load()

# Gather user input for the custom speech they wish to synthesize.
custom_text = [input("input >")]

# Synthesize custom text and output to specified directory.
graph.synthesize(custom_text, "./custom_samples")