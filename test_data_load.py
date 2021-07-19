# test_data_load.py


import numpy as np
import tensorflow as tf
from data_load import *


#data1, data2, data3, = get_batch()
data3 = get_batch()

z = list(data3.as_numpy_iterator())
print(len(z))
print(z[0])