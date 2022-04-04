
import tensorflow as tf 
import numpy as np
def sgd(cost,params,lr=np.float32(0.01)):
    g_params = tf.gradients(cost, params)
    updates = []
    for param, g_param in zip(params, g_params):
        updates.append(param.assign(param-lr*g_param))
        return updates
        #creating an optimizer for tf: stochastic optimizers.