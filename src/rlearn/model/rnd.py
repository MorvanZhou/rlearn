# Random Network Distillation
# https://arxiv.org/abs/1810.12894

import typing as tp

import numpy as np
import tensorflow as tf
from tensorflow import keras


class RND:
    def __init__(
            self,
            target: keras.Model,
            predictor: tp.Optional[keras.Model] = None,
            learning_rate: float = 1e-4,
    ):
        self.target = target
        if predictor is None:
            self.predictor = keras.models.clone_model(target)
        else:
            self.predictor = predictor
        if self.target.output_shape != self.predictor.output_shape:
            raise ValueError(f"target output shape {self.target.output_shape}"
                             f" is not equal to the predictor output shape {self.predictor.output_shape}")
        for layer in self.target.layers:
            layer.trainable = False
            ws = []
            for w in layer.get_weights():
                if w.ndim != 1:
                    w = np.random.normal(loc=0., scale=np.sqrt(2), size=w.shape)
                ws.append(w)
            layer.set_weights(ws)

        self.opt = keras.optimizers.RMSprop(learning_rate=learning_rate)

    def intrinsic_reward(self, x):
        with tf.GradientTape() as tape:
            target = self.target(x)
            predict = self.predictor(x)
            r_int = tf.reduce_mean(tf.square(tf.nn.tanh(target - predict)), axis=1)
            loss = tf.reduce_mean(r_int)
        tv = self.predictor.trainable_variables
        grads = tape.gradient(loss, tv)
        self.opt.apply_gradients(zip(grads, tv))
        return r_int.numpy()
