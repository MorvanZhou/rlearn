import os
import shutil
from typing import Optional, Union, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras


class Tensorboard:
    def __init__(
            self,
            logdir: str,
            models: Union[Sequence[keras.Model], keras.Model],
            trace_weights: bool = True
    ):
        self.logdir = logdir
        if isinstance(models, keras.Model):
            models = [models]
        self.models = models
        self.trace_weights = trace_weights
        if os.path.exists(self.logdir):
            shutil.rmtree(self.logdir)
        os.makedirs(self.logdir)
        self.writer = tf.summary.create_file_writer(self.logdir)
        self.set_graph()
        self.step = 0

    def set_graph(self):
        tf.summary.trace_on(graph=True, profiler=False)
        for m in self.models:
            input_shape = [(1, *[s for s in i.shape[1:]]) for i in m.inputs]
            x = [tf.zeros(shape) for shape in input_shape]

            @tf.function
            def trace_graph(inputs):
                m(inputs)

            trace_graph(x)
        with self.writer.as_default():
            tf.summary.trace_export(
                name="model",
                step=0,
                profiler_outdir=self.logdir
            )

    def __trace_img(self, name, data):
        if isinstance(data, np.ndarray):
            data = tf.convert_to_tensor(data, dtype=tf.float32)
        data = tf.squeeze(data)
        if tf.reduce_any(data > 1):
            data /= 255
        ndim = len(data.shape)
        if ndim == 1:
            tf.summary.histogram(name, data, step=self.step)
        elif ndim == 2:
            data = tf.expand_dims(data, 0)
            data = tf.expand_dims(data, -1)
            tf.summary.image(name, data, step=self.step)
        elif ndim == 3:
            data = tf.expand_dims(data, 0)
            tf.summary.image(name, data, step=self.step)
        elif ndim == 4:
            tf.summary.image(name, data, step=self.step)
        else:
            raise ValueError(f"shape of data not right, {data.shape}")

    def trace(self, dict_data: dict, step: Optional[int] = None):
        if step is None:
            self.step += 1
        else:
            self.step = step
        with self.writer.as_default():
            for name, data in dict_data.items():
                if isinstance(data, (int, float)) or data.size == 1:
                    tf.summary.scalar(name, data, step=self.step)
                elif isinstance(data, tf.Variable):
                    tf.summary.histogram(name, data, step=self.step)
                elif 2 <= len(data.shape) <= 4:
                    self.__trace_img(name, data)
                else:
                    raise ValueError(f"data={data} not support")
            if self.trace_weights:
                for m in self.models:
                    for layer in m.layers:
                        if len(layer.weights) == 0:
                            continue
                        tf.summary.histogram(f"model_{layer.name}_weights", layer.weights[0], step=self.step)
                        tf.summary.histogram(f"model_{layer.name}_bias", layer.weights[1], step=self.step)
