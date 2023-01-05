from abc import ABC, abstractmethod

from tensorflow import keras


class BaseRLModel(ABC):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        self.training = training

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def add_encoder(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def predict(self, x):
        pass

    @abstractmethod
    def save(self, path: str):
        """Save to zip file"""
        pass

    @abstractmethod
    def load_weights(self, path: str):
        """Load from zip file"""
        pass

    @staticmethod
    def clone_model(model):
        try:
            new_model = keras.models.clone_model(model)
        except ValueError:
            new_model = type(model)()
            new_model.set_weights(model.get_weights())
        return new_model
