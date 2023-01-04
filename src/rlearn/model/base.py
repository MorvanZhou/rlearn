from abc import ABC, abstractmethod


class BaseRLNet(ABC):
    name = __qualname__

    def __init__(
            self,
            training: bool = True,
    ):
        self.training = training

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    @abstractmethod
    def build(self, *args, **kwargs):
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
