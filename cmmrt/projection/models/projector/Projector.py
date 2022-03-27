from abc import ABC, abstractmethod


class Projector(ABC):
    @abstractmethod
    def train_step(self, x, y, optimizer, **kwargs):
        pass

    @abstractmethod
    def metaparams(self):
        pass

    @abstractmethod
    def update_metaparams(self):
        pass

    @abstractmethod
    def prepare_metatraining(self):
        pass

    @abstractmethod
    def prepare_metatesting(self):
        pass
