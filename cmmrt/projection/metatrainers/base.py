from abc import ABC, abstractmethod

from cmmrt.projection.data import ProjectionsTasks


class MetaTrainer(ABC):

    @abstractmethod
    def metatrain(self, tasks: ProjectionsTasks):
        pass
