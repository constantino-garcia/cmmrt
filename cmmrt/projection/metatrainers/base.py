from abc import ABC, abstractmethod

from cmmrt.projection.projection_tasks import ProjectionsTasks


class MetaTrainer(ABC):

    @abstractmethod
    def metatrain(self, tasks: ProjectionsTasks):
        pass
