from abc import ABC, abstractmethod

class AbstractUpdate(ABC):

    def update(self):
        pass

class LloydUpdate(AbstractUpdate):

    def update(self):
        pass

class MacQueenUpdate(AbstractUpdate):
    def update(self):
        pass