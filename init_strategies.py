from abc import ABC, abstractmethod

class AbstractInit(ABC):

    @abstractmethod
    def init(self):
        pass

class RandomInit(AbstractInit):

    def init(self):
        pass

class FarthestPointsInit(AbstractInit):
    #See page 17: http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    #https://larssonjohan.com/post/2016-10-30-farthest-points/
    def init(self):
        pass

class PreClusterdSampleInit(AbstractInit):
    #http://infolab.stanford.edu/%7Eullman/mmds/ch7.pdf
    def init(self):
        pass