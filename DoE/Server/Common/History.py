from typing import Dict
import copy

class HistoryItem():
    def __init__(self) -> None:
        pass

class History():
    def __init__(self) -> None:
        self.container = []

    def add(self, item : HistoryItem):
        self.container.append(item)

    def filter(self, predicate) -> HistoryItem:
        return [item for item in self.container if predicate(item)]

    def choose(self, predicate):
        return [predicate(item) for item in self.container]

    def items(self): 
        return self.container

    def __len__(self): 
        return len(self.container)

    def __getitem__(self, key):
        return self.container[key]

class CombiScoreHistoryItem(HistoryItem):
    def __init__(self, index, combinations, model, scaledModel, context, r2Score, q2Score, excludedFactors, scoreCombis : Dict = {}) -> None:
        super().__init__()

        self.index = index
        self.combinations = combinations
        self.model = model
        self.scaledModel = scaledModel
        self.r2 = r2Score
        self.q2 = q2Score
        self.scoreCombis = scoreCombis
        self.excludedFactors = excludedFactors.copy()
        self.context = copy.deepcopy(context)

class DoEHistoryItem(HistoryItem):
    def __init__(self, index, combiScoreHistory, bestCombiScoreItem, numberOfExperiments, ) -> None:
        super().__init__()

        self.index = index
        self.numberOfExperiments = numberOfExperiments 
        self.combiScoreHistory = combiScoreHistory
        self.bestCombiScoreItem = bestCombiScoreItem