from typing import Dict


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
    def __init__(self, index, combinations, scaledModel, context, r2Score, q2Score, excludedFactors, scoreCombis : Dict = {}) -> None:
        super().__init__()

        self.index = index
        self.combinations = combinations
        self.scaledModel = scaledModel
        self.r2 = r2Score
        self.q2 = q2Score
        self.scoreCombis = scoreCombis
        self.excludedFactors = excludedFactors.copy()
        self.context = context

class DoEHistoryItem(HistoryItem):
    def __init__(self, index, combiScoreHistory, bestCombiScoreItem) -> None:
        super().__init__()

        self.index = index
        self.combiScoreHistory = combiScoreHistory
        self.bestCombiScoreItem = bestCombiScoreItem