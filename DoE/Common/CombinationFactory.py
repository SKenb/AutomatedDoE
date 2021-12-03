from typing import Callable, Dict, Iterable

import numpy as np

def reduceAllCombinations(dim, condition : Callable, func : Callable, stringBuilder : Callable) -> dict: 
    char = lambda index: chr(65 + index % 26)
    combinations = {}

    for outerIndex in range(dim):
        for innerIndex in range(dim):

            if condition(outerIndex, innerIndex):
                combinations[stringBuilder(char(outerIndex), char(innerIndex))] = \
                    lambda eV, a=outerIndex, b=innerIndex: func(eV, a, b)

    return combinations

def allLinearCombinations(dim) -> dict: 
    return reduceAllCombinations(
        dim, 
        lambda o, i: i>o,
        lambda e, a, b: e[a] * e[b],
        lambda a, b: "{}*{}".format(a, b)
    )

def allSelfSquaredCombinations(dim) -> dict:
    return reduceAllCombinations(
        dim, 
        lambda o, i: o!=i,
        lambda e, a, b: e[a]**2,
        lambda a, b: "{}^2".format(a)
    )

def combineCombinations(*functions, dim) -> dict:
    combinationsList = {}
    for func in functions: combinationsList.update(func(dim))
    return combinationsList

def removeFactor(combinations:dict, factorIndex, baseCombinationSet:dict):
    char = lambda index: chr(65 + index % 26)
    
    isAtoZ = lambda a: a >= 65 and a <= (65+26)
    aOrd = lambda str_: np.array([ord(c) for c in str_])

    def getLabelsWithOffset(label, offset, bound):
        return "".join([
                    chr(ord_+offset) if isAtoZ(ord_) and ord_-65 > bound else chr(ord_) 
                    for ord_ in aOrd(label)
                ])

    newAllowedLabels = [getLabelsWithOffset(label, -1, factorIndex) for label in combinations]

    return {
        getLabelsWithOffset(label, 1, factorIndex-1) : func 
        for (label, func) in baseCombinationSet.items() 
        if label in newAllowedLabels
    }
    

def allCombinations(dim) -> dict:
    return combineCombinations(allLinearCombinations, allSelfSquaredCombinations, dim=dim)

if __name__ == "__main__":

    allLinearCombinations(3)

    #c = combineCombinations(allLinearCombinations, allSelfSquaredCombinations, 5)
    #print(c.keys())
    #print(c["C^2"]([1, 2, 3, 4, 5, 6]))