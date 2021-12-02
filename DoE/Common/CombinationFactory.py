from typing import Callable, Dict, Iterable

def reduceAllCombinations(dim, condition : Callable, func : Callable, stringBuilder : Callable) -> Dict: 
    char = lambda index: chr(65 + index % 26)
    combinations = {}

    for outerIndex in range(dim):
        for innerIndex in range(dim):

            if condition(outerIndex, innerIndex):
                combinations[stringBuilder(char(outerIndex), char(innerIndex))] = \
                    lambda eV, a=outerIndex, b=innerIndex: func(eV, a, b)

    return combinations

def allLinearCombinations(dim): 
    return reduceAllCombinations(
        dim, 
        lambda o, i: i>o,
        lambda e, a, b: e[a] * e[b],
        lambda a, b: "{}*{}".format(a, b)
    )

def allSelfSquaredCombinations(dim):
    return reduceAllCombinations(
        dim, 
        lambda o, i: o!=i,
        lambda e, a, b: e[a]**2,
        lambda a, b: "{}^2".format(a)
    )

def combineCombinations(*functions, dim):
    combinationsList = {}
    for func in functions: combinationsList.update(func(dim))
    return combinationsList

def allCombinations(dim):
    return combineCombinations(allLinearCombinations, allSelfSquaredCombinations, dim=dim)

if __name__ == "__main__":

    allLinearCombinations(3)

    #c = combineCombinations(allLinearCombinations, allSelfSquaredCombinations, 5)
    #print(c.keys())
    #print(c["C^2"]([1, 2, 3, 4, 5, 6]))