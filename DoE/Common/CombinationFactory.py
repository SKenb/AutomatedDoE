from typing import Callable, Dict, Iterable

def reduceAllCombinations(dim, predicate : Callable, stringBuilder : Callable) -> Dict: 
    letter = lambda index: chr(65 + index % 26)
    combinations = {}

    for outerIndex in range(dim):
        for innerIndex in range(dim):

            func = predicate(outerIndex, innerIndex)
            if func is not None:
                combinations[stringBuilder(letter(outerIndex), letter(innerIndex))] = \
                    lambda eV, a=outerIndex, b=innerIndex: func(eV, a, b)

    return combinations

def allLinearCombinations(dim=4): 
    return reduceAllCombinations(
        dim, 
        lambda i, o: None if i>o else lambda e, a, b: e[a] * e[b],
        lambda a, b: "{}*{}".format(a, b)
    )

def allSelfSquaredCombinations(dim=4):
    return reduceAllCombinations(
        dim, 
        lambda i, o: None if i!=o else lambda e, a, b: e[a]**2,
        lambda a, b: "{}^2".format(a)
    )

def combineCombinations(*functions, dim=4):
    combinationsList = {}
    for func in functions: combinationsList.update(func(dim))
    return combinationsList

def allCombinations(dim=4):
    return combineCombinations(allLinearCombinations, allSelfSquaredCombinations, dim=dim)

if __name__ == "__main__":

    c = combineCombinations(allLinearCombinations, allSelfSquaredCombinations)
    print(c.keys())
    print(c["C^2"]([1, 2, 3, 4, 5, 6]))