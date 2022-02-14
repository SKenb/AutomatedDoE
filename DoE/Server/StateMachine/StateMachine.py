from typing import Callable

import time
import Common.Logger as Logger

class State():
    def __init__(self, name : str):
        self.name = name

    def __str__(self):
        return "STATE: {}".format(self.name)

    def __call__(self):
        try:
            Logger.logState(str(self))
            return self.onCall()
        except Exception as e:
            return self.onException(e)

    def onCall(self):
        return None

    def onException(self, excpetion):
        Logger.logException(excpetion)
        return None

    def result(self):
        return None


class StateMachine():
    def __init__(self, currentState : State):
        self.currentState = currentState

    def next(self) -> State:
        if self.currentState is None: raise Exception("No init state :0")
        self.currentState = self.currentState()
        return self.currentState

    def __iter__(self):
        return self
 
    def __next__(self) -> State:
        if self.currentState is None: raise StopIteration
        calledState = self.currentState
        self.next()
        return calledState
        
var = 0
class TmpStateInit(State):
    def __init__(self):
        super().__init__("Init")

    def onCall(self):
        print("Init everything")
        global var
        var = 5
        return TmpStateA()

class TmpStateA(State):
    def __init__(self):
        super().__init__("A")

    def onCall(self):
        global var
        print("Do sth in state A (var={})".format(var))
        time.sleep(1)
        var = 1
        return TmpStateB()

class TmpStateB(State):
    def __init__(self):
        super().__init__("B")

    def onCall(self):
        print("Do sth in state B (var={})".format(var))
        raise Exception("Test")
        return None


if __name__ == "__main__":
    sm = StateMachine(TmpStateInit())
    for state in sm: print(state)