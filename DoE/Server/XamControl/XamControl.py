from tkinter.messagebox import NO
from typing import Iterable, Dict
from pathlib import Path
from Common import Logger
from Common import Factor

import numpy as np

import time
import csv

class XamControlExperiment:

    def __init__(self, propertyDict : Dict):

        self.propertyDict = propertyDict

    def __str__(self):
        return str(self.propertyDict)

    def getProperty(self, propertyName):
        return None if propertyName not in self.propertyDict else self.propertyDict[propertyName]

    def setProperty(self, propertyName, newValue):
        if propertyName not in self.propertyDict: return False

        self.propertyDict[propertyName] = newValue
        return True

    def __getitem__(self, key):
        return self.getProperty(key)

    def __setitem__(self, key, value):
        self.setProperty(key, value)
    
    def getValueArray(self):
        return list(self.propertyDict.values())
    

class XamControlExperimentRequest(XamControlExperiment):
    def __init__(self, factorSet:Factor.FactorSet, factorValues:Iterable):
        assert len(factorSet.factors) == len(factorValues), "No - we don't do this"
        super().__init__({factor.name: factorValues[index] for index, factor in enumerate(factorSet.factors) })

class XamControlExperimentResult(XamControlExperiment):
    def __init__(self, reponseValues:Iterable, request : XamControlExperimentRequest = True):
        if len(reponseValues)<=0: return

        responseDict = {}

        responseDict["Response"] = reponseValues[0]
        for index, value in enumerate(reponseValues[1:]):
            responseDict["Additional {}".format(index)] = value

        super().__init__(responseDict)
        self.requestExperiment = request

    def getFactorResponseArray(self):
        if self.requestExperiment is None: return self.getValueArray()
        
        values = self.requestExperiment.getValueArray()
        values.extend(self.getValueArray())
        return np.array(values)
            


class XamControlBase:

    def __init__(self, name="Base class", importedExperiments:Iterable=None):
        self.name = name
        self.importedExperiments = importedExperiments

    def importExperiments(self, importExperiments:Iterable):
        self.importedExperiments = importExperiments

    def resetImport(self):
        self.importedExperiments = None

    def startExperiment(self, experiment : XamControlExperimentRequest) -> XamControlExperimentResult:
        if self.importedExperiments is None: return None

        dataSet = np.array(self.importedExperiments)
        for index, value in enumerate(experiment.propertyDict.values()):
            dataSet = dataSet[dataSet[:, index] == value]

        if dataSet.size <= 0: return None 
        if dataSet.size <= len(experiment.propertyDict): return None

        
        Logger.logXamControl("Result IMPORTED")
        return XamControlExperimentResult(dataSet[0,len(experiment.propertyDict):])

    def startExperimentFromvalues(self, factorSet : Factor.FactorSet, valueArray : Iterable) -> XamControlExperimentResult:
        return self.startExperiment(XamControlExperimentRequest(factorSet, valueArray))

    def workOffExperiments(self, valueArrays : Iterable) -> Iterable[XamControlExperimentResult]:
        for valueArray in valueArrays: yield self.startExperimentFromvalues(valueArray)

    def _startExperimentRequest(self, experiment : XamControlExperimentRequest):
        Logger.logXamControl("Request -> {}".format(str(experiment)))
        pass

    def _receivedExperimentResult(self, result : XamControlExperimentResult):
        Logger.logXamControl("Result -> {}".format(str(result)))
        pass

class XamControl(XamControlBase):

    def __init__(self):
        super().__init__("Xam control - CSV Implementation")

        self.path = Path("\\\\RCPEPC01915\\UHPLC-Data\\") #Path("./Tmp")
        self.xFileName = Path("xnewtrue.csv")
        self.yFileName = Path("ynewtrue.csv")

        self.numberOfExpectedValues = 3

        self.oldYValues = None
        self.yValuesEpsilon = 1e-5

    def xPath(self): return self.path / self.xFileName

    def yPath(self): return self.path / self.yFileName

    def yPathExists(self): return self.yPath().is_file()

    def xPathExists(self): return self.xPath().is_file()


    def readFirstValueRow(self, path):
        with open(path, newline='') as csvfile:

            fileReader = csv.reader(csvfile, delimiter=';', quotechar='|')
            return [float(v_) for v_ in fileReader.__next__()]         

    def loadOldYValues(self):
        self.oldYValues = None

        if not self.yPathExists(): return False
        self.oldYValues = np.array(self.readFirstValueRow(self.yPath()))

    def newYValuesAvailable(self):

        if self.oldYValues is None and self.yPathExists(): return True
        if not self.yPathExists(): return False
        
        newVaules = np.array(self.readFirstValueRow(self.yPath()))

        if len(newVaules) != self.numberOfExpectedValues:
            Logger.logWarn("XAMControl - ynewdata contains wrong number of values") 
            return False
            
        if newVaules.shape != self.oldYValues.shape: return True

        return not (np.abs(self.oldYValues - newVaules) <= self.yValuesEpsilon).all()

    def writeNewExperimentValuesInFile(self, experiment : XamControlExperimentRequest):
        valuestoWrite = experiment.getValueArray()

        if self.xPathExists():
            currentXValues = np.array(self.readFirstValueRow(self.xPath()))
            if len(valuestoWrite) == len(currentXValues) and np.linalg.norm(currentXValues - valuestoWrite) <= 1e-4:
                #valuestoWrite += (.5 - np.random.rand(len(valuestoWrite))) / 1000
                valuestoWrite += np.sign((.5 - np.random.rand(len(valuestoWrite)))) * valuestoWrite / 1000

        with open(self.xPath(), 'w', newline='') as csvfile:

            fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            #fileWriter.writerow(valuestoWrite)
            fileWriter.writerow(np.array(experiment.getValueArray())[[2, 1, 3, 0]])

    def waitForNewResponseValues(self):
        # TODO: Threading
        while not self.yPathExists() or not self.newYValuesAvailable(): 
            time.sleep(1)

    def readNewResponseValues(self) -> XamControlExperimentResult:
       
        firstRow = self.readFirstValueRow(self.yPath())
        return XamControlExperimentResult(firstRow)
    
    def startExperiment(self, experiment : XamControlExperimentRequest) -> XamControlExperimentResult:
        self._startExperimentRequest(experiment)

        superResult = super().startExperiment(experiment)
        if superResult is not None:
            experimentResult = superResult

        else:
            self.loadOldYValues()
            self.writeNewExperimentValuesInFile(experiment)
            self.waitForNewResponseValues()
            experimentResult = self.readNewResponseValues()

        self._receivedExperimentResult(experimentResult)
        return experimentResult


if __name__ == "__main__":

    print(" Test XamControlMock ".center(80, "-"))

    xc = XamControl()

    print(xc.startExperiment(XamControlExperimentRequest(0.8, .25, .33, 10, 100, 0)))
    print(xc.startExperiment(XamControlExperimentRequest(0.8, .25, .33, 10, 100, 0)))
    print(xc.startExperiment(XamControlExperimentRequest(0.8, .25, .33, 10, 100, 0)))