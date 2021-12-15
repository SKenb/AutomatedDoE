from typing import Iterable, Dict
from pathlib import Path
from Common import Logger

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
    EQUIVALENTS_NBS = "Equivalents NBS"
    CONCENTRATION = "Concentration"
    RESIDENCE_TIME = "ResidenceTime"
    TEMPERATURE = "Temperature"
    LIGHT_INTENSITY= "Light intensity"
    QUANTITY_ACOH = "Quantity AcOH"

    def __init__(self, equivalentsNBS, concentration, residenceTime, temperature, lightIntensity, quantityACOH):
        super().__init__({
            XamControlExperimentRequest.EQUIVALENTS_NBS: equivalentsNBS, 
            XamControlExperimentRequest.CONCENTRATION: concentration,  
            XamControlExperimentRequest.RESIDENCE_TIME: residenceTime,   
            XamControlExperimentRequest.TEMPERATURE: temperature,  
            XamControlExperimentRequest.LIGHT_INTENSITY: lightIntensity,
            XamControlExperimentRequest.QUANTITY_ACOH: quantityACOH
        })

class XamControlExperimentResult(XamControlExperiment):
    STY = "Space-time yield"
    CONVERSION = "Conversion"
    SELECTIVITY = "Selectivity"

    def __init__(self, sty, conversion, selectivity, request : XamControlExperimentRequest = None):

        super().__init__({
            XamControlExperimentResult.STY: sty,
            XamControlExperimentResult.CONVERSION: conversion,  
            XamControlExperimentResult.SELECTIVITY: selectivity
        })

        self.requestExperiment = request

    def getFactorResponseArray(self):
        if self.requestExperiment is None: return self.getValueArray()
        
        values = self.requestExperiment.getValueArray()
        values.extend(self.getValueArray())
        return np.array(values)
            


class XamControlBase:

    def __init__(self, name="Base class"):
        self.name = name

    def startExperiment(self, experiment : XamControlExperimentRequest) -> XamControlExperimentResult:
        raise NotImplementedError("Use a derived class pls")

    def startExperimentFromvalues(self, valueArray : Iterable) -> XamControlExperimentResult:
        return self.startExperiment(XamControlExperimentRequest(
            equivalentsNBS=valueArray[0], 
            concentration=valueArray[1],
            residenceTime=valueArray[2], 
            temperature=valueArray[3],
            lightIntensity=valueArray[4], 
            quantityACOH=valueArray[5]
        ))

    def workOffExperiments(self, valueArrays : Iterable) -> Iterable[XamControlExperimentResult]:
        for valueArray in valueArrays: yield self.startExperimentFromvalues(valueArray)

    def _startExperimentRequest(self, experiment : XamControlExperimentRequest):
        Logger.logXamControl("Request -> {}".format(str(experiment)))
        pass

    def _receivedExperimentResult(self, result : XamControlExperimentResult):
        Logger.logXamControl("Result -> {}".format(str(result)))
        pass
    

class XamControlFactorsOnlyMock(XamControlBase):

    def __init__(self):
        super().__init__("Xam control - Factors Only Mock")

    def startExperiment(self, experiment : XamControlExperimentRequest, simulateExperimentTime = 0) -> XamControlExperimentResult:

        self._startExperimentRequest(experiment)
        
        if simulateExperimentTime > 0: time.sleep(simulateExperimentTime)

        experimentResult = self._wrapXamControlExperimentResult(experiment)

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:
        return XamControlExperimentResult(
            self._genericStyModel(experiment),
            self._genericConversionModel(experiment),
            self._genericSelectivityModel(experiment),
            request=experiment
        )

    def _genericStyModel(self, exp : XamControlExperimentRequest):
        return 10 + exp[exp.TEMPERATURE] + 100*exp[exp.CONCENTRATION]

    def _genericConversionModel(self, exp : XamControlExperimentRequest):
        return 0

    def _genericSelectivityModel(self, exp : XamControlExperimentRequest):
        return 1

class XamControl(XamControlBase):

    def __init__(self):
        super().__init__("Xam control - CSV Implementation")

        self.path = Path("./Tmp")
        self.xFileName = Path("xnewtrue.csv")
        self.yFileName = Path("ynewtrue.csv")

        self.oldYValues = None
        self.yValuesEpsilon = 1e-3

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
            fileWriter.writerow(valuestoWrite)

    def waitForNewResponseValues(self):
        # TODO: Threading
        while not self.yPathExists() or not self.newYValuesAvailable(): 
            time.sleep(1)

    def readNewResponseValues(self) -> XamControlExperimentResult:
       
        firstRow = self.readFirstValueRow(self.yPath())
        return XamControlExperimentResult(firstRow[0], firstRow[1])
    
    def startExperiment(self, experiment : XamControlExperimentRequest) -> XamControlExperimentResult:
        self._startExperimentRequest(experiment)

        self.loadOldYValues()
        self.writeNewExperimentValuesInFile(experiment)
        self.waitForNewResponseValues()
        experimentResult = self.readNewResponseValues()

        self._receivedExperimentResult(experimentResult)
        return experimentResult


if __name__ == "__main__":

    print(" Test XamControlMock ".center(80, "-"))
