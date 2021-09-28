from typing import Iterable, Dict

import numpy as np

import logging
import time

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
    TEMPERATURE = "Temperature"
    CONCENTRATION = "Concentration"
    REAGENTRATIO = "ReagentRatio"
    RESIDENCETIME = "ResidenceTime"

    def __init__(self, temperature, concentration, reagentRatio, residenceTime):
        super().__init__({
            XamControlExperimentRequest.TEMPERATURE: temperature,  
            XamControlExperimentRequest.CONCENTRATION: concentration,  
            XamControlExperimentRequest.REAGENTRATIO: reagentRatio,  
            XamControlExperimentRequest.RESIDENCETIME: residenceTime,  
        })

class XamControlExperimentResult(XamControlExperiment):
    CONVERSION = "Conversion"
    STY = "Sty"

    def __init__(self, conversion, Sty, request : XamControlExperimentRequest = None):

        super().__init__({
            XamControlExperimentResult.CONVERSION: conversion,  
            XamControlExperimentResult.STY: Sty
        })

        self.requestExperiment = request

    def getFactorResponseArray(self):
        if self.requestExperiment is None: return self.getValueArray()
        
        values = self.requestExperiment.getValueArray()
        values.extend(self.getValueArray())
        return np.array(values)
            


class XamControl:

    def __init__(self, name="Base class"):
        self.name = name

    def startExperiment(self, experiment : XamControlExperimentRequest) -> XamControlExperimentResult:
        raise NotImplementedError("Use a derived class pls")

    def startExperimentFromvalues(self, valueArray : Iterable) -> XamControlExperimentResult:
        return self.startExperiment(XamControlExperimentRequest(
            temperature=valueArray[0], 
            concentration=valueArray[1],
            reagentRatio=valueArray[2], 
            residenceTime=valueArray[3]
        ))

    def workOffExperiments(self, valueArrays : Iterable) -> Iterable[XamControlExperimentResult]:
        for valueArray in valueArrays: yield self.startExperimentFromvalues(valueArray)

    def _startExperimentRequest(self, experiment : XamControlExperimentRequest):
        logging.debug("Experiment Request -> {}".format(str(experiment)))

    def _receivedExperimentResult(self, result : XamControlExperimentResult):
        logging.debug("Experiment Result -> {}".format(str(result)))
    

class XamControlSimpleMock(XamControl):

    def __init__(self):
        super().__init__("Xam control - Mock")

    def startExperiment(self, experiment : XamControlExperimentRequest, simulateExperimentTime = 0) -> XamControlExperimentResult:

        self._startExperimentRequest(experiment)
        
        if simulateExperimentTime > 0: time.sleep(simulateExperimentTime)

        experimentResult = XamControlExperimentResult(
            self._genericConversionModel(experiment, -0.335595, -0.098665, 0.9325, 0.00379995, 0.000531994, 0.00190386, 0.000248572),
            self._genericStyModel(experiment, 0.200104, -0.0789001, -0.99375, -0.00594246, 0.00201024, 0.023325),
            request=experiment
        )

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _genericConversionModel(self, exp : XamControlExperimentRequest, const, ratio, conc, ResT, Temp, Ra2Temp, ResTemp):

        return const \
            + ratio * exp[XamControlExperimentRequest.REAGENTRATIO] \
            + conc * exp[XamControlExperimentRequest.CONCENTRATION] \
            + ResT * exp[XamControlExperimentRequest.RESIDENCETIME] \
            + Temp * exp[XamControlExperimentRequest.TEMPERATURE] \
            + Ra2Temp * exp[XamControlExperimentRequest.REAGENTRATIO]**2 * exp[XamControlExperimentRequest.TEMPERATURE] \
            + ResTemp * exp[XamControlExperimentRequest.RESIDENCETIME] * exp[XamControlExperimentRequest.TEMPERATURE]  \

    def _genericStyModel(self, exp : XamControlExperimentRequest, const, ratio, conc, Temp, Ra2Temp, ConTemp):

        return const \
            + ratio * exp[XamControlExperimentRequest.REAGENTRATIO] \
            + conc * exp[XamControlExperimentRequest.CONCENTRATION] \
            + Temp * exp[XamControlExperimentRequest.TEMPERATURE] \
            + Ra2Temp * exp[XamControlExperimentRequest.REAGENTRATIO]**2 * exp[XamControlExperimentRequest.TEMPERATURE] \
            + ConTemp * exp[XamControlExperimentRequest.CONCENTRATION] * exp[XamControlExperimentRequest.TEMPERATURE]  \


if __name__ == "__main__":

    print(" Test XamControlMock ".center(80, "-"))

    XamControl = XamControlMock()

    exp = XamControlExperimentRequest(60, .2, .8, 10)

    print(XamControl.startExperiment(exp))