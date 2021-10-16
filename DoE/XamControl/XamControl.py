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

        experimentResult = self._wrapXamControlExperimentResult(experiment)

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:
        return XamControlExperimentResult(
            self._genericConversionModel(experiment, 0.0960168, -0.135, 0.63125, -0.0512857, -0.0016125, 0.00213095, 0.000592857),
            self._genericStyModel(experiment, 0.200104, -0.0789001, -0.99375, -0.00594246, 0.00201024, 0.023325),
            request=experiment
        )

    def _genericConversionModel(self, exp : XamControlExperimentRequest, const, ratio, conc, ResT, Temp, RaTemp, ResTemp):

        return const \
            + ratio * exp[XamControlExperimentRequest.REAGENTRATIO] \
            + conc * exp[XamControlExperimentRequest.CONCENTRATION] \
            + ResT * exp[XamControlExperimentRequest.RESIDENCETIME] \
            + Temp * exp[XamControlExperimentRequest.TEMPERATURE] \
            + RaTemp * exp[XamControlExperimentRequest.REAGENTRATIO] * exp[XamControlExperimentRequest.TEMPERATURE] \
            + ResTemp * exp[XamControlExperimentRequest.RESIDENCETIME] * exp[XamControlExperimentRequest.TEMPERATURE]  \

    def _genericStyModel(self, exp : XamControlExperimentRequest, const, ratio, conc, Temp, RaTemp, ConTemp):

        return const \
            + ratio * exp[XamControlExperimentRequest.REAGENTRATIO] \
            + conc * exp[XamControlExperimentRequest.CONCENTRATION] \
            + Temp * exp[XamControlExperimentRequest.TEMPERATURE] \
            + RaTemp * exp[XamControlExperimentRequest.REAGENTRATIO] * exp[XamControlExperimentRequest.TEMPERATURE] \
            + ConTemp * exp[XamControlExperimentRequest.CONCENTRATION] * exp[XamControlExperimentRequest.TEMPERATURE]  \


class XamControlNoMixtureTermsMock(XamControlSimpleMock):

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:
        return XamControlExperimentResult(
            self._genericConversionModel(experiment, -0.974799, 0.129215, 0.96875, 0.0332143, 0.00568852, 0, 0),
            self._genericStyModel(experiment, 0.200104, -0.0789001, -0.99375, -0.00594246, 0, 0),
            request=experiment
        )


class XamControlModdeYMock(XamControlSimpleMock):

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:

        dataSet = np.array([
            [ 60, 0.2, 0.9, 2.5, 0.28, 0.0005],
            [ 60, 0.4, 0.9, 2.5, 0.14, 0.0005],
            [ 60, 0.4, 3, 6, 0.25, 0.1988],
            [ 60, 0.2, 3, 2.5, 0, 0.0005],
            [ 60, 0.2, 3, 6, 0.03, 0.0356],
            [ 60, 0.4, 3, 2.5, 0.13, 0.1394],
            [ 60, 0.4, 0.9, 6, 0.05, 0.0227],
            [ 60, 0.2, 0.9, 6, 0, 0.0002],
            [110, 0.3, 1.95, 4.25, 0.35, np.array([0.2771, 0.2773, 0.2813]).mean()],
            [160, 0.4, 3, 2.5, 0.86, 1.5726],
            [160, 0.4, 0.9, 6, 0.57, 0.4377],
            [160, 0.2, 3, 2.5, 0.63, 0.4241],
            [160, 0.2, 3, 6, 0.84, 0.3616],
            [160, 0.4, 0.9, 2.5, 0.41, 0.4238],
            [160, 0.4, 3, 6, 1, 0.8503],
            [160, 0.2, 0.9, 2.5, 0.26, 0.1189],
            [160, 0.2, 0.9, 6, 0.36, 0.1892],
        ])

        for (index, value) in {
                    0: experiment[XamControlExperimentRequest.TEMPERATURE],
                    1: experiment[XamControlExperimentRequest.CONCENTRATION],
                    2: experiment[XamControlExperimentRequest.REAGENTRATIO], 
                    3: experiment[XamControlExperimentRequest.RESIDENCETIME]
                }.items():
            dataSet = dataSet[dataSet[:, index] == value]
            if dataSet.size == 0: raise Exception("Data not found in dataset :/ - Note: only defined exp. r allowed")

        return XamControlExperimentResult(dataSet[0, 4], dataSet[0, 5], request=experiment)


if __name__ == "__main__":

    print(" Test XamControlMock ".center(80, "-"))

    xamControl = XamControlSimpleMock()

    ## Test from Moode    
    def testExp(number, ratio, conc, resT, temp, expectedConv, expectedSty): 
        result = xamControl.startExperiment(XamControlExperimentRequest(temp, conc, ratio, resT))

        print(">> Test {}:\n\r\tConv-Delta: {}".format(number, round(expectedConv - result[XamControlExperimentResult.CONVERSION], 5)))
        print("\tSty-Delta: {}".format(round(expectedSty - result[XamControlExperimentResult.STY], 5)))

    testExp(2, 0.9, 0.2, 6, 60, 0.00912016, -0.0377515)
    testExp(3, 0.9, 0.4, 2.5, 60, 0.13012, 0.0433987)
    testExp(5, 3, 0.2, 2.5, 60, -0.0236858, 0.0498486)
    testExp(6, 3, 0.2, 6, 60, 0.0418101, 0.0498486)
    testExp(7, 3, 0.4, 2.5, 60, 0.16281, 0.130999)
    testExp(15, 0.9, 0.4, 6, 160, 0.56931, 0.563074)
    testExp(16, 3, 0.2, 2.5, 160, 0.66281, 0.525174)

    xamControl = XamControlModdeYMock()
    print(xamControl.startExperiment(XamControlExperimentRequest(60, .2, .9, 2.5)))
