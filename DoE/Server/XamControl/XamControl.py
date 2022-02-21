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
    STY = "Space-time yield"
    CONVERSION = "Conversion"
    EFACTOR = "E-Factor"

    def __init__(self, sty, conversion, efactor, request : XamControlExperimentRequest = True):

        super().__init__({
            XamControlExperimentResult.STY: sty,
            XamControlExperimentResult.CONVERSION: conversion,
            XamControlExperimentResult.EFACTOR: efactor
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
            temperature=valueArray[0], 
            concentration=valueArray[1],
            reagentRatio=valueArray[2], 
            residenceTime=valueArray[3]
        ))

    def workOffExperiments(self, valueArrays : Iterable) -> Iterable[XamControlExperimentResult]:
        for valueArray in valueArrays: yield self.startExperimentFromvalues(valueArray)

    def _startExperimentRequest(self, experiment : XamControlExperimentRequest):
        Logger.logXamControl("Request -> {}".format(str(experiment)))
        pass

    def _receivedExperimentResult(self, result : XamControlExperimentResult):
        Logger.logXamControl("Result -> {}".format(str(result)))
        pass

 
class XamControlTestRun1Mock(XamControlBase):

    def __init__(self):
        super().__init__("Xam control - TestRun 2 Mock")

    def startExperiment(self, experiment : XamControlExperimentRequest, simulateExperimentTime = 5) -> XamControlExperimentResult:

        self._startExperimentRequest(experiment)
        
        if simulateExperimentTime > 0: time.sleep(simulateExperimentTime)

        experimentResult = self._wrapXamControlExperimentResult(experiment)

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:
        # TestRun1
        # Data from test run 1
        # Conv. was not rec. unf. but Sty
        # change factor: temp [min 100 max 160]
        dataSet = np.array([
            [100, 0.2, 0.9, 2.5, 0, 0.000539177],
            [100, 0.4, 0.9, 2.5, 0, 0.066285834],
            [100, 0.4, 3, 6, 0, 0.523382715],
            [100, 0.2, 3, 6, 0, 0.124081704],
            [130, 0.3, 1.95, 4.25, 0, 0.340612845],
            [130, 0.3, 1.95, 4.25, 0, 0.342071572],
            [160, 0.2, 0.9, 2.5, 0, 0.066361663],
            [160, 0.4, 0.9, 2.5, 0, 0.414149606],
            [160, 0.4, 3, 6, 0, 0.801074925],
            [160, 0.2, 3, 6, 0, 0.317247699],
            [160, 0.2, 0.9, 6, 0, 0.137450315],
            [160, 0.4, 0.9, 6, 0, 0.39538596],
            [160, 0.2, 3, 2.5, 0, 0.324086264],
            [130, 0.3, 1.95, 4.25, 0, 0.360507817],
            [100, 0.2, 0.9, 6, 0, 0.018517327],
            [100, 0.4, 0.9, 6, 0, 0.119053332],
            [100, 0.4, 3, 2.5, 0, 0.569830021],
            [100, 0.2, 3, 2.5, 0, 0.10577096],
            [160, 0.3, 1.95, 4.25, 0, 0.508014395],
            [100, 0.3, 1.95, 4.25, 0, 0.174797307],
            [130, 0.4, 1.95, 4.25, 0, 0.618870595],
            [130, 0.2, 1.95, 4.25, 0, 0.15085711],
            [130, 0.3, 0.9, 4.25, 0, 0.144623276],
            [130, 0.3, 3, 4.25, 0, 0.505340362],
            [130, 0.3, 1.95, 6, 0, 0.335538103],
            [130, 0.3, 1.95, 2.5, 0, 0.341930795],
            [160.0, 0.4, 3.0, 2.5, 0, 1.34986653333749], #0.20732678630067]
        ])

        for (index, value) in {
                    0: experiment[XamControlExperimentRequest.TEMPERATURE],
                    1: experiment[XamControlExperimentRequest.CONCENTRATION],
                    2: experiment[XamControlExperimentRequest.REAGENTRATIO], 
                    3: experiment[XamControlExperimentRequest.RESIDENCETIME]
                }.items():
            dataSet = dataSet[dataSet[:, index] == value]

            if dataSet.size == 0: 
                raise Exception("Data not found in dataset :/ - Note: only defined exp. r allowed (Idx:" + str(index) + ")")

        # Conversion not in test data set and Sty and Conv switched
        return XamControlExperimentResult(np.mean(dataSet[0, 5]), np.mean(dataSet[:, 4]), 0, request=experiment)


class XamControlTestRun1RobustnessMock(XamControlBase):

    def __init__(self):
        super().__init__("Xam control - TestRun 1 Robustness Mock")

    def startExperiment(self, experiment : XamControlExperimentRequest, simulateExperimentTime = 5) -> XamControlExperimentResult:

        self._startExperimentRequest(experiment)
        
        if simulateExperimentTime > 0: time.sleep(simulateExperimentTime)

        experimentResult = self._wrapXamControlExperimentResult(experiment)

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:
        dataSet = np.array([
            [157.0, 0.39, 2.895, 2.325, 1.50686373957754, 0.228764999825665, 12.8053439686205],
            [157.0, 0.41, 2.895, 2.325, 1.61231090628216, 0.232833324619125, 11.9224581910588],
            [157.0, 0.41, 3.105, 2.675, 1.5929374387514, 0.24676460146991, 10.3794055097971],
            [157.0, 0.39, 3.105, 2.675, 1.49910704619314, 0.244138367148585, 11.0723781304237],
            [160.0, 0.4, 3.0, 2.5, 1.59097225499565, 0.244358350012602, 11.1754681385396],
            [160.0, 0.4, 3.0, 2.5, 1.59090852804596, 0.244348562153494, 11.1759558507065],
            [163.0, 0.39, 2.895, 2.325, 1.58944846233698, 0.241302625883987, 12.0880445208794],
            [163.0, 0.41, 2.895, 2.325, 1.90561645562109, 0.275189489249491, 9.9334804577077],
            [163.0, 0.41, 3.105, 2.675, 1.65859488710968, 0.256935706551326, 9.92893822847714],
            [163.0, 0.39, 3.105, 2.675, 1.54216579081013, 0.251150736031084, 10.7353057806574],
            [163.0, 0.39, 2.895, 2.675, 1.49992694342721, 0.261991096834528, 11.0545299009508],
            [163.0, 0.41, 2.895, 2.675, 1.59176351661597, 0.264469578634774, 10.3766540499954],
            [163.0, 0.41, 3.105, 2.325, 1.77964009785386, 0.239615823288188, 10.7189024792087],
            [163.0, 0.39, 3.105, 2.325, 1.70179534213833, 0.240885074591057, 11.2354225946357],
            [160.0, 0.4, 3.0, 2.5, 1.58001532161022, 0.242675468268527, 11.2599013661835],
            [157.0, 0.39, 2.895, 2.675, 1.40040025992631, 0.244606846828909, 11.9112473813283],
            [157.0, 0.41, 2.895, 2.675, 1.52605665254613, 0.253552462824189, 10.8664944893954],
            [157.0, 0.41, 3.105, 2.325, 1.69417076417585, 0.228107988204091, 11.3101101706144],
            [157.0, 0.39, 3.105, 2.325, 1.63454555905879, 0.231366028080451, 11.7388221547248],
            [157.0, 0.4, 3.0, 2.5, 1.54508935415494, 0.237311168700658, 11.5370302681261],
            [163.0, 0.4, 3.0, 2.5, 1.6620148243936, 0.255269819388783, 10.655029615676],
            [160.0, 0.39, 3.0, 2.5, 1.51512534917782, 0.23867587674194, 11.7749200490265],
            [160.0, 0.41, 3.0, 2.5, 1.64937568518964, 0.247149821150085, 10.7535737758684],
            [160.0, 0.4, 3.105, 2.5, 1.61986472238924, 0.240382570832716, 10.9640178665136],
            [160.0, 0.4, 2.895, 2.5, 1.54782986099746, 0.246354491952009, 11.5088512683965],
            [160.0, 0.4, 3.0, 2.325, 1.67169220918139, 0.23878324230614, 11.4597407934938],
            [160.0, 0.4, 3.0, 2.675, 1.51218296148285, 0.248515050768851, 10.9718193958877]
        ])

        for (index, value) in {
                    0: experiment[XamControlExperimentRequest.TEMPERATURE],
                    1: experiment[XamControlExperimentRequest.CONCENTRATION],
                    2: experiment[XamControlExperimentRequest.REAGENTRATIO], 
                    3: experiment[XamControlExperimentRequest.RESIDENCETIME]
                }.items():
            dataSet = dataSet[dataSet[:, index] == value]

            if dataSet.size == 0: 
                raise Exception("Data not found in dataset :/ - Note: only defined exp. r allowed (Idx:" + str(index) + ")")

        return XamControlExperimentResult(np.mean(dataSet[:, 4]), np.mean(dataSet[:, 5]), np.mean(dataSet[:, 6]), request=experiment)



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
        return 10 + exp[exp.TEMPERATURE] + 100*exp[exp.CONC_SM] -  200*exp[exp.TEMPERATURE]*exp[exp.CONC_SM]

    def _genericConversionModel(self, exp : XamControlExperimentRequest):
        return 0

    def _genericSelectivityModel(self, exp : XamControlExperimentRequest):
        return 1

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
        return XamControlExperimentResult(firstRow[0], firstRow[1], firstRow[2])
    
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

    xc = XamControl()

    print(xc.startExperiment(XamControlExperimentRequest(0.8, .25, .33, 10, 100, 0)))
    print(xc.startExperiment(XamControlExperimentRequest(0.8, .25, .33, 10, 100, 0)))
    print(xc.startExperiment(XamControlExperimentRequest(0.8, .25, .33, 10, 100, 0)))