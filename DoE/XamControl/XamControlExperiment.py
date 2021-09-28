from typing import Dict


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

    def __init__(self, conversion, Sty):

        super().__init__({
            XamControlExperimentResult.CONVERSION: conversion,  
            XamControlExperimentResult.STY: Sty
        })