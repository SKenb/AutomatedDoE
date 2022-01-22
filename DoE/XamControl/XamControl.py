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
    RATIO_R1_SM1 = "Ratio R_1/SM1"
    CONC_SM = "Conc_SM"
    RESIDENCE_TIME = "Residence Time"
    TEMPERATURE = "Temperature"
    INTENSITY= "Intensity"
    RATIO_R2_SM1 = "Ratio R_2/SM1"

    def __init__(self, ratioR1SM1, concSM, residenceTime, temperature, intensity, ratioR2SM1):
        super().__init__({
            XamControlExperimentRequest.RATIO_R1_SM1: ratioR1SM1, 
            XamControlExperimentRequest.CONC_SM: concSM,  
            XamControlExperimentRequest.RESIDENCE_TIME: residenceTime,   
            XamControlExperimentRequest.TEMPERATURE: temperature,  
            XamControlExperimentRequest.INTENSITY: intensity,
            XamControlExperimentRequest.RATIO_R2_SM1: ratioR2SM1
        })

class XamControlExperimentResult(XamControlExperiment):
    STY = "Space-time yield"
    CONVERSION = "Conversion"
    SELECTIVITY = "Selectivity"

    def __init__(self, sty, conversion, selectivity, request : XamControlExperimentRequest = True):

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
            ratioR1SM1=valueArray[0], 
            concSM=valueArray[1],
            residenceTime=valueArray[2], 
            temperature=valueArray[3],
            intensity=valueArray[4], 
            ratioR2SM1=valueArray[5]
        ))

    def workOffExperiments(self, valueArrays : Iterable) -> Iterable[XamControlExperimentResult]:
        for valueArray in valueArrays: yield self.startExperimentFromvalues(valueArray)

    def _startExperimentRequest(self, experiment : XamControlExperimentRequest):
        Logger.logXamControl("Request -> {}".format(str(experiment)))
        pass

    def _receivedExperimentResult(self, result : XamControlExperimentResult):
        Logger.logXamControl("Result -> {}".format(str(result)))
        pass

 
class XamControlTestRun2Mock(XamControlBase):

    def __init__(self):
        super().__init__("Xam control - TestRun 2 Mock")

    def startExperiment(self, experiment : XamControlExperimentRequest, simulateExperimentTime = 0) -> XamControlExperimentResult:

        self._startExperimentRequest(experiment)
        
        if simulateExperimentTime > 0: time.sleep(simulateExperimentTime)

        experimentResult = self._wrapXamControlExperimentResult(experiment)

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _wrapXamControlExperimentResult(self, experiment) -> XamControlExperimentResult:
        # TestRun2
        # Data from test run 2
        dataSet = np.array([
            [0.9, 0.4, 0.8, 40.0, 750.0, 0.01, 0.0443911883676991, 0.0761227156755557, 0.10283252717708], 
            [0.9, 0.45, 0.8, 40.0, 750.0, 0.01, 0.0481164093260401, 0.0527946857918671, 0.142856001360846], 
            [1.1, 0.45, 1.2, 40.0, 950.0, 0.05, 3.93084795647584, 0.993639573393294, 0.930130988470871], 
            [1.1, 0.4, 1.2, 40.0, 950.0, 0.05, 3.21918666749447, 0.995915084735362, 0.854994104016711], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.07688711582829, 0.982658253301662, 0.860706862452885], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.0625616334369, 0.986335389485986, 0.852777715396857], 
            [0.9, 0.4, 0.8, 50.0, 750.0, 0.01, 0.0720124234495608, 0.081114292473934, 0.156551795581054], 
            [0.9, 0.45, 0.8, 50.0, 750.0, 0.01, 0.0874840959507545, 0.0651677799955119, 0.210422254869087], 
            [1.1, 0.45, 1.2, 50.0, 950.0, 0.05, 3.78142363639141, 0.995270794851949, 0.893307173902948], 
            [1.1, 0.4, 1.2, 50.0, 950.0, 0.05, 3.08269087162893, 0.989814880599755, 0.823787632881352], 
            [1.1, 0.4, 0.8, 50.0, 950.0, 0.05, 4.08007660424932, 0.86231801199788, 0.834351152906289], 
            [1.1, 0.45, 0.8, 50.0, 950.0, 0.05, 4.83031096779965, 0.844457703089836, 0.896587495746127], 
            [0.9, 0.45, 1.2, 50.0, 750.0, 0.01, 3.54379496664439, 0.933865053118857, 0.8922184730007], 
            [0.9, 0.4, 1.2, 50.0, 750.0, 0.01, 3.28499270406652, 0.895124311608523, 0.970711839428525], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.07483821901457, 0.993407076992749, 0.850965996401698], 
            [1.1, 0.4, 0.8, 40.0, 950.0, 0.05, 1.78705379150516, 0.439825048406312, 0.71648264808816], 
            [1.1, 0.45, 0.8, 40.0, 950.0, 0.05, 2.43514526600041, 0.467036858877512, 0.817276865218177], 
            [0.9, 0.45, 1.2, 40.0, 750.0, 0.01, 3.62553242186439, 0.897460851209615, 0.949823756626949], 
            [0.9, 0.4, 1.2, 40.0, 750.0, 0.01, 3.05856770896612, 0.89864513968013, 0.900262456797316], 
            [1.1, 0.4, 0.8, 40.0, 750.0, 0.01, 0.0699681385336413, 0.0643007066564503, 0.191881270345226], 
            [1.1, 0.45, 0.8, 40.0, 750.0, 0.01, 0.0581498388064096, 0.0330069972490038, 0.276145510325806], 
            [0.9, 0.45, 1.2, 40.0, 950.0, 0.05, 3.50991228689682, 0.924227580160809, 0.892902598412257], 
            [0.9, 0.4, 1.2, 40.0, 950.0, 0.05, 3.04121057110495, 0.940502434901791, 0.855314505012933], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.02055737589892, 0.976011637251986, 0.854595011658839], 
            [1.1, 0.4, 0.8, 50.0, 750.0, 0.01, 0.120570463175426, 0.0908624588308394, 0.233993856474851], 
            [1.1, 0.45, 0.8, 50.0, 750.0, 0.01, 0.21458947915694, 0.090093720594235, 0.373344185389757], 
            [0.9, 0.45, 1.2, 50.0, 950.0, 0.05, 3.53461409761507, 0.913728452044059, 0.909518640840108], 
            [0.9, 0.4, 1.2, 50.0, 950.0, 0.05, 3.049031564967, 0.913754053927024, 0.882616157742669], 
            [0.9, 0.4, 0.8, 50.0, 950.0, 0.05, 4.43142902087969, 0.899528518736704, 0.868714102178279], 
            [0.9, 0.45, 0.8, 50.0, 950.0, 0.05, 5.17878341958552, 0.880134726393533, 0.922303975526518], 
            [1.1, 0.45, 1.2, 50.0, 750.0, 0.01, 3.77745284977611, 0.982646410137988, 0.903833694988684], 
            [1.1, 0.4, 1.2, 50.0, 750.0, 0.01, 3.25888845740284, 0.959519511929196, 0.898369409855066], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.18611105205991, 0.911338246361311, 0.952928319886093], 
            [0.9, 0.4, 0.8, 40.0, 950.0, 0.05, 0.393730359623324, 0.164808929120319, 0.421275475031881], 
            [0.9, 0.45, 0.8, 40.0, 950.0, 0.05, 0.726978211471111, 0.178234455366531, 0.639330200156856], 
            [1.1, 0.45, 1.2, 40.0, 750.0, 0.01, 4.04920219626234, 0.979125234365964, 0.972339587402194], 
            [1.1, 0.4, 1.2, 40.0, 750.0, 0.01, 3.41577085449181, 0.970920793624054, 0.930559605362218], 
            [0.9, 0.4, 0.8, 40.0, 950.0, 0.01, 0.066043886286129, 0.110433960488231, 0.105457624508451], 
            [0.9, 0.45, 0.8, 40.0, 950.0, 0.01, 0.0753278409246997, 0.0748908088788909, 0.157660350188917], 
            [1.1, 0.45, 1.2, 40.0, 750.0, 0.05, 4.02647095578344, 0.984301761227719, 0.961796198279483], 
            [1.1, 0.4, 1.2, 40.0, 750.0, 0.05, 3.45276213465927, 0.996876575294465, 0.916145682470408], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.13905395422309, 0.967537816232946, 0.887487458272303], 
            [0.9, 0.4, 0.8, 50.0, 950.0, 0.01, 0.412253855690025, 0.25702883186323, 0.282833530475158], 
            [0.9, 0.45, 0.8, 50.0, 950.0, 0.01, 1.29738824082812, 0.342646297906399, 0.593498194216583], 
            [1.1, 0.45, 1.2, 50.0, 750.0, 0.05, 4.06750373652383, 0.992213996233759, 0.963849794680717], 
            [1.1, 0.4, 1.2, 50.0, 750.0, 0.05, 3.45316562985823, 0.9998990162808, 0.913483145158041], 
            [1.1, 0.4, 0.8, 50.0, 750.0, 0.05, 0.506184617135736, 0.211690972063847, 0.42165243888218], 
            [1.1, 0.45, 0.8, 50.0, 750.0, 0.05, 1.00983223374774, 0.240306066490097, 0.658688351195126], 
            [0.9, 0.45, 1.2, 50.0, 950.0, 0.01, 3.67620043275219, 0.911794566979882, 0.947957636113733], 
            [0.9, 0.4, 1.2, 50.0, 950.0, 0.01, 3.20693291152293, 0.943047127982676, 0.899488774712609], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.30850473895881, 0.988951540867945, 0.903817297522009], 
            [1.1, 0.4, 0.8, 40.0, 750.0, 0.05, 0.162311482882376, 0.11471532291417, 0.249503030203975], 
            [1.1, 0.45, 0.8, 40.0, 750.0, 0.05, 0.257337552056493, 0.0946130437845141, 0.426331713366577], 
            [0.9, 0.45, 1.2, 40.0, 950.0, 0.01, 3.68012622833144, 0.883743861208712, 0.979090987068962], 
            [0.9, 0.4, 1.2, 40.0, 950.0, 0.01, 3.24364474418676, 0.926877873717665, 0.925656889278143], 
            [1.1, 0.4, 0.8, 40.0, 950.0, 0.01, 0.703238116866332, 0.298742779472285, 0.41510025095802], 
            [1.1, 0.45, 0.8, 40.0, 950.0, 0.01, 1.49947049446392, 0.359767333368841, 0.653298537225695], 
            [0.9, 0.45, 1.2, 40.0, 750.0, 0.05, 3.59827916754072, 0.90210565375335, 0.937830164497617], 
            [0.9, 0.4, 1.2, 40.0, 750.0, 0.05, 3.26773839242045, 0.928065922535278, 0.931338860494722], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.27971694059228, 0.97845030372438, 0.907413755959027], 
            [1.1, 0.4, 0.8, 50.0, 950.0, 0.01, 2.76337017893302, 0.623436672884356, 0.781618323466471], 
            [1.1, 0.45, 0.8, 50.0, 950.0, 0.01, 4.59117119769526, 0.780051434292237, 0.922562369198351], 
            [0.9, 0.45, 1.2, 50.0, 750.0, 0.05, 3.57009197585955, 0.90299739460285, 0.92956476871369], 
            [0.9, 0.4, 1.2, 50.0, 750.0, 0.05, 3.21705627661816, 0.931371227495853, 0.913640012064151], 
            [0.9, 0.4, 0.8, 50.0, 750.0, 0.05, 0.21252554442549, 0.141850994976415, 0.264196318736034], 
            [0.9, 0.45, 0.8, 50.0, 750.0, 0.05, 0.364460336670512, 0.138928084458227, 0.41120246821306], 
            [1.1, 0.45, 1.2, 50.0, 950.0, 0.01, 4.12940610715686, 0.998884615844266, 0.971983782552305], 
            [1.1, 0.4, 1.2, 50.0, 950.0, 0.01, 3.47900773621573, 0.999999999968302, 0.920226349342964], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.03, 4.27615151797595, 0.986077982945685, 0.899644457026274], 
            [0.9, 0.4, 0.8, 40.0, 750.0, 0.05, 0.0499179125994853, 0.0901770474137247, 0.0976131697742897], 
            [0.9, 0.45, 0.8, 40.0, 750.0, 0.05, 0.089939562407955, 0.0719588480139576, 0.195912459395661], 
            [1.1, 0.45, 1.2, 40.0, 950.0, 0.01, 4.11548867655713, 0.997574455417134, 0.969980132303631], 
            [1.1, 0.4, 1.2, 40.0, 950.0, 0.01, 3.59841614370975, 0.999999998706877, 0.951810862922007], 
            [1.0, 0.425, 1.0, 40.0, 850.0, 0.03, 3.1060913877519, 0.77729030065727, 0.829010566817183], 
            [1.0, 0.425, 1.0, 50.0, 850.0, 0.03, 4.01019270165031, 0.930980454126005, 0.893621819548353], 
            [1.0, 0.425, 1.0, 45.0, 750.0, 0.03, 3.03178887575184, 0.759196540594381, 0.828464340471849], 
            [1.0, 0.425, 1.0, 45.0, 950.0, 0.03, 4.18339780470682, 0.990874003560428, 0.875870330024656], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.05, 3.89140094628677, 0.893015099142378, 0.904016304626911], 
            [1.0, 0.425, 1.0, 45.0, 850.0, 0.01, 3.75804791189647, 0.87322433291318, 0.892823409606689], 
            [0.9, 0.425, 1.0, 45.0, 850.0, 0.03, 3.7204497635918, 0.891464352510675, 0.86580591497804], 
            [1.1, 0.425, 1.0, 45.0, 850.0, 0.03, 4.2042474377818, 0.971144057944103, 0.898118611326715], 
            [1.0, 0.45, 1.0, 45.0, 850.0, 0.03, 4.41382719021498, 0.958025779019591, 0.902700393475494], 
            [1.0, 0.4, 1.0, 45.0, 850.0, 0.03, 3.92180106186297, 0.911869589380138, 0.948005487812239], 
            [1.0, 0.425, 0.8, 45.0, 850.0, 0.03, 0.752846076390402, 0.243338104285562, 0.513470081095914], 
            [1.0, 0.425, 1.2, 45.0, 850.0, 0.03, 3.70172262777766, 0.983543486639247, 0.936959110245952]
        ])

        for (index, value) in {
                    0: experiment[XamControlExperimentRequest.RATIO_R1_SM1],
                    1: experiment[XamControlExperimentRequest.CONC_SM],
                    2: experiment[XamControlExperimentRequest.RESIDENCE_TIME], 
                    3: experiment[XamControlExperimentRequest.TEMPERATURE],
                    4: experiment[XamControlExperimentRequest.INTENSITY],
                    5: experiment[XamControlExperimentRequest.RATIO_R2_SM1]
                }.items():
            dataSet = dataSet[dataSet[:, index] == value]

            if dataSet.size == 0: 
                raise Exception("Data not found in dataset :/ - Note: only defined exp. r allowed (Idx:" + str(index) + ")")

        return XamControlExperimentResult(np.mean(dataSet[0, 6]), np.mean(dataSet[:, 7]), np.mean(dataSet[:, 8]), request=experiment)


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

        self.path = Path("./Tmp")
        self.xFileName = Path("xnewtrue.csv")
        self.yFileName = Path("ynewtrue.csv")

        self.numberOfExpectedValues = 1

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
            fileWriter.writerow(valuestoWrite)

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