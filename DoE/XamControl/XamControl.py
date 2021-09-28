from XamControlExperiment import XamControlExperimentRequest as Request, XamControlExperimentResult as Result

import logging
import time

class XamControl:

    def __init__(self, name="Base class"):
        self.name = name

    def doExperiment(self, experiment : Request) -> Result:
        raise NotImplementedError("Use a derived class pls")

    def _startExperimentRequest(self, experiment : Request):
        logging.info("ExperimentRequest -> {}".format(str(experiment)))

    def _receivedExperimentResult(self, result : Result):
        logging.info("ExperimentRequest -> {}".format(str(result)))
    

class XamControlMock(XamControl):

    def __init__(self):
        super().__init__("Xam control - Mock")

    def doExperiment(self, experiment : Request, simulateExperimentTime = 0) -> Result:

        self._startExperimentRequest(experiment)
        
        if simulateExperimentTime > 0: time.sleep(simulateExperimentTime)

        experimentResult = Result(
            self._genericConversionModel(experiment, -0.335595, -0.098665, 0.9325, 0.00379995, 0.000531994, 0.00190386, 0.000248572),
            self._genericStyModel(experiment, 0.200104, -0.0789001, -0.99375, -0.00594246, 0.00201024, 0.023325)
        )

        self._receivedExperimentResult(experimentResult)
        return experimentResult

    def _genericConversionModel(self, exp : Request, const, ratio, conc, ResT, Temp, Ra2Temp, ResTemp):

        return const \
            + ratio * exp[Request.REAGENTRATIO] \
            + conc * exp[Request.CONCENTRATION] \
            + ResT * exp[Request.RESIDENCETIME] \
            + Temp * exp[Request.TEMPERATURE] \
            + Ra2Temp * exp[Request.REAGENTRATIO]**2 * exp[Request.TEMPERATURE] \
            + ResTemp * exp[Request.RESIDENCETIME] * exp[Request.TEMPERATURE]  \

    def _genericStyModel(self, exp : Request, const, ratio, conc, Temp, Ra2Temp, ConTemp):

        return const \
            + ratio * exp[Request.REAGENTRATIO] \
            + conc * exp[Request.CONCENTRATION] \
            + Temp * exp[Request.TEMPERATURE] \
            + Ra2Temp * exp[Request.REAGENTRATIO]**2 * exp[Request.TEMPERATURE] \
            + ConTemp * exp[Request.CONCENTRATION] * exp[Request.TEMPERATURE]  \


if __name__ == "__main__":

    print(" Test XamControlMock ".center(80, "-"))

    XamControl = XamControlMock()

    exp = Request(60, .2, .8, 10)

    print(XamControl.doExperiment(exp))