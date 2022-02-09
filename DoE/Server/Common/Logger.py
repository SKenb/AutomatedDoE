from pickle import TRUE
import traceback
import warnings
import logging
import sys
import csv
import os

from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np


logFolder = Path("./Logs")

# Initialize logging in one defines place
# import logging in all other files
#
# Use:
#   - logging.debug('...')
#   - logging.info('...')
#   - logging.warning('...')
#   - logging.error('...')
#
def initLogging():
    global logFolder

    dateString = datetime.now().strftime("%d%m%Y_%H.%M.%S")
    #hashString = str(random.getrandbits(32))
    subFolder = Path("Experiment_{}".format(dateString))
    appendToLogFolder(subFolder)

    logPath = Path(logFolder / "log_{}.log".format(datetime.now().strftime("%d%m%Y_%H")))

    logging.basicConfig(
        filename=str(logPath), 
        format='%(asctime)s\t%(levelname)s\t%(message)s',
        datefmt='%d.%m.%Y %I:%M:%S %p',
        level=logging.DEBUG
    )

    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    logging.getLogger('matplotlib.font_manager').disabled = True
    
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning) 

def getCurrentLogFolder():
    return logFolder

def getSubfoldersInLogFolder():

    def folderNameToString(walkStuff):
        folder = walkStuff.replace("Logs\\", "")
        
        if "\\" in folder: return None
        if "Logs" in folder: return None

        print(folder)
        parts = folder.split("_")
        date = parts[1]
        date = "{}.{}.{}".format(date[0:2], date[2:4], date[4:])
        time = parts[2].replace(".", ":")

        return "Exp - {} - {}".format(date, time)
        
    return [folderNameToString(x[0]) for x in os.walk(logFolder) if folderNameToString(x[0]) is not None]

def appendToLogFolder(newSubfolder:str):
    global logFolder

    newSubfolder = Path(newSubfolder)

    logFolder = logFolder / newSubfolder
    logFolder.mkdir(parents=False, exist_ok=True)


def genericLog(predicate : Callable, prefix : str, msg : str, suffix : str = "", stringBase : str = "{} {} {}"):
    predicate(stringBase.format(prefix, msg, suffix))

def logException(exceptionOrMsg):
    if isinstance(exceptionOrMsg, Exception): exceptionOrMsg = str(exceptionOrMsg)

    genericLog(logging.error, "[EXCEPTION]", exceptionOrMsg, "\n" + traceback.format_exc())


def logError(errorMSG):
    genericLog(logging.debug, "[ERR]", errorMSG)

def logDebug(debugMSG):
    genericLog(logging.debug, "[DEBUG]", debugMSG)

def logInfo(infoMsg):
    genericLog(logging.info, "[INFO]", infoMsg)

def logWarn(warnMsg):
    genericLog(logging.info, "[WARN]", warnMsg)

def logXamControl(msg):
    genericLog(logging.info, "\t[XamCtrl]", msg)

def logState(stateName):
    genericLog(logging.info, "(>)", stateName)

def logStateInfo(stateInfo, predicate=logging.info):
    genericLog(predicate, "\t-", stateInfo)


def logEntireRun(history, factorSet, experiments, responses, modelCoeffs, scaledModel, transformer):

    runNumber = len(history)

    with open(logFolder / "experiment_data_{}.csv".format(runNumber), 'w', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for expRespRow in np.append(experiments, responses, axis=1): fileWriter.writerow(expRespRow)


    with open(logFolder / "experiment_model_{}.csv".format(runNumber), 'w', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        fileWriter.writerow(["Details"]) 
        fileWriter.writerow([
            "Run", runNumber, 
            "DateTime", datetime.now().strftime("%d.%m.%Y %H:%M:%S")
        ]) 

        fileWriter.writerow([
            "Factor Set", 
            factorSet.getFactorString(False) +  factorSet.getCombinationsString(False)
        ]) 

        fileWriter.writerow(["Model Coeffs"])
        fileWriter.writerow(modelCoeffs)

        fileWriter.writerow(["Scaled Model Coeffs"])
        fileWriter.writerow(scaledModel.params)

        
        fileWriter.writerow(["Transformation"])
        fileWriter.writerow(["NO TRANSFORMATION" if transformer is None else str(transformer)])

        fileWriter.writerow(["Scaled Model - Summary"])
        fileWriter.writerow([str(scaledModel.summary())]) 
