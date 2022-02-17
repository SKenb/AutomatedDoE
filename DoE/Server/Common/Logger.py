from itertools import combinations
from pickle import TRUE
import traceback
import warnings
import logging
import shutil
import sys
import csv
import os

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List

import numpy as np


logBasePath = Path("./Logs")
logFolder = logBasePath

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
    global logFolder, logBasePath
    logFolder = logBasePath

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

def importData(fileContent):
    folder = logBasePath / Path("../Upload/import/")
    folder.mkdir(parents=True, exist_ok=True)

    f = open(folder / "importFile.csv", "w")
    f.write(fileContent)
    f.close()

def getAvailablePlots(logFolder):
    folder = logBasePath / Path(logFolder)


    def getFileInfos(folder, aroundOptimumFlag = False):
        if not os.path.isdir(folder): return []

        filelist=os.listdir(folder)

        for fichier in filelist[:]:
            if not(fichier.endswith(".png")):
                filelist.remove(fichier)

        return [{
                "path": str(folder / Path(f)), 
                "name": str(f),
                "cleanName": ("🌟 " if aroundOptimumFlag else "") + cleanName(str(f)),
                "optimum": aroundOptimumFlag,
            } for f in filelist
        ]

    def cleanName(name, addIcons=True):
        name = name.replace(".png", "")
        name = name.replace("Plot_", "Result ")
        name = name.replace("Score_", "Result ")

        if addIcons:
            if "Exp_" in name: name = "📊 " + name
            if "Resp_" in name: name = "🔬 " + name
            if "Best" in name: name = "📍 " + name
            if "Robustness" in name: name = "🎯 " + name
            if "Contour" in name: name = "💠 " + name

        return name


    filelist = getFileInfos(folder)
    filelist.extend(getFileInfos(folder / Path('DoE_Around_Optimum'), True))

    return filelist

def getSubfoldersInLogFolder():

    def folderNameToString(walkStuff):
        folder = walkStuff.replace("Logs\\", "")
        
        if "\\" in folder: return None
        if "Logs" in folder: return None

        parts = folder.split("_")
        date = parts[1]
        date = "{}.{}.{}".format(date[0:2], date[2:4], date[4:])
        time = parts[2].replace(".", ":")

        return "Exp - {} - {}".format(date, time)
        
    return [folderNameToString(x[0]) for x in os.walk(logBasePath) if folderNameToString(x[0]) is not None]

def closeLogging():
    #logging.shutdown()
    for h in logging.getLogger().handlers[:]:
        logging.getLogger().removeHandler(h)

def appendToLogFolder(newSubfolder:str):
    global logFolder

    newSubfolder = Path(newSubfolder)

    logFolder = logFolder / newSubfolder
    logFolder.mkdir(parents=False, exist_ok=True)

def deleteLogFolder(folderName:str):
    global logBasePath
    shutil.rmtree(logBasePath / Path(folderName))

def exportCurrentState(factorNames:list, experiments:np.array, responses:np.array):
    try:
        exportFolder = logFolder / Path("Export_{}".format(datetime.now().strftime("%d%m%Y_%H")))
        exportFolder.mkdir(parents=False, exist_ok=True)
        exportFileName = "Export.csv"

        with open(exportFolder / exportFileName, 'w', newline='') as csvfile:
            fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            factorNames.extend(["Response"])
            factorNames.extend(["Additional" for _ in range(np.size(responses, 1)-1)])
            fileWriter.writerow(factorNames)

            for expRespRow in np.append(experiments, responses, axis=1): fileWriter.writerow(expRespRow)

        return exportFolder / exportFileName
    
    except Exception as e:
        logException(e)

    return None


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


def logEntireRun(history, factorSet, excludedFactors, experiments, responses, modelCoeffs, scaledModel, transformer):

    runNumber = len(history)

    with open(logFolder / "experiment_data_{}.csv".format(runNumber), 'w', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        for expRespRow in np.append(experiments, responses, axis=1): fileWriter.writerow(expRespRow)


    with open(logFolder / "experiment_model_{}.csv".format(runNumber), 'w', newline='') as csvfile:
        fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

        def insertNewLine(cnt=1):
            for _ in range(cnt): fileWriter.writerow([]) 

        fileWriter.writerow([
            "Experiment-Iteration: {} on the {}".format(runNumber, datetime.now().strftime("%d.%m.%Y %H:%M:%S"))
        ]) 

        insertNewLine(2)

        fileWriter.writerow(["Factor Set"]) 
        insertNewLine()
        fileWriter.writerow(["0", "Contant", "", "", "", ""]) 
        for index, factor in enumerate(factorSet.factors):
            fileWriter.writerow([chr(65 + (index % 26)), factor.name, str(factor.min), str(factor.max), factor.unit, factor.symbol]) 

        insertNewLine(2)

        fileWriter.writerow(["Excluded factors"]) 
        insertNewLine()
        if len(excludedFactors) <= 0: fileWriter.writerow(["-"]) 
        for index in excludedFactors:
            fileWriter.writerow([chr(65 + (index % 26))]) 

        insertNewLine(2)

        fileWriter.writerow(["Combinations"]) 
        insertNewLine()
        if len(factorSet.experimentValueCombinations) <= 0: fileWriter.writerow(["-"]) 
        for combi in factorSet.experimentValueCombinations.keys():
            fileWriter.writerow([combi]) 

        
        insertNewLine(2)

        abcList = ["0"]
        abcList.extend([chr(65 + (index % 26)) for index in range(len(factorSet)) if index not in excludedFactors])
        abcList.extend(factorSet.experimentValueCombinations.keys())

        fileWriter.writerow(["Model Coeffs"])
        fileWriter.writerow(abcList)
        fileWriter.writerow(modelCoeffs)

        
        insertNewLine()

        fileWriter.writerow(["Scaled Model Coeffs"])
        fileWriter.writerow(abcList)
        fileWriter.writerow(scaledModel.params)


        insertNewLine(2)

        fileWriter.writerow(["Transformation"])
        fileWriter.writerow(["NO TRANSFORMATION" if transformer is None else str(transformer)])


        insertNewLine(2)

        fileWriter.writerow(["Scaled Model - Summary"])
        fileWriter.writerow([str(scaledModel.summary())]) 
