from datetime import datetime
from pathlib import Path
import csv, os

import numpy as np
from Common import Logger

importExportBasePath = Path("./Upload/import/")

def importData(fileContent):
    folder = importExportBasePath
    folder.mkdir(parents=True, exist_ok=True)

    f = open(folder / "importFile.csv", "w")
    f.write(fileContent)
    f.close()

def exportCurrentState(factorNames:list, experiments:np.array, responses:np.array):
    try:
        exportFolder = Logger.getCurrentLogFolder() / Path("Export_{}".format(datetime.now().strftime("%d%m%Y_%H")))
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
        Logger.logException(e)

    return None