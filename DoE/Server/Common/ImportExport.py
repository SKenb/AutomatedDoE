from datetime import datetime
from pathlib import Path
import csv, os, codecs

import numpy as np
from Common import Logger

importExportBasePath = Path("./Upload/import/")
importExportFile = importExportBasePath / "importFile.csv"


def importIsAvailable():
    return importExportFile.exists()

def importInfos():
    if not importIsAvailable(): return { "isAvailable": False }

    try:
        with open(importExportFile, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)      
            rows = [row for row in reader]
        
        factors = rows[0][0:rows[0].index('Response')]
        data = rows[1:]

        def parseFactorData(f):
            try:
                parts = f.split("**")
                return {
                    "name": parts[0],
                    "min": float(parts[1]),
                    "max": float(parts[2]),
                    "symbol": parts[3],
                    "unit": parts[4]
                }
            except Exception as e:
                Logger.logException(e)
                return {"name": "Error parsing data"} 


        return {
            "isAvailable": True,
            "factors": [parseFactorData(f) for f in factors],
            "factorCount": len(factors),
            "responseCount": len(rows[0]) - len(factors),
            #"raw": rows,
            #"data": data,
            "experiments": [[float(v) for v in d[0:len(factors)]] for d in data],
            "repsonse": [[float(v) for v in d[len(factors):]] for d in data],
            "dataCount": len(rows)-1
        }

    except Exception as e:
        Logger.logException(e)
        return None

def importData(fileContent):
    folder = importExportBasePath
    folder.mkdir(parents=True, exist_ok=True)

    f = codecs.open(importExportFile, "w", "utf-8")
    f.write(fileContent)
    f.close()

def deleteCurrentImportFile():
    os.remove(importExportFile)

def exportCurrentState(factorSet:list, experiments:np.array, responses:np.array):
    try:
        exportFolder = Logger.getCurrentLogFolder() / Path("Export_{}".format(datetime.now().strftime("%d%m%Y_%H")))
        exportFolder.mkdir(parents=False, exist_ok=True)
        exportFileName = "Export.csv"

        with open(exportFolder / exportFileName, 'w', newline='') as csvfile:
            fileWriter = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)

            factorNames = [
                "**".join([f.name, str(f.min), str(f.max), f.symbol, f.unit]) 
                for f in factorSet
            ]

            factorNames.extend(["Response"])
            factorNames.extend(["Additional" for _ in range(np.size(responses, 1)-1)])
            fileWriter.writerow(factorNames)

            for expRespRow in np.append(experiments, responses, axis=1): fileWriter.writerow(expRespRow)

        return exportFolder / exportFileName
    
    except Exception as e:
        Logger.logException(e)

    return None