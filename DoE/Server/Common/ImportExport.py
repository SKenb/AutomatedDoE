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

        infoRowCount = 5
        factorCount = rows[0].index('Response')
        
        factorData = [r[0:factorCount] for r in rows[0:infoRowCount]]
        data = rows[infoRowCount:]

        def parseFactorData(f):
            try:
                return {
                    "name": f[0],
                    "min": float(f[1]),
                    "max": float(f[2]),
                    "symbol": f[3],
                    "unit": f[4]
                }
            except Exception as e:
                Logger.logException(e)
                return {"name": "Error parsing data"} 


        return {
            "isAvailable": True,
            "factors": [parseFactorData(f) for f in list(map(list, zip(*factorData)))],
            "factorCount": factorCount,
            "responseCount": len(rows[0]) - factorCount,
            #"raw": rows,
            #"data": data,
            "experiments": [[float(v) for v in d[0:factorCount]] for d in data],
            "repsonse": [[float(v) for v in d[factorCount:]] for d in data],
            "dataCount": len(rows)-infoRowCount
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

            factorNames = [f.name for f in factorSet]

            factorNames.extend(["Response"])
            factorNames.extend(["Additional" for _ in range(np.size(responses, 1)-1)])
            fileWriter.writerow(factorNames)

            for predicate in [
                lambda f: str(f.min), 
                lambda f: str(f.max), 
                lambda f: f.symbol, 
                lambda f: f.unit, 
            ]:
                fileWriter.writerow([predicate(f) for f in factorSet])

            for expRespRow in np.append(experiments, responses, axis=1): fileWriter.writerow(expRespRow)

        return exportFolder / exportFileName
    
    except Exception as e:
        Logger.logException(e)

    return None