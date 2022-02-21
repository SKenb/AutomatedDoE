import http.server
from pickle import TRUE
import socketserver
import json
import html
from tkinter.tix import Tree
from urllib.parse import urlparse
import threading
import time, os
import numpy as np

from Common import Logger
from Common import ImportExport
from Common import History
from Common import Statistics
from Common import Optimization
from XamControl import XamControl
from StateMachine import StateMachine
from StateMachine import DoE

from mainDoE import optimization

from Common.Factor import FactorSet, Factor, getDefaultFactorSet

writePath = readPath = "//TMP/TODO"
factorSet = getDefaultFactorSet()
processRunningFlag = processStopRequest = processPauseRequest = processIsPausingFlag = False
processThread = None
processState = "Ready"
processProgess = (0, 100)

class Server(http.server.SimpleHTTPRequestHandler):

    def do_POST(self):
        content_length = int(self.headers['Content-Length'])  # <--- Gets the size of data
        post_data = self.rfile.read(content_length)  # <--- Gets the data itself

        if "factors" in self.path:
            data = json.loads(post_data.decode('utf8').replace("'", '"'))
            factors = data["factors"]

            global factorSet
            factorSet = FactorSet([
                Factor(d["name"],d["min"], d["max"], d["unit"], d["symbol"]) 
                for d in factors
            ])


            self.send_response(200)
            self.send_header("Content-type", "text")
            self.end_headers()
            self.wfile.write(str(factorSet).encode())

        if "import/data" in self.path:
            prepData = post_data.decode('utf8')
            prepData = prepData.replace("\\n", "\n")
            prepData = prepData.replace("\\r", "")
            prepData = prepData.replace("\"", "")

            ImportExport.importData(prepData)
            
            self.send_response(200)
            self.send_header("Content-type", "text")
            self.end_headers()
            self.wfile.write("Imported".encode())



    def do_GET(self):
        if self.path == "/":
            self.path = './assets/index.html'

        if ".json" in self.path: 
            return self.handleJSONRequest(self.path)
        elif "update" in self.path:
            return self.updateData(self.path)
        elif "action" in self.path:
            return self.action(self.path)
        else:   
            return http.server.SimpleHTTPRequestHandler.do_GET(self)

    def handleJSONRequest(self, requestURL):
        
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()

        dataset = self.getJSONDataset(requestURL)
        self.wfile.write(json.dumps(dataset).encode("utf-8"))

    def getJSONDataset(self, requestURL):
        if "info" in requestURL: return self.getJSONInfo()
        if "defines" in requestURL: return self.getJSONDefines()
        if "process" in requestURL: return self.getJSONProcessInfo()
        if "server" in requestURL: return self.getJSONServerInfo()
        if "experiments" in requestURL: return self.getJSONExperiments()
        if "plots" in requestURL: return self.getJSONPlotInfo()
        if "import" in requestURL: return self.getJSONImportInfo()

        return self.getJSONDefault()
    
    def getJSONImportInfo(self):

        return ImportExport.importInfos()

    def getJSONPlotInfo(self):
        plots = Logger.getAvailablePlots(self.getLogFolderFromURL())

        return {
            "plots": plots,
            "hasPlots": len(plots) > 0
        }

    def getJSONServerInfo(self):
        return {"serverRunningFlag": True}

    def getJSONDefault(self):
        return {"Error": "What do u need? 0.o"}


    def getJSONExperiments(self):
        s = [{"experiment": f} for f in Logger.getSubfoldersInLogFolder()][::-1]
        return { 
                "experiments": s,
                "experimentsAvailable": len(s) > 0
            }   

    def getJSONInfo(self):
        return { 
                "name": "TODO",
                "version": "1.0.0.0",
                "notes": "main/stable"
            }
        
    def getJSONProcessInfo(self):
        global processPauseRequest, processRunningFlag, processStopRequest, processThread
        
        return { 
                "processRunningFlag": processRunningFlag,
                "processPauseRequest": processPauseRequest,
                "processIsPausing": processIsPausingFlag,
                "processStopRequest": processStopRequest,
                "processThread":  processThread is not None,
                "processState": processState,
                "processProgess": processProgess[0],
                "processProgessMax": processProgess[1],
            }

    def getJSONDefines(self):
        global factorSet

        return {
            "factors": [
                {
                    "name": f.name, 
                    "symbol": f.symbol,
                    "unit": f.unit, 
                    "min": f.min, 
                    "max": f.max
                } for f in factorSet.factors
            ],
            "readXamControl": readPath,
            "writeXamControl": writePath
        }

    def updateData(self, path):
        query = urlparse(path).query
        query = html.unescape(query)
        query = query.replace("%2F", "/")
        query_components = dict(qc.split("=") for qc in query.split("&"))
        
        successFlag = False

        global writePath, readPath
        def update_(keyWord, defaultValue):
            if keyWord in query_components:
                return True, query_components[keyWord]
            else:
                return False, defaultValue

        successFlag, writePath = update_("write", writePath)
        successFlag, readPath = update_("read", readPath)

        self.genericResponse({"success": successFlag })
       
        
    def action(self, path):
        global processRunningFlag, processThread, processStopRequest, processPauseRequest, processIsPausingFlag
        
        if "start" in path:
            # Start DoE
            
            if processStopRequest:
                self.genericResponse({"state": "Just stopping - then we can start again" })
                return False

            if processRunningFlag: 
                self.genericResponse({"state": "Process already running" })
                return False
            else:
                processPauseRequest = False
                processIsPausingFlag = False
                processRunningFlag = True
                print("Start DoE")
                
                processThread = threading.Thread(target=process)
                processThread.start()
                
                self.genericResponse({"state": "Process started" })
                
        if "stop" in path:
            # Stop DoE (@ l request it)
            
            if processThread is None:
                self.genericResponse({"state": "Nothing to stop :)" })
                return False
            
            if processStopRequest:
                self.genericResponse({"state": "Stop request already set" })
            else:
                processStopRequest = True
                self.genericResponse({"state": "Stop request sent" })
                
            
            return True
        
        if "pause" in path:            
            self.genericResponse({"state": "Already requested to pause" if processPauseRequest else "Pause request sent" })
            processPauseRequest = True
            
            return processPauseRequest
         
        if "resume" in path:
            
            self.genericResponse({"state": "Already requested to resume" if not processPauseRequest else "Resume request sent" })
            processPauseRequest = False
            
            return processPauseRequest   

        if "remove" in path:
            Logger.deleteLogFolder(self.getLogFolderFromURL())
            self.genericResponse({"state": "should be done" })
            return True

        if "export" in path:

            if DoE.context is None:
                self.genericResponse({"state": "fatal 0.o", "exportPath": None })
                return False

            path = ImportExport.exportCurrentState(
                DoE.context.factorSet.factors, 
                DoE.context._experimentValues, 
                DoE.context.Y
            )
            
            self.genericResponse({
                "state": "exported" if path is not None else "failed",
                "exportPath": str(path)
            })

            return path is not None

        if "deleteImport" in self.path:
            ImportExport.deleteCurrentImportFile()
            self.genericResponse({"state": "Yep - Should be done"})

        self.genericResponse({"state": "Yep - I don't know what u want from me"})



    def getLogFolderFromURL(self):
        logInfo = self.path.split("/")[-1]
        folderParts = logInfo.split("%20")

        return "Experiment_{}_{}".format(
                folderParts[2].replace(".", ""),
                folderParts[4].replace(":", ".")
            )

    def genericResponse(self, dataset):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        
        self.wfile.write(json.dumps(dataset).encode("utf-8"))

def process():
    global processStopRequest, processPauseRequest, processState, processProgess, factorSet
    

    def endProcess():
        onDone()
        return True

    def log(msg):
        global processState
        Logger.logInfo(msg)
        processState = msg

    def possibillityToPause():
        global processPauseRequest, processStopRequest, processIsPausingFlag

        while(processPauseRequest and (not processStopRequest)):
            processIsPausingFlag = True
            log("Pausing")
            time.sleep(2)

        processIsPausingFlag = False


    try:
        Logger.initLogging()
        Logger.logInfo("Start main DoE program")
        np.set_printoptions(suppress=True)

        log("Start main DoE")
        possibillityToPause()
        if processStopRequest: return endProcess()
        
        mainSM = StateMachine.StateMachine(DoE.InitDoE(setFactorSet=factorSet,setXAMControl=XamControl.XamControlTestRun1Mock()))
        for state in mainSM: 
            processState = str(state)  

            possibillityToPause()
            if processStopRequest: return endProcess()

        log("Find optimum")
        possibillityToPause()
        if processStopRequest: return endProcess()
        
        optimum = optimization(state.result())
        Logger.logInfo("Optimum @: {}".format(optimum))

        Statistics.plotContour(
            state.result().scaledModel, 
            getDefaultFactorSet(), 
            state.result().excludedFactors, 
            state.result().combinations
        )
        
        log("Start DoE around optimum")
        possibillityToPause()
        if processStopRequest: return endProcess()
        
        Logger.appendToLogFolder("DoE_Around_Optimum")
        mainSM = StateMachine.StateMachine(
            DoE.InitDoE(
                optimum=optimum,
                previousResult=state.result(),
                #previousContext=state.result().context,
                setXAMControl=XamControl.XamControlTestRun1RobustnessMock()
            )
        )

        for state in mainSM: 
            processState = str(state)

            possibillityToPause()
            if processStopRequest: return endProcess()
    
    except Exception as e:
        Logger.logError(str(e))
        time.sleep(2)
    
    onDone()
        
def onDone():
    global processRunningFlag, processStopRequest, processThread, processState
    
    Logger.closeLogging()

    processStopRequest = processRunningFlag = False
    processThread = None
    processState = "Ready"
    print("Process finished")  
    

if __name__ == "__main__":
    hostname, port = 'localhost', 8080
    server = socketserver.TCPServer((hostname, port), Server)

    if False:
        process()
        exit()
    
    print("Start DoE-Server @{}:{}".format(hostname, server))
    server.serve_forever()
    server.server_close()