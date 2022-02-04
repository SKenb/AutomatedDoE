import http.server
from pickle import TRUE
import socketserver
import json
import html
from urllib.parse import urlparse

from Common.Factor import FactorSet, getDefaultFactorSet

writePath = readPath = "??"
factorSet = getDefaultFactorSet()

class Server(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self.path = './assets/index.html'

        if ".json" in self.path: 
            return self.handleJSONRequest(self.path)
        elif "update" in self.path:
            return self.updateData(self.path)
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

        return self.getJSONDefault()

    def getJSONDefault(self):
        return {"Error": "What do u need? 0.o"}

    def getJSONInfo(self):
        return { 
                "name": "TODO",
                "version": "1.0.0.0",
                "notes": "main/stable"
            }

    def getJSONDefines(self):
        global factorSet

        return {
            "factors": [
                {
                    "name": f.name, 
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


        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        dataset = {"success": successFlag }
        self.wfile.write(json.dumps(dataset).encode("utf-8"))


if __name__ == "__main__":

    server = socketserver.TCPServer(('localhost', 8080), Server)
    server.serve_forever()
    server.server_close()