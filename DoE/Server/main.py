import http.server
import socketserver
import json

class Server(http.server.SimpleHTTPRequestHandler):

    def do_GET(self):
        if self.path == "/":
            self.path = './assets/index.html'

        if ".json" in self.path: 
            return self.handleJSONRequest(self.path)
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

        return self.getJSONDefault()

    def getJSONDefault(self):
        return {"Error": "What do u need? 0.o"}

    def getJSONInfo(self):
        return { 
                "name": "TODO",
                "version": "1.0.0.0",
                "notes": "main/stable"
            }

if __name__ == "__main__":

    server = socketserver.TCPServer(('localhost', 8080), Server)
    server.serve_forever()
    server.server_close()