import asyncio
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import subprocess
import BLE
import FishTankAITrain

hostName = "localhost"
port = 8080

webServer = None
sizeOfTank = 9 #gallons

def addLine(date, line):
   append = True
   file = open("data.csv", "r")
   lines = file.readlines()
   for i in range(len(lines)):
      if lines[i].split(",")[0] == date:
         append = False
         lines[i] = line
   if append:
      lines.append(line)
   file.close()
   file = open("data.csv", "w")
   for l in lines:
      file.write(l)
   file.close()

class Server(BaseHTTPRequestHandler):
  def do_GET(self):
    self.send_response(200)
    if self.path == "/":
      file = open("main.html", "r").read()
      self.send_header("Content-type", "text/html")
      self.end_headers()
      self.wfile.write(bytes(file, "utf-8"))
    elif self.path == "/save":
      self.send_header("Content-type", "application/json")
      self.end_headers()
      save = open("save.json").read()
      self.wfile.write(bytes(save, "utf-8"))
    elif self.path == "/train":
      #run FiahTankAITrain.py and send results to user
      ai = FishTankAITrain.Train()
      ai.setup()
      results = ai.predict()
      self.send_header("Content-type", "application/json")
      self.end_headers()
      results = json.dumps(results)
      self.wfile.write(bytes(results, "utf-8"))

  def do_PUT(self):
    self.send_response(200)
    self.end_headers()
    contentLen = int(self.headers['content-length'])
    byte = self.rfile.read(contentLen)
    string = ''.join(map(chr, byte))
    if self.path == "/waterParams":
      #add data point to file
      saveFile = open("data.csv", "a")
      saveFile.write("\n")
      values = json.loads(string)
      doseValues = [values[0], values[1], values[2], values[3], values[4], values[5], sizeOfTank, values[6]]
      stringVals = ""
      for i in range(len(doseValues)):
         stringVals += str(doseValues[i])
         if i != len(doseValues) - 1:
            stringVals += ","
      saveFile.close()
      addLine(values[0], stringVals)
      subprocess.run(["git", "add", "data.csv"])
      subprocess.run(["git", "commit", "-m", "added data point for " + datetime.datetime.now().strftime("%m/%d/%Y")])
      subprocess.run(["git", "push"])
    else:
      timeObjects = json.loads(string)
      save = open("save.json", "w")
      save.write("{ \"save\": " + string + "}")
      save.close()
      asyncio.run(BLE.sendSchedule(timeObjects)) # send to light over BLE

def startServer():
    webServer = HTTPServer((hostName, port), Server)
    print("Server started http://%s:%s" % (hostName, port))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass
    
    webServer.server_close()

def stopServer():
   if webServer:
        webServer.server_close()
        webServer = None