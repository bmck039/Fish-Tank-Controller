import asyncio
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from zeroconf import IPVersion, ServiceInfo, Zeroconf
from urllib.parse import urlparse, parse_qs
import socket
import json
import subprocess
import threading
import BLE
import FishTankAITrain

hostName = "0.0.0.0"
port = 80

ai = None

name = socket.gethostname()
ip_address = socket.gethostbyname(name)

#netsh interface portproxy add v4tov4 listenport=80 listenaddress=[ip] connectport=80 connectaddress=fishtank.local

ip_version = IPVersion.V4Only
info = ServiceInfo(
      "_http._tcp.local.",
      "FishTankAI._http._tcp.local.",
      addresses=[socket.inet_aton(ip_address)],
      port=80,
      properties={},
      server="fishtank.local.",
)

zeroconf = Zeroconf(ip_version=ip_version)
zeroconf.register_service(info)

webServer = None
sizeOfTank = 9 #gallons

def addLine(date, line):
   line += "\n"
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

  def train(self):
      global ai
      ai.setup()

  def do_GET(self):
    global ai
    fullPath = self.path
    self.path = urlparse(fullPath).path
    self.queries = parse_qs(urlparse(fullPath).query)
    if self.path == "/":
      file = open("main.html", "r").read()
      self.send_response(200)
      self.send_header("Content-type", "text/html")
      self.end_headers()
      self.wfile.write(bytes(file, "utf-8"))
    elif self.path == "/save":
      self.send_response(200)
      self.send_header("Content-type", "application/json")
      self.end_headers()
      save = open("save.json").read()
      self.wfile.write(bytes(save, "utf-8"))
    elif self.path == "/status":
       #run FiahTankAITrain.py and send results to user
      if ai != None:
         self.send_response(200)
         self.send_header("Content-type", "application/json")
         self.end_headers()
         results = ai.getProgress()
         results['trained'] = ai.isTrained()
         if ai.isTrained():
            results["predictions"] = ai.getOutput()
         results = json.dumps(results)
         self.wfile.write(bytes(results, "utf-8"))
      else:
         self.send_response(204)
         self.end_headers()
    elif self.path == "/waterParams":
       self.send_response(200)
       self.send_header("Content-type", "application/json")
       self.end_headers()
       print(self.queries)
       date = self.queries["date"][0]
       values = FishTankAITrain.getValuesFromDate(date)
       self.wfile.write(bytes(json.dumps(values), "utf-8"))

  def do_PUT(self):
    global ai
    self.send_response(200)
    self.end_headers()
    contentLen = int(self.headers['content-length'])
    byte = self.rfile.read(contentLen)
    string = ''.join(map(chr, byte))
    if self.path == "/waterParams":
      #add data point to file
      saveFile = open("data.csv", "a")
      # saveFile.write("\n")
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
      subprocess.run(["git", "commit", "-m", "added data point for " + values[0]])
      subprocess.run(["git", "push"])
    elif self.path == "/train":
      #run FiahTankAITrain.py and send results to user
      ai = FishTankAITrain.Train()
      
      thread = threading.Thread(daemon = True, target=self.train)
      thread.start()

    else:
      timeObjects = json.loads(string)
      save = open("save.json", "w")
      save.write("{ \"save\": " + string + "}")
      save.close()
      asyncio.run(BLE.sendSchedule(timeObjects)) # send to light over BLE

webServer = HTTPServer((hostName, port), Server)
print("Server started http://%s:%s" % (hostName, port))

try:
   webServer.serve_forever()
except KeyboardInterrupt:
   pass

webServer.server_close()