import datetime
import time
import subprocess
import webServer

def checkUpdate():
    webServer.stopServer()
    subprocess.run(["git", "pull"])
    webServer.startServer()

if __name__ == "__main__":
    webServer.startServer()
    time.sleep(60 * (60 - datetime.datetime.now().minute))
    while True:
        checkUpdate()
        time.sleep(60 * 60)