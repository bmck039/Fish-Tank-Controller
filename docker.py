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
    previous_day = None
    while True:
        if previous_day != datetime.datetime.now().day: #check for update every day
            previous_day = datetime.datetime.now().day
            checkUpdate()
        time.sleep(60 * 60)