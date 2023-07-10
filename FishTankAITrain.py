import threading
import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from datetime import date
import math

print("Tensorflow Version: ", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

def getValuesFromDate(date):
    values = pd.read_csv("data.csv", names=["Day", "Phos", "N", "Phos Dose", "N Dose", "Water Change", "Tank Size", "Potassium"])
    try:
        index = values['Day'].tolist().index(date)
        return values.loc[index].to_dict()
    except ValueError:
        return {}

class Callbacks(keras.callbacks.Callback):

    def __init__(self, callback, numEpochs):
        self.callback = callback
        self.numEpochs = numEpochs
        super(Callbacks, self).__init__()

    def on_epoch_end(self, epoch, logs={}):
        self.updateProgress(self.callback, epoch, self.numEpochs)

    def updateProgress(self, callback, epoch, numEpochs):
        callback(epoch, numEpochs)

class Train:

    values = None
    num_epochs = None
    increment = None
    dates = None

    phos = None
    nit = None
    phosDose = None
    nitDose = None
    waterChange = None
    tankSize = None
    potDose = None

    formatted_data = None
    formatted_output = None

    test_data = None
    test_output = None

    daysSincePotDose = None
    potInitDose = None

    def __init__(self):
        self.values = pd.read_csv("data.csv", names=["Day", "Phos", "N", "Phos Dose", "N Dose", "Water Change", "Tank Size", "Potassium"])
        self.num_epochs = 10
        self.increment = 0.1
        self.numDaysIncluded = 7

        self.dates = self.values.get("Day")
        self.phos = self.values.get("Phos")
        self.nit = self.values.get("N")
        self.phosDose = self.values.get("Phos Dose")
        self.nitDose = self.values.get("N Dose")
        self.waterChange = self.values.get("Water Change")
        self.tankSize = self.values.get("Tank Size")
        self.potDose = self.values.get("Potassium")

        self.formatted_data = []
        self.formatted_output = []

        self.test_data = []
        self.test_output = []

        self.daysSincePotDose = 0
        self.potInitDose = self.potDose.get(0)
        self.training = False
        self.trained = False

        self.progress = ""
        self.percent = 0

        self.output = {}
        self.numTrained = 0
        self.numRepetitions = 5
        self.history = None


    def isTrained(self):
        return self.trained

    def getNumEpochs(self, numDataPoints):
        return int( (numDataPoints + 4000) / (1 + math.exp((numDataPoints - 1000) / 900)) + 200)

    def getDaySeparation(self, i, j):
        day1 = self.dates[i].split('/')
        day1 = date(int(day1[2]), int(day1[0]), int(day1[1]))
        day2 = self.dates[j].split('/')
        day2 = date(int(day2[2]), int(day2[0]), int(day2[1]))

        date_change = day2 - day1
        return date_change.days

    def evaluatePhos(self, i, days, dose):
        if(self.phos.get(i) < 0):
            return self.evaluatePhos(i - 1, days + self.getDaySeparation(i - 1, i), dose + self.phosDose.get(i))
        return (self.phos.get(i), days, dose + self.phosDose.get(i))

    def evaluateNit(self, i, days, dose):
        if(self.nit.get(i) < 0):
            return self.evaluateNit(i - 1, days + self.getDaySeparation(i - 1, i), dose + self.nitDose.get(i))
        return (self.nit.get(i), days, dose + self.nitDose.get(i))

    def formatData(self):
        for i in range(self.numDaysIncluded, self.values.shape[0] - 1):
            if(self.phos.get(i + 1) >= 0 and self.nit.get(i + 1) >= 0):
                date_change = self.getDaySeparation(i, i + 1)

                phosVals = self.evaluatePhos(i, 0, 0)
                nitVals = self.evaluateNit(i, 0, 0)

                phosDoseToday = self.phosDose.get(i)
                nitDoseToday = self.nitDose.get(i)

                dateVals = [list(self.evaluatePhos(i - j, 0, 0)) + list(self.evaluateNit(i - j, 0, 0)) + list((self.potDose.get(i - j), self.getDaySeparation(i - j, i))) + [self.waterChange.get(i - j)] for j in range(self.numDaysIncluded, 0, -1)]
                dateVals = [j for sub in dateVals for j in sub]

                phos_arr = [n * self.increment for n in range(0, int(phosDoseToday / self.increment) + 1)]
                nit_arr = [n * self.increment for n in range(0, int(nitDoseToday / self.increment) + 1)]

                date_change_phos = date_change + phosVals[1]
                date_change_nit = date_change + nitVals[1]

                for end_phos_dose in phos_arr:
                    for end_nit_dose in nit_arr:
                        init_phos_dose = phosVals[2] - end_phos_dose
                        init_nit_dose = nitVals[2] - end_nit_dose

                        if end_nit_dose == nitDoseToday and end_phos_dose == phosDoseToday:
                            self.test_data.append(np.array(list(dateVals + [phosVals[0], nitVals[0], self.phos.get(i + 1), self.nit.get(i + 1), init_phos_dose, init_nit_dose, date_change_phos, date_change_nit, self.tankSize.get(i), self.potDose.get(i), date_change, self.waterChange.get(i)])))
                            self.test_output.append(np.array([end_phos_dose, end_nit_dose]))
                        else:
                            self.formatted_data.append(np.array(list(dateVals + [phosVals[0], nitVals[0], self.phos.get(i + 1), self.nit.get(i + 1), init_phos_dose, init_nit_dose, date_change_phos, date_change_nit, self.tankSize.get(i), self.potDose.get(i), date_change, self.waterChange.get(i)])))
                            self.formatted_output.append(np.array([end_phos_dose, end_nit_dose]))

        self.formatted_data = np.array(self.formatted_data)
        self.formatted_output = np.array(self.formatted_output)

        self.test_data = np.array(self.test_data)
        self.test_output = np.array(self.test_output)

    def getLayerSize(self, inputNodes, output):
        return math.ceil(2/3 * inputNodes + output)

    def getPatience(self, x):
        return math.floor(math.sqrt(x + 10) * 2/3)

    def createModel(self):
        self.formatData()
        hidden_layer_size = self.getLayerSize(self.formatted_data.shape[1], self.formatted_output.shape[1])
        hidden_layer_size2 = self.getLayerSize(hidden_layer_size, self.formatted_output.shape[1])
        self.model = keras.models.Sequential()
        self.model.add(keras.layers.Dense(hidden_layer_size, input_shape=(self.formatted_data.shape[1],)))
        self.model.add(keras.layers.Dense(hidden_layer_size2))
        self.model.add(keras.layers.Dense(self.formatted_output.shape[1], activation=tf.nn.softplus))
        self.model.build((None, self.formatted_data.shape[1]))
        # self.num_epochs = self.getNumEpochs(len(self.formatted_data))
        self.trained = False
        #creates model

    def setProgress(self, epoch, numEpochs):
        self.progress = str(epoch + 1) + "/" + str(numEpochs) + " " + str(int(100 * (epoch + 1)/ numEpochs)) + "%"
        self.percent = (self.numTrained * numEpochs + epoch + 1) / (self.numRepetitions * numEpochs) * 100

    def getProgress(self):
        return { "progress": self.progress, "percent": self.percent }

    def train(self):
        self.training = True
        self.model.compile(loss = keras.losses.MeanSquaredError(), optimizer = keras.optimizers.Adam(), metrics=['accuracy'])

        print("Training on " + str(len(self.formatted_data)) + " data points for " + str(self.num_epochs) + " epochs with " + str(self.getPatience(self.num_epochs / 2)) + " patience")

        earlyStop = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience = self.getPatience(self.num_epochs / 2), restore_best_weights = True, start_from_epoch = self.num_epochs / 2)

        self.history = self.model.fit(self.formatted_data, self.formatted_output, epochs=self.num_epochs, verbose=1, validation_data=(self.test_data, self.test_output), use_multiprocessing = True, callbacks=[Callbacks(self.setProgress, self.num_epochs), earlyStop])
        self.training = False
        # self.trained = True
        

    def save(self):
        self.model.save("test")

    def predict(self):
        i = len(self.phos) - 1

        dateVals = [list(self.evaluatePhos(i - j, 0, 0)) + list(self.evaluateNit(i - j, 0, 0)) + list((self.potDose.get(i - j), self.getDaySeparation(i - j, i))) + [self.waterChange.get(i - j)] for j in range(self.numDaysIncluded, 0, -1)]
        dateVals = [j for sub in dateVals for j in sub]
        currentPhosVals = self.evaluatePhos(len(self.phos) - 1, 0, 0)
        currentNitVals = self.evaluateNit(len(self.nit) - 1, 0, 0)

        currentPhos = currentPhosVals[0]
        currentNit = currentNitVals[0]

        phosDateChange = currentPhosVals[1]
        nitDateChange = currentNitVals[1]

        current_phos_dose = currentPhosVals[2]
        current_nit_dose = currentNitVals[2]

        prev_date = self.dates[len(self.dates) - 1].split('/')
        prev_date = date(int(prev_date[2]), int(prev_date[0]), int(prev_date[1]))
        current_date_change = date.today() - prev_date

        current_size = self.tankSize[len(self.tankSize) - 1]

        if(self.potDose[len(self.potDose) - 1] == 0):
            self.daysSincePotDose += current_date_change.days + 1
        else:
            self.daysSincePotDose = 1
            self.potInitDose = self.potDose[len(self.potDose) - 1]

        currentWaterChange = self.waterChange.get(len(self.waterChange) - 1)

        predict_data = np.array(dateVals + [currentPhos, currentNit, 1.0, 20, current_phos_dose, current_nit_dose, current_date_change.days + phosDateChange + 1, current_date_change.days + nitDateChange + 1, current_size, self.potDose.get(i), current_date_change.days, currentWaterChange])
        predict_data = np.array([predict_data])

        prediction = self.model.predict(predict_data)[0]

        predicted_phos = max(round(prediction[0], 1), 0)
        predicted_nit = max(round(prediction[1], 1), 0)

        print("\n" + str(date.today()))

        print("Phosphate Dose: " + str(predicted_phos) + " mL")
        print("Nitrate Dose: " + str(predicted_nit) + " mL")

        return { "phos": str(predicted_phos), "nit": str(predicted_nit) }

    def setup(self):
        self.createModel()
        thread = threading.Thread(daemon = True, target=self.multiTrain)
        thread.start()
    
    def avg(self, A, B):
        return round(sum([x * y for x, y in zip(A, B)]) / sum(B), 1)

    def multiTrain(self):
        self.trained = False
        self.numTrained = 0
        outputs = {k: [] for k in ["phos", "nit"]}
        accuracies = []
        for i in range(self.numRepetitions):
            self.train()
            output = self.predict()
            self.numTrained += 1
            outputs = {k: outputs[k] + [float(output[k])] for k in outputs}
            accuracies.append(self.history.history["accuracy"][-1])
        self.output = {k: self.avg([v for v in outputs[k]], accuracies) for k in outputs}
        self.trained = True
    
    def getOutput(self):
        return self.output
# tank.doseP(predicted_phos)
# tank.doseN(predicted_nit)
# tank.changeWater(predicted_water_change)