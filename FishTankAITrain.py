import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import date
import math

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
        self.num_epochs = 300
        self.increment = 0.1
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
        for i in range(self.values.shape[0] - 1):
            if(self.phos.get(i + 1) >= 0 and self.nit.get(i + 1) >= 0):
                date_change = self.getDaySeparation(i, i + 1)

                if(self.potDose.get(i) == 0):
                    self.daysSincePotDose += date_change
                else:
                    self.daysSincePotDose = date_change
                    self.potInitDose = self.potDose.get(i)


                phosVals = self.evaluatePhos(i, 0, 0)
                nitVals = self.evaluateNit(i, 0, 0)

                phosDoseToday = self.phosDose.get(i)
                nitDoseToday = self.nitDose.get(i)

                phos_arr = [n * self.increment for n in range(0, int(phosDoseToday / self.increment) + 1)]
                nit_arr = [n * self.increment for n in range(0, int(nitDoseToday / self.increment) + 1)]

                date_change_phos = date_change + phosVals[1]
                date_change_nit = date_change + nitVals[1]

                for end_phos_dose in phos_arr:
                    for end_nit_dose in nit_arr:
                        init_phos_dose = phosVals[2] - end_phos_dose
                        init_nit_dose = nitVals[2] - end_nit_dose

                        if end_nit_dose == nitDoseToday and end_phos_dose == phosDoseToday:
                            self.test_data.append(np.array([phosVals[0], nitVals[0], self.phos.get(i + 1), self.nit.get(i + 1), init_phos_dose, init_nit_dose, date_change_phos, date_change_nit, self.tankSize.get(i), self.potInitDose, self.daysSincePotDose, self.waterChange.get(i)]))
                            self.test_output.append(np.array([end_phos_dose, end_nit_dose]))
                        else:
                            self.formatted_data.append(np.array([phosVals[0], nitVals[0], self.phos.get(i + 1), self.nit.get(i + 1), init_phos_dose, init_nit_dose, date_change_phos, date_change_nit, self.tankSize.get(i), self.potInitDose, self.daysSincePotDose, self.waterChange.get(i)]))
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
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_layer_size, input_shape=(self.formatted_data.shape[1],)))
        self.model.add(tf.keras.layers.Dense(hidden_layer_size))
        self.model.add(tf.keras.layers.Dense(self.formatted_output.shape[1], activation=tf.nn.softplus))
        self.model.build((None, self.formatted_data.shape[1]))
        self.num_epochs = self.getNumEpochs(len(self.formatted_data))
        #creates model

    def train(self):
        self.model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())

        print("Training on " + str(len(self.formatted_data)) + " data points for " + str(self.num_epochs) + " epochs with " + str(self.getPatience(self.num_epochs / 2)) + " patience")

        earlyStop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = self.getPatience(self.num_epochs / 2), restore_best_weights = True, start_from_epoch = self.num_epochs / 2)

        self.model.fit(self.formatted_data, self.formatted_output, epochs=self.num_epochs, verbose=1, validation_data=(self.test_data, self.test_output), use_multiprocessing = True)

    def save(self):
        self.model.save("test")

    def predict(self):
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

        predict_data = np.array([currentPhos, currentNit, 1.0, 30, current_phos_dose, current_nit_dose, current_date_change.days + phosDateChange + 1, current_date_change.days + nitDateChange + 1, current_size, self.potInitDose, self.daysSincePotDose, currentWaterChange])
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
        self.train()
# tank.doseP(predicted_phos)
# tank.doseN(predicted_nit)
# tank.changeWater(predicted_water_change)