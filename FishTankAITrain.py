import tensorflow as tf
import pandas as pd
import numpy as np
from datetime import date
import math

print("Tensorflow version:", tf.__version__)

num_epochs = 300
increment = 0.1

values = pd.read_csv("data.csv", names=["Day", "Phos", "N", "Phos Dose", "N Dose", "Water Change", "Tank Size", "Potassium"])

dates = values.get("Day")

phos = values.get("Phos")
nit = values.get("N")
phosDose = values.get("Phos Dose")
nitDose = values.get("N Dose")
waterChange = values.get("Water Change")
tankSize = values.get("Tank Size")
potDose = values.get("Potassium")

formatted_data = []
formatted_output = []

test_data = []
test_output = []

daysSincePotDose = 0
potInitDose = potDose.get(0)

def getNumEpochs(numDataPoints):
    return int( (numDataPoints + 4000) / (1 + math.exp((numDataPoints - 1000) / 900)) + 200)

def getDaySeparation(i, j):
    day1 = dates[i].split('/')
    day1 = date(int(day1[2]), int(day1[0]), int(day1[1]))
    day2 = dates[j].split('/')
    day2 = date(int(day2[2]), int(day2[0]), int(day2[1]))

    date_change = day2 - day1
    return date_change.days

def evaluatePhos(i, days, dose):
    if(phos.get(i) < 0):
        return evaluatePhos(i - 1, days + getDaySeparation(i - 1, i), dose + phosDose.get(i))
    return (phos.get(i), days, dose + phosDose.get(i))

def evaluateNit(i, days, dose):
    if(nit.get(i) < 0):
        return evaluateNit(i - 1, days + getDaySeparation(i - 1, i), dose + nitDose.get(i))
    return (nit.get(i), days, dose + nitDose.get(i))

for i in range(values.shape[0] - 1):
    if(phos.get(i + 1) >= 0 and nit.get(i + 1) >= 0):
        date_change = getDaySeparation(i, i + 1)

        if(potDose.get(i) == 0):
            daysSincePotDose += date_change
        else:
            daysSincePotDose = date_change
            potInitDose = potDose.get(i)


        phosVals = evaluatePhos(i, 0, 0)
        nitVals = evaluateNit(i, 0, 0)

        phosDoseToday = phosDose.get(i)
        nitDoseToday = nitDose.get(i)

        phos_arr = [n * increment for n in range(0, int(phosDoseToday / increment) + 1)]
        nit_arr = [n * increment for n in range(0, int(nitDoseToday / increment) + 1)]

        date_change_phos = date_change + phosVals[1]
        date_change_nit = date_change + nitVals[1]

        for end_phos_dose in phos_arr:
            for end_nit_dose in nit_arr:
                init_phos_dose = phosVals[2] - end_phos_dose
                init_nit_dose = nitVals[2] - end_nit_dose

                if end_nit_dose == nitDoseToday and end_phos_dose == phosDoseToday:
                    test_data.append(np.array([phosVals[0], nitVals[0], phos.get(i + 1), nit.get(i + 1), init_phos_dose, init_nit_dose, date_change_phos, date_change_nit, tankSize.get(i), potInitDose, daysSincePotDose, waterChange.get(i)]))
                    test_output.append(np.array([end_phos_dose, end_nit_dose]))
                else:
                    formatted_data.append(np.array([phosVals[0], nitVals[0], phos.get(i + 1), nit.get(i + 1), init_phos_dose, init_nit_dose, date_change_phos, date_change_nit, tankSize.get(i), potInitDose, daysSincePotDose, waterChange.get(i)]))
                    formatted_output.append(np.array([end_phos_dose, end_nit_dose]))

num_epochs = getNumEpochs(len(formatted_data))

formatted_data = np.array(formatted_data)
formatted_output = np.array(formatted_output)

test_data = np.array(test_data)
test_output = np.array(test_output)

def getLayerSize(inputNodes, output):
    return math.ceil(2/3 * inputNodes + output)

def getPatience(x):
    return math.floor(math.sqrt(x + 10) * 2/3)

hidden_layer_size = getLayerSize(formatted_data.shape[1], formatted_output.shape[1])
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Dense(hidden_layer_size, input_shape=(formatted_data.shape[1],)))
model.add(tf.keras.layers.Dense(hidden_layer_size))
model.add(tf.keras.layers.Dense(formatted_output.shape[1], activation=tf.nn.softplus))
model.build((None, formatted_data.shape[1]))
#creates model


model.compile(loss = tf.keras.losses.MeanSquaredError(), optimizer = tf.keras.optimizers.Adam())

print("Training on " + str(len(formatted_data)) + " data points for " + str(num_epochs) + " epochs with " + str(getPatience(num_epochs / 2)) + " patience")

earlyStop = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience = getPatience(num_epochs / 2), restore_best_weights = True, start_from_epoch = num_epochs / 2)

model.fit(formatted_data, formatted_output, epochs=num_epochs, verbose=1, validation_data=(test_data, test_output), use_multiprocessing = True)

model.save("test")

currentPhosVals = evaluatePhos(len(phos) - 1, 0, 0)
currentNitVals = evaluateNit(len(nit) - 1, 0, 0)

currentPhos = currentPhosVals[0]
currentNit = currentNitVals[0]

phosDateChange = currentPhosVals[1]
nitDateChange = currentNitVals[1]

current_phos_dose = currentPhosVals[2]
current_nit_dose = currentNitVals[2]

prev_date = dates[len(dates) - 1].split('/')
prev_date = date(int(prev_date[2]), int(prev_date[0]), int(prev_date[1]))
current_date_change = date.today() - prev_date

current_pot_dose = potDose[len(potDose) - 1]

current_size = tankSize[len(tankSize) - 1]

if(potDose[len(potDose) - 1] == 0):
    daysSincePotDose += current_date_change.days + 1
else:
    daysSincePotDose = 1
    potInitDose = potDose[len(potDose) - 1]

currentWaterChange = waterChange.get(len(waterChange) - 1)

predict_data = np.array([currentPhos, currentNit, 1.0, 30, current_phos_dose, current_nit_dose, current_date_change.days + phosDateChange + 1, current_date_change.days + nitDateChange + 1, current_size, potInitDose, daysSincePotDose, currentWaterChange])
predict_data = np.array([predict_data])

prediction = model.predict(predict_data)[0]

predicted_phos = max(round(prediction[0], 1), 0)
predicted_nit = max(round(prediction[1], 1), 0)

print("\n" + str(date.today()))

print("Phosphate Dose: " + str(predicted_phos) + " mL")
print("Nitrate Dose: " + str(predicted_nit) + " mL")

# tank.doseP(predicted_phos)
# tank.doseN(predicted_nit)
# tank.changeWater(predicted_water_change)