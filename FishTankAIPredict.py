import tensorflow as tf
import numpy as np
from datetime import date
import pandas as pd

print("Tensorflow version:", tf.__version__)

model = tf.keras.models.load_model("test")

values = pd.read_csv("data.csv", names=["Day", "Phos", "N", "Phos Dose", "N Dose", "Water Change", "Tank Size", "Potassium"])

dates = values.get("Day")

phos = values.get("Phos")
nit = values.get("N")
phosDose = values.get("Phos Dose")
nitDose = values.get("N Dose")
waterChange = values.get("Water Change")
tankSize = values.get("Tank Size")
potDose = values.get("Potassium")

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

daysSincePotDose = 0
potCurrentDose = potDose.get(0)

for i in range(values.shape[0] - 1):
    date_change = getDaySeparation(i, i + 1)

    if(potDose.get(i) == 0):
        daysSincePotDose += date_change
    else:
        daysSincePotDose = date_change
        potCurrentDose = potDose.get(i)

phosVals = evaluatePhos(len(phos) - 1, 0, 0)
nitVals = evaluateNit(len(nit) - 1, 0, 0)

currentPhos = phosVals[0]
currentNit = nitVals[0]

prev_date = dates[len(dates) - 1].split('/')
prev_date = date(int(prev_date[2]), int(prev_date[0]), int(prev_date[1]))
current_date_change = date.today() - prev_date

phosDateChange = phosVals[1]
nitDateChange = nitVals[1]

current_phos_dose = phosVals[2]
current_nit_dose = nitVals[2]

current_size = tankSize[len(tankSize) - 1]

if(potDose[len(potDose) - 1] == 0):
    daysSincePotDose += current_date_change.days + 1
else:
    daysSincePotDose = 1
    potCurrentDose = potDose[len(potDose) - 1]

predict_data1 = np.array([currentPhos, currentNit, 1.0, 30, current_phos_dose, current_nit_dose, current_date_change.days + phosDateChange + 1, current_date_change.days + nitDateChange + 1, current_size, potCurrentDose, daysSincePotDose])

predict_data = np.array([predict_data1])

prediction = model.predict(predict_data)[0]

predicted_phos = max(round(prediction[0], 1), 0)
predicted_nit = max(round(prediction[1], 1), 0)
predicted_water_change = max(round(prediction[2], 1), 0)

print("\n" + str(date.today()))

print("Phosphate Dose: " + str(predicted_phos) + " mL")
print("Nitrate Dose: " + str(predicted_nit) + " mL")
print("Water Change: " + str(predicted_water_change) + " gallons")