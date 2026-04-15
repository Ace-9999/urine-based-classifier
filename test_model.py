import pickle
import numpy as np

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

sample_input = np.array([[
    45,   # age
    85,   # bp
    1.020, # sg (normal)
    0,    # al (normal → avoid kidney trigger)
    2,    # su (HIGH → sugar trigger)
    0,0,0,0,
    180,  # bgr (HIGH → sugar trigger)
    30,   # bu (normal)
    1.0,  # sc (normal)
    140,  # sodium
    4.0,  # potassium
    14,   # hemo
    42,   # pcv
    7000, # wc
    5.0,  # rc
    0,0,0,0,0,0,0
]])

prediction = model.predict(sample_input)

labels = {
    0: "Normal",
    1: "High Blood Sugar",
    2: "Kidney Stress",
    3: "Dehydration",
    4: "Infection Risk"
}

print("Prediction:", labels[prediction[0]])