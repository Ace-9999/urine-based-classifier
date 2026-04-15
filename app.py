from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Class labels
labels = {
    0: "Normal",
    1: "High Blood Sugar",
    2: "Kidney Stress",
    3: "Dehydration",
    4: "Infection Risk"
}

# Diet recommendations
diet = {
    0: [
        "✅ Maintain a balanced, nutrient-rich diet",
        "💧 Stay well-hydrated (8–10 glasses of water daily)",
        "🥦 Include plenty of vegetables and fruits",
        "🏃 Exercise regularly and maintain healthy weight",
        "🫘 Eat adequate protein from lean sources (chicken, legumes)",
        "🫐 Consume antioxidant-rich foods like berries and green tea",
        "🚫 Avoid excessive alcohol and smoking",
        "😴 Get 7–8 hours of quality sleep nightly"
    ],
    1: [
        "🚫 Reduce white rice and refined carbohydrates",
        "🌾 Eat more whole-grain roti, oats, and barley",
        "🍬 Strictly avoid sugar, sweets, and soft drinks",
        "🥗 Increase dietary fiber (vegetables, lentils, seeds)",
        "🍎 Choose low-glycemic fruits: guava, papaya, berries",
        "🥚 Include high-protein breakfast to stabilize blood sugar",
        "🥜 Snack on nuts (almonds, walnuts) instead of biscuits",
        "🕐 Eat small, frequent meals every 3–4 hours",
        "🧅 Add cinnamon and fenugreek to your diet (help lower glucose)",
        "🚰 Drink bitter gourd or amla juice on an empty stomach"
    ],
    2: [
        "🧂 Strictly reduce salt and sodium intake",
        "🥩 Control protein intake — prefer plant protein over animal",
        "🚫 Avoid processed and packaged foods (high sodium)",
        "🍟 Eliminate fried foods, pickles, and papadums",
        "💧 Drink adequate water but follow doctor's fluid advice",
        "🍋 Limit vitamin C supplements (monitor oxalate levels)",
        "🥛 Avoid excess dairy if creatinine is elevated",
        "🫑 Eat kidney-friendly veggies: cabbage, cauliflower, bell peppers",
        "🚫 Avoid spinach, tomatoes, and bananas (high potassium)",
        "🌿 Try herbal support: dandelion tea, parsley (consult doctor)"
    ],
    3: [
        "💧 Drink at least 2–3 liters of water daily",
        "🥥 Consume coconut water for natural electrolyte replenishment",
        "🍉 Eat water-rich fruits: watermelon, cucumber, oranges",
        "🧃 Drink fresh fruit juices and oral rehydration solutions (ORS)",
        "🍌 Eat bananas and avocados for potassium restoration",
        "🏠 Avoid diuretics like caffeine and alcohol",
        "🌡️ In hot weather, increase water intake by 500ml",
        "🍲 Include soups and broths in your daily meals",
        "🧂 Light salt intake to retain fluids effectively",
        "🌿 Try lemon water with a pinch of salt and sugar for quick hydration"
    ],
    4: [
        "💧 Increase fluid intake to flush out infection",
        "🍊 Eat vitamin C rich foods: oranges, amla, guava, kiwi",
        "🚫 Avoid junk food, fast food, and oily snacks",
        "🧄 Add garlic and ginger to meals (natural antimicrobials)",
        "🫐 Consume cranberry juice (helps prevent UTI bacteria adhesion)",
        "🥦 Eat iron-rich foods: spinach, lentils, fortified cereals",
        "🍯 Honey and turmeric milk to boost immunity",
        "😴 Rest adequately — immune recovery requires sleep",
        "🚫 Avoid sugar (feeds harmful bacteria)",
        "🌿 Probiotics like yogurt help restore gut and urinary flora",
        "💊 Take prescribed antibiotics/medications on time"
    ]
}

# Risk level info
risk_info = {
    0: {"level": "Low", "color": "#22c55e", "icon": "✅", "desc": "Your urine parameters appear normal. Keep up your healthy habits!"},
    1: {"level": "Moderate-High", "color": "#f59e0b", "icon": "⚠️", "desc": "Elevated blood sugar detected. Early dietary intervention is key."},
    2: {"level": "High", "color": "#ef4444", "icon": "🔴", "desc": "Signs of kidney stress detected. Consult a nephrologist soon."},
    3: {"level": "Moderate", "color": "#3b82f6", "icon": "💧", "desc": "Dehydration indicators detected. Increase your fluid intake immediately."},
    4: {"level": "Moderate-High", "color": "#a855f7", "icon": "🦠", "desc": "Possible infection risk. Seek medical attention and boost immunity."}
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Extract features in correct order
        features = [
            float(data.get("age", 0)),
            float(data.get("bp", 0)),
            float(data.get("sg", 1.020)),
            float(data.get("al", 0)),
            float(data.get("su", 0)),
            float(data.get("rbc", 0)),
            float(data.get("pc", 0)),
            float(data.get("pcc", 0)),
            float(data.get("ba", 0)),
            float(data.get("bgr", 0)),
            float(data.get("bu", 0)),
            float(data.get("sc", 0)),
            float(data.get("sod", 0)),
            float(data.get("pot", 0)),
            float(data.get("hemo", 0)),
            float(data.get("pcv", 0)),
            float(data.get("wc", 0)),
            float(data.get("rc", 0)),
            float(data.get("htn", 0)),
            float(data.get("dm", 0)),
            float(data.get("cad", 0)),
            float(data.get("appet", 0)),
            float(data.get("pe", 0)),
            float(data.get("ane", 0)),
            float(data.get("classification", 0)),
        ]
        
        input_array = np.array([features])
        prediction = int(model.predict(input_array)[0])
        proba = model.predict_proba(input_array)[0].tolist()
        
        return jsonify({
            "prediction": prediction,
            "label": labels[prediction],
            "diet": diet[prediction],
            "risk": risk_info[prediction],
            "probabilities": {labels[i]: round(p * 100, 1) for i, p in enumerate(proba)}
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
