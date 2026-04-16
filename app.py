from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"))

# Load urine ML model
with open(os.path.join(BASE_DIR, "model.pkl"), "rb") as f:
    model = pickle.load(f)

# ─── URINE CLASSIFIER ─────────────────────────────────────────────────────────

labels = {
    0: "Normal",
    1: "High Blood Sugar",
    2: "Kidney Stress",
    3: "Dehydration",
    4: "Infection Risk"
}

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

risk_info = {
    0: {"level": "Low", "color": "#22c55e", "icon": "✅", "desc": "Your urine parameters appear normal. Keep up your healthy habits!"},
    1: {"level": "Moderate-High", "color": "#f59e0b", "icon": "⚠️", "desc": "Elevated blood sugar detected. Early dietary intervention is key."},
    2: {"level": "High", "color": "#ef4444", "icon": "🔴", "desc": "Signs of kidney stress detected. Consult a nephrologist soon."},
    3: {"level": "Moderate", "color": "#3b82f6", "icon": "💧", "desc": "Dehydration indicators detected. Increase your fluid intake immediately."},
    4: {"level": "Moderate-High", "color": "#a855f7", "icon": "🦠", "desc": "Possible infection risk. Seek medical attention and boost immunity."}
}

# ─── STOOL CLASSIFIER ─────────────────────────────────────────────────────────

stool_labels = {
    0: "Normal",
    1: "Constipation",
    2: "Diarrhea",
    3: "Gastrointestinal Infection",
    4: "Malabsorption Syndrome"
}

stool_diet = {
    0: [
        "✅ Your stool health appears normal — maintain your routine!",
        "🥦 Eat a high-fiber diet: fruits, vegetables, whole grains",
        "💧 Drink 8–10 glasses of water daily for healthy gut motility",
        "🥛 Include probiotics: yogurt, kefir, fermented foods daily",
        "🌾 Consume both soluble fiber (oats) and insoluble fiber (bran)",
        "🏃 Regular exercise supports healthy bowel movements",
        "🚫 Limit processed foods, refined sugar, and alcohol",
        "😴 Maintain regular sleep patterns for gut rhythm"
    ],
    1: [
        "💧 Dramatically increase water intake — aim for 3+ liters/day",
        "🥑 Eat fiber-rich foods: prunes, figs, flaxseed, avocados",
        "🌾 Switch to whole-grain bread, brown rice, and oats",
        "🥛 Try warm milk with ghee at bedtime to soften stools",
        "🍌 Eat ripe bananas and papaya for natural laxative effect",
        "🚫 Avoid constipating foods: cheese, red meat, white bread",
        "☕ Limit caffeine and alcohol which dehydrate the bowel",
        "🏃 Walk 20–30 minutes after meals to stimulate bowel movement",
        "🥜 Soak and eat 4–5 prunes or figs overnight for relief",
        "🍵 Drink warm ginger or licorice root tea in the morning"
    ],
    2: [
        "💧 Rehydrate immediately — use ORS (oral rehydration solution)",
        "🍌 Eat the BRAT diet: Bananas, Rice, Applesauce, Toast",
        "🥣 Consume bland, easily digestible foods",
        "🚫 Avoid dairy, greasy, spicy, and high-fiber foods temporarily",
        "🧂 Replenish electrolytes: coconut water, sports drinks",
        "🥛 Take probiotics to restore gut flora (yogurt, supplements)",
        "🍵 Drink chamomile or peppermint tea to soothe the gut",
        "🚫 Avoid caffeine, alcohol, and artificial sweeteners",
        "🍚 White rice with a pinch of salt helps bind loose stools",
        "🌿 Psyllium husk (Isabgol) can help normalize stool consistency"
    ],
    3: [
        "💊 Seek medical attention immediately — may require antibiotics",
        "💧 Stay well-hydrated with ORS, broth, and coconut water",
        "🚫 Avoid solid food until vomiting/diarrhea subsides",
        "🥣 Gradually reintroduce bland foods: toast, rice, bananas",
        "🧄 Garlic has natural antimicrobial properties — add to food",
        "🍯 Manuka honey can help fight gut bacteria naturally",
        "🥛 Avoid dairy products until the infection has fully resolved",
        "😴 Rest maximally to support immune system recovery",
        "🚫 Avoid raw foods, street food, and unfiltered water",
        "🌿 Probiotic supplements after antibiotics to restore gut flora",
        "🩺 Do not self-medicate — follow prescribed treatment strictly"
    ],
    4: [
        "🩺 Consult a gastroenterologist for malabsorption diagnosis",
        "🥩 Increase easily digestible protein: eggs, fish, tofu, chicken",
        "🌾 Avoid gluten if celiac disease is suspected (wheat, rye, barley)",
        "🥛 Consider lactose-free dairy if lactose intolerance is a factor",
        "💊 Take fat-soluble vitamin supplements: A, D, E, K",
        "🦴 Monitor and supplement calcium and vitamin D for bone health",
        "🍳 Cook vegetables thoroughly to aid digestion and absorption",
        "🥜 Eat small, frequent meals to reduce digestive burden",
        "🚫 Avoid raw salads, high-fat foods, and alcohol",
        "🌿 Digestive enzyme supplements may help improve absorption",
        "🔬 Get tested for iron, B12, and folate deficiencies regularly"
    ]
}

stool_risk_info = {
    0: {"level": "Low", "color": "#22c55e", "icon": "✅", "desc": "Your stool indicators appear normal. Maintain your healthy diet and lifestyle."},
    1: {"level": "Moderate", "color": "#f59e0b", "icon": "🪨", "desc": "Signs of constipation detected. Increase fiber and fluid intake."},
    2: {"level": "Moderate", "color": "#3b82f6", "icon": "💧", "desc": "Diarrhea detected. Rehydrate immediately and follow a bland diet."},
    3: {"level": "High", "color": "#ef4444", "icon": "🦠", "desc": "Possible gastrointestinal infection. Seek medical attention promptly."},
    4: {"level": "Moderate-High", "color": "#a855f7", "icon": "🔬", "desc": "Signs of malabsorption syndrome. Consult a gastroenterologist for proper diagnosis."}
}

def stool_label(row):
    wc         = float(row.get('water_content', 50))
    freq       = float(row.get('frequency', 1))
    hardness   = float(row.get('stool_hardness', 5))
    mucus      = int(row.get('mucus', 0))
    blood      = int(row.get('blood', 0))
    pain       = int(row.get('pain', 0))
    urgency    = int(row.get('urgency', 0))
    undigested = int(row.get('undigested_food', 0))
    smell      = int(row.get('foul_smell', 0))

    # 1. Gastrointestinal Infection (highest priority)
    if blood == 1 or (mucus == 1 and pain == 1):
        return 3

    # 2. Malabsorption Syndrome
    elif undigested == 1 and smell == 1:
        return 4

    # 3. Diarrhea
    elif wc > 75 and freq >= 3:
        return 2

    # 4. Constipation
    elif hardness >= 8 and freq <= 1:
        return 1

    # 0. Normal
    else:
        return 0

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

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

@app.route("/predict-stool", methods=["POST"])
def predict_stool():
    try:
        data = request.get_json()
        prediction = stool_label(data)
        return jsonify({
            "prediction": prediction,
            "label": stool_labels[prediction],
            "diet": stool_diet[prediction],
            "risk": stool_risk_info[prediction]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True, port=5000)
