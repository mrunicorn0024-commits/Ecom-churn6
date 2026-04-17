from flask import Flask, render_template, request, jsonify
import numpy as np
import os
import pickle

app = Flask(__name__)

# =====================================================
# SAFE PATH SETUP (IMPORTANT FOR RENDER DEPLOYMENT)
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")


# =====================================================
# SAFE MODEL LOADER
# =====================================================
def load_model(filename):
    path = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


# =====================================================
# LOAD MODELS
# =====================================================
rf_model = load_model("random_forest.pkl")
svm_model = load_model("svm.pkl")
lr_model = load_model("logistic_regression.pkl")
gb_model = load_model("gradient_boosting.pkl")

scaler = load_model("scaler.pkl")
encoder = load_model("encoder.pkl")


# =====================================================
# ROUTES (FRONTEND PAGES)
# =====================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/prediction")
def prediction():
    return render_template("prediction.html")


@app.route("/search")
def search():
    return render_template("search.html")


@app.route("/comparison")
def comparison():
    return render_template("comparison.html")


@app.route("/visualization")
def visualization():
    return render_template("visualization.html")


# =====================================================
# PREDICTION API
# =====================================================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json

        model_name = data.get("model")
        features = np.array(data.get("features")).reshape(1, -1)

        # scale input
        features = scaler.transform(features)

        # model selection
        models = {
            "random_forest": rf_model,
            "svm": svm_model,
            "logistic_regression": lr_model,
            "gradient_boosting": gb_model
        }

        model = models.get(model_name)

        if model is None:
            return jsonify({"error": "Invalid model selected"}), 400

        probability = model.predict_proba(features)[0][1] * 100

        return jsonify({
            "churn_probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# SEARCH PREDICTION API
# =====================================================
@app.route("/search_predict", methods=["POST"])
def search_predict():
    try:
        data = request.json

        features = np.array(data.get("features")).reshape(1, -1)
        features = scaler.transform(features)

        probability = rf_model.predict_proba(features)[0][1] * 100

        return jsonify({
            "churn_probability": round(probability, 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =====================================================
# RUN APP (RENDER SAFE)
# =====================================================
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)