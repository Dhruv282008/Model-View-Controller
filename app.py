from flask import Flask, jsonify, request
from classifier import getPrediction

app = Flask(__name__)

@app.route('/predict-digit', methods=['POST'])
def pred_data():
    image = request.files.get("x`")
    prediction = getPrediction(image)
    return jsonify({

        "prediction": prediction,

        }, 200)


if __name__ == "__main__":
    app.run(debug = True)
