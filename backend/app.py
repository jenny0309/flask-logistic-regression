from flask import Flask, request, jsonify, abort
import pandas as pd
import json

from model import TitanicLogisticRegression


app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False


@app.route('/')
def hello_world():
    return '<h1>hello world</h1>'


@app.route('/train')
def train():

    # train model
    trainer = TitanicLogisticRegression()
    trainer.train_model()

    response = {
        "status": "OK"
    }

    return jsonify(response), 200


@app.route('/api/predict', methods=["Post"])
def predict():

    if not request.is_json:
        abort(400, {"message": "Input Content-Type is not correct"})
    
    data = request.get_json()

    data = json.dumps(data)
    person_dict = json.loads(data)
    df = pd.io.json.json_normalize(person_dict)

    # input data into the model to get predicted value
    predicter = TitanicLogisticRegression()
    result = predicter.predict(df)

    print(result)

    # result = result.to_json(orient="records")
    # result = json.loads(result)

    return jsonify(result)


@app.errorhandler(400)
def bad_request_handler(error):
    output_json = jsonify({
        "code": error.code,
        "message": error.description,
    })

    return output_json, error.code


@app.errorhandler(404)
def not_found_handler(error):
    output_json = jsonify({
        "code": error.code,
        "Message": "Request resource is not found",
    })

    return output_json, error.code


if __name__ == "__main__":
    app.run(host='0.0.0.0', port="8080", debug=True)
