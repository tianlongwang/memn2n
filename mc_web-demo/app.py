from flask import Flask, request, jsonify
import numpy as np
import json

from server.data_utils import process_data, decode
from server.model_helper import get_pred

app = Flask(__name__, static_url_path='')

@app.after_request
def add_header(r):
    """
    Add headers to both force latest IE rendering engine or Chrome Frame,
    and also to cache the rendered page for 10 minutes.
    """
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route('/')
def root():
    return app.send_static_file('index.html')

@app.route('/answer', methods=['GET', 'POST'])
def get_answer():
    data = json.loads(request.form['data'])
    sentences = data['sentences']
    question = data['question']
    choices = data['choices']

    testS, testQ, testAS, testL = process_data(sentences, question, choices)
    print('testAS', testAS)
    print('testL', testL)

    answer, answer_probability, mem_probs = get_pred(testS, testQ, testAS)
    print('answer', answer)

    memory_probabilities = np.round(mem_probs, 4)

    answer_dict = 'ABCD'

    response = {
        "answer": answer_dict[int(answer)],
        "answerProbability": answer_probability,
        "memoryProbabilities": memory_probabilities.tolist()
    }

    print('response', response)

    return jsonify(response)

if __name__ == "__main__":
    app.run(port=5001)
