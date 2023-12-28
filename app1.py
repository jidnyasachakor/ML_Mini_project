from flask import Flask, render_template, request
import pickle
from sklearn.feature_extraction.text import CountVectorizer


model = pickle.load(open('pra.pkl', 'rb'))

app = Flask(__name__)


vec = CountVectorizer()

@app.route('/')
def man():
    return render_template('homepage.html')

@app.route('/predict', methods=['POST'])
def predict():
    
    with open('vec.pkl', 'rb') as vectorizer_file:
        vec = pickle.load(vectorizer_file)

    data = [request.form['review']]

   
    data_vec = vec.transform(data)

    
    pred = model.predict(data_vec)

    return render_template('result1.html', review=pred[0])

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8080)
