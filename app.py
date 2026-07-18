#Garvit Rajput 
#Application to detect whether the news is fake or real using machine learning model and flask framework

from flask import Flask, render_template, request
import re
import pickle

app = Flask(__name__, template_folder='./Fronted', static_folder='./static')

loaded_model = pickle.load(open("model.pkl", 'rb'))
vector = pickle.load(open("vector.pkl", 'rb'))

stpwrds = {
    'a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', 'as', 'at',
    'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', 'could',
    'did', 'do', 'does', 'doing', 'down', 'during', 'each', 'few', 'for', 'from', 'further',
    'had', 'has', 'have', 'having', 'he', 'her', 'here', 'hers', 'herself', 'him', 'himself', 'his',
    'how', 'i', 'if', 'in', 'into', 'is', 'it', 'its', 'itself', 'just', 'me', 'more', 'most', 'my',
    'myself', 'no', 'nor', 'not', 'now', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'our',
    'ours', 'ourselves', 'out', 'over', 'own', 'same', 'she', 'should', 'so', 'some', 'such', 'than',
    'that', 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', 'these', 'they',
    'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', 'we', 'were', 'what',
    'when', 'where', 'which', 'while', 'who', 'whom', 'why', 'with', 'would', 'you', 'your', 'yours',
    'yourself', 'yourselves'
}


def fake_news_det(news):
    cleaned = re.sub(r'[^a-zA-Z\s]', '', news)
    cleaned = cleaned.lower().strip()
    tokens = cleaned.split()
    filtered = [token for token in tokens if token not in stpwrds]
    input_data = [' '.join(filtered)]
    vectorized_input_data = vector.transform(input_data)
    prediction = loaded_model.predict(vectorized_input_data)
    return prediction

        

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        message = request.form.get('news', '')
        pred = fake_news_det(message)
        if pred[0] == 1:
            result = "Prediction of the News : Looking Fake News📰"
        else:
            result = "Prediction of the News : Looking Real News📰"
        return render_template('prediction.html', prediction_text=result)

    return render_template('prediction.html', prediction_text='')



if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=True)