from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        with open('nyubo_pickle', 'rb') as r:
            model = pickle.load(r)

        age = int(request.form['age'])
        sex = float(request.form['sex'])
        cp = float(request.form['cp'])
        trtbps = float(request.form['trtbps'])
        chol = float(request.form['chol'])
        fbs = float(request.form['fbs'])
        restecg = float(request.form['restecg'])
        thalachh = float(request.form['thalachh'])
        exng = float(request.form['exng'])
        oldpeak = float(request.form['oldpeak'])
        slp = float(request.form['slp'])
        caa = float(request.form['caa'])
        thall = float(request.form['thall'])

        duata = np.array((age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall))
        duata = np.reshape(duata, (1, -1))

        isBener = model.predict(duata)
        fuinal=isBener
        if fuinal == 1:
          val="Positif"
        else:
          val = "Negatif"
        return render_template('index.html', predict='Anda {}'.format(val))
    else:
        return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
