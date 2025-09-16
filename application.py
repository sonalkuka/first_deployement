import pickle
from flask import Flask, request, render_template

application = Flask(__name__)
app = application

# load model
model = pickle.load(open('regressor', 'rb'))

@app.route("/predict", methods=["GET", "POST"])
def predict():
    results = None
    if request.method == "POST":
        weight = float(request.form.get('Weight'))
        results = model.predict([[weight]])[0]
    return render_template('predict.html', results=results)

if __name__ == "__main__":
    app.run(debug=True)
