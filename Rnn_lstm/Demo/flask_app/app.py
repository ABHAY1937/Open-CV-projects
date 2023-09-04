from flask import Flask , render_template,request,jsonify

app = Flask(__name__)

@app.route('/') #endpoint

def index():
    return render_template('index.html')

@app.route('/predict',methods = ['POST'])

def predict():
    input_text = request.form['input_text']
    num_words = int(request.form['num_words'])

    return jsonify({"input":input_text})

if __name__ == '__main__':
    app.run(debug=True) #,host wifi connect cheiyam