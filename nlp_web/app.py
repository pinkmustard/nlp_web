from flask import Flask, render_template, request
from pred import prediction  # pred.py에서 prediction 함수를 가져옵니다.

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    text = None
    if request.method == "POST":
        text = request.form["text"]
        result = prediction(text)  # 입력받은 텍스트를 prediction 함수에 전달
    return render_template("index.html", result=result, text=text)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=60017)
