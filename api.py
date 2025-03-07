from flask import Flask, request
import ktp_ocr
import json

app = Flask(__name__)

@app.route("/" , methods=['POST'])
def hello():
    filename = request.form.get('filename')
    print(filename)
    json_data = ktp_ocr.startOCR(filename)
    return json_data

if __name__ == "__main__":
    app.run(host='0.0.0.0')