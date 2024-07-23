from flask import Flask, request, jsonify, render_template, url_for
from chatbot import generateAnswer, initialize_rag_system
from chat_with_pandasai import initialize_pandasai_system, generatePandasAIAnswer, update_pandasai_system
import os
import glob
import markdown2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tkinter as tk
from threading import Thread

app = Flask(__name__)

initialize_rag_system()
initialize_pandasai_system()

try:
    datas = glob.glob("pandasai_data/*")
    for data in datas:
        os.remove(data)
except:
    pass

DATA_KEYWORDS = ['veri', 'csv', 'tablo', 'grafik', 'veri seti', 'pandas', "eklenti"]

@app.route('/')
def index():
    return render_template('chat.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_message = request.form['msg']
    try:
        imgs = glob.glob("static/exports/charts/*")
        for img in imgs:
            os.remove(img)
    except:
        pass
    if any(keyword in user_message.lower() for keyword in DATA_KEYWORDS) and os.path.exists("pandasai_data/kullanici_data.csv"):
        response = generatePandasAIAnswer(user_message)
        if len(os.listdir('static/exports/charts/')) != 0:
            img_path = os.path.join("static/exports/charts/", os.listdir('static/exports/charts/')[0])
            return jsonify({'image': img_path})
        else:
            return jsonify({'text': markdown2.markdown(response)})
    else:
        response = generateAnswer(user_message)
        return jsonify({'text': markdown2.markdown(response)})

@app.route('/delete', methods=["POST"])
def delete_file():
    os.remove("pandasai_data/kullanici_data.csv")
    return "File deleted successfully."

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file and file.filename.endswith('.csv'):
        filepath = "pandasai_data/kullanici_data.csv"
        file.save(filepath)
        update_pandasai_system(filepath)
        return 'File uploaded and processed successfully'
    return 'Invalid file type. Only CSV files are allowed.'

if __name__ == '__main__':
    app.run(debug=True)
