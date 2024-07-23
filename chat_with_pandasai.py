import pandas as pd
import os
from pandasai import Agent
from pandasai.llm.google_gemini import GoogleGemini
from pandasai import SmartDataframe
from dotenv import load_dotenv
load_dotenv()

csv_file_path = "pandasai_data/kullanici_data.csv"

google_api_key = os.environ["GOOGLE_API_KEY"]

prompt = """
Sorulara Akbank' ın dijital asistanı olarak cevapları, saygı çerçevesinde ve resmi bir dille, Türkçe ve 'string' olacak şekilde ver.
Görselleştirme yaptığında sadece istenilen görseli bir kere oluştur.
"""

llm = None
sdf = None

def initialize_pandasai_system():
    global llm, sdf
    llm = GoogleGemini(api_key=google_api_key)
    
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
    else:
        df = pd.DataFrame() 

    sdf = SmartDataframe(df=df, config={"llm": llm, 'open_charts': False, "save_charts": True, "save_charts_path": "static/exports/charts/"})

def update_pandasai_system(filepath):
    global sdf
    df = pd.read_csv(filepath)
    sdf = SmartDataframe(df=df, config={"llm": llm, 'open_charts': False, "save_charts": True, "save_charts_path": "static/exports/charts/"})

def generatePandasAIAnswer(query):
    query = prompt + query
    response = sdf.chat(query)
    return response