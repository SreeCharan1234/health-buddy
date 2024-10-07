import PyPDF2
import streamlit as st
import pandas as pd
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import re
from datetime import date
import PIL.Image
from PIL import Image
from gtts import gTTS
from bs4 import BeautifulSoup
from streamlit_extras.let_it_rain import rain
import streamlit.components.v1 as components
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from pathlib import Path
from streamlit_ace import st_ace
from streamlit_option_menu import option_menu
import datetime
from googletrans import Translator
import os
import base64
import streamlit_shadcn_ui as ui
import textwrap
import google.generativeai as genai
from IPython.display import display,Markdown
from streamlit_lottie import st_lottie
import requests 
from local_components import card_container
import sys
import io
import speech_recognition as sr
import pdf2image
import plotly.graph_objects as go
import plotly.express as px
from googleapiclient.discovery import build
import pickle
from deepface import DeepFace
import cv2
global s
k=0
os.getenv("AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE")
genai.configure(api_key="AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE")
st.set_page_config(page_title="Ai-Consultant", page_icon='chart_with_upwards_trend', layout="wide", initial_sidebar_state="auto", menu_items=None)
sym_des = pd.read_csv("datasets/symtoms_df.csv")
precautions = pd.read_csv("datasets/precautions_df.csv")
workout = pd.read_csv("datasets/workout_df.csv")
description = pd.read_csv("datasets/description.csv")
medications = pd.read_csv('datasets/medications.csv')
diets = pd.read_csv("datasets/diets.csv")
svc = pickle.load(open('models/svc.pkl','rb'))
EXAMPLE_NO = 1
is_listening = False
recognizer = sr.Recognizer()
def example():
    rain(
        emoji="*",
        font_size=54,
        falling_speed=5,
        animation_length="infinite",
    )
def input_pdf_setup(uploaded_file):
        if uploaded_file is not None:
            ## Convert the PDF to image
            images=pdf2image.convert_from_bytes(uploaded_file.read())
            first_page=images[0]
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
                }
            ]
            return pdf_parts
        else:
            raise FileNotFoundError("No file uploaded")
def get_gemini_response1(input,pdf_cotent,prompt):
    model=genai.GenerativeModel('gemini-1.5-flash')
    response=model.generate_content([input,pdf_content[0],prompt])
    return response.text
def get_gemini_response(question):
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=[])
    response = model.generate_content(question)
    return response.text
def load_lottieurl(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json() 
def recognize_speech_from_microphone():
    with sr.Microphone() as source:
        while is_listening:
            st.write("Listening...")
            audio = recognizer.listen(source)
            try:
                text = recognizer.recognize_google(audio)
                
                return text
            except sr.UnknownValueError:
                st.error("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                st.error(f"Could not request results from Google Speech Recognition service; {e}")
def to_markdown(text):
  text = text.replace('•', '  *')
  return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))
def pseudo_bold(text):
    bold_text = ''.join(chr(0x1D5D4 + ord(c) - ord('A')) if 'A' <= c <= 'Z' else
                        chr(0x1D5EE + ord(c) - ord('a')) if 'a' <= c <= 'z' else c
                        for c in text)
    return bold_text
def streamlit_menu(example=1):  
    if example == 1:
        with st.sidebar:
            selected = option_menu(
                
                menu_title="Healt Dash-Bord📑" ,  # required
                options=["Dashbord","Bot Asistance","Know Your Tablet","Chat Bot","Bmibot"],  # required
                icons=["bi bi-border-all","bi bi-person-lines-fill","bi bi-capsule","bi bi-chat-dots-fill"],  # optional
                menu_icon="cast",  # optional
                default_index=0,
            )
        return selected

def input_pdf_setup(uploaded_file):
        if uploaded_file is not None:
            ## Convert the PDF to image
            images=pdf2image.convert_from_bytes(uploaded_file.read())
            first_page=images[0]
            # Convert to bytes
            img_byte_arr = io.BytesIO()
            first_page.save(img_byte_arr, format='JPEG')
            img_byte_arr = img_byte_arr.getvalue()

            pdf_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(img_byte_arr).decode()  # encode to base64
                }
            ]
            return pdf_parts
        else:
            raise FileNotFoundError("No file uploaded")
def text_to_speech(text, language):
    tts = gTTS(text=text, lang=language)
    tts.save("output.mp3")
    return "output.mp3"
def search_youtube(query, max_results=4):
    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)
    

    request = youtube.search().list(
        part="snippet",
        q=query,
        maxResults=max_results,  # Fetch the top 3-4 results
        type="video"
    )
    response = request.execute()
    
    # Extract video IDs and titles from the results
    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_title = item['snippet']['title']
        videos.append((video_id, video_title))
    
    return videos
def analyze_frame(frame):
    result = DeepFace.analyze(img_path=frame, actions=['age', 'gender', 'race', 'emotion'],
                              enforce_detection=False,
                              detector_backend="opencv",
                              align=True,
                              silent=False)
    return result
def overlay_text_on_frame(frame, texts):
    overlay = frame.copy()
    alpha = 0.9  # Adjust the transparency of the overlay
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 100), (255, 255, 255), -1)  # White rectangle
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    text_position = 15 # Where the first text is put into the overlay
    for text in texts:
        cv2.putText(frame, text, (10, text_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        text_position += 20

    return frame

    cap = cv2.VideoCapture(0)
    stframe = st.image([]) 

    while True:
       
        ret, frame = cap.read()

        
        result = analyze_frame(frame)

        
        face_coordinates = result[0]["region"]
        x, y, w, h = face_coordinates['x'], face_coordinates['y'], face_coordinates['w'], face_coordinates['h']

        # Draw bounding box around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{result[0]['dominant_emotion']}"
        cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        # Convert the BGR frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        
        texts = [
            f"Age: {result[0]['age']}",
            f"Face Confidence: {round(result[0]['face_confidence'],3)}",
            # f"Gender: {result[0]['dominant_gender']} {result[0]['gender'][result[0]['dominant_gender']]}",
            f"Gender: {result[0]['dominant_gender']} {round(result[0]['gender'][result[0]['dominant_gender']], 3)}",
            f"Race: {result[0]['dominant_race']}",
            f"Dominant Emotion: {result[0]['dominant_emotion']} {round(result[0]['emotion'][result[0]['dominant_emotion']], 1)}",
        ]

        frame_with_overlay = overlay_text_on_frame(frame_rgb, texts)

        # Display the frame in Streamlit
        stframe.image(frame_with_overlay, channels="RGB")
        

    # Release the webcam and close all windows
    cap.release()
    cv2.destroyAllWindows()

selected = streamlit_menu(example=EXAMPLE_NO)
YOUTUBE_API_KEY = 'AIzaSyCW0EXAPjHiZiyG6ebgW5FSt8g3cAIebBw'
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}
if 'questions' not in st.session_state:
    st.session_state.questions = []
if selected == "Dashbord":
    
   
    import random


    user_name = ":rainbow[Rehan]"
    age = 20
    weight =  60 # in kg
    height = 165  # in cm
    bmi = round(weight / (height / 100) ** 2, 2)

    # Example Health Metrics
    heart_rate = "104 bpm"
    steps = 8000
    blood_pressure = "120/80 mmHg"
    sleep_condition = "Good"
    overall_health = "Healthy"

    # Health trends data (Dummy Data for the graph)
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    health_data = [random.randint(4000, 12000) for _ in days]  # Random steps for each day
    a,b=st.columns([1,8])
    link="https://lottie.host/b3dde178-3bf0-40d0-a855-44f504c88252/W86eqEMu6D.json"
    l=load_lottieurl(link)
    with a:
        st.lottie(l, height=100, width=100)
        
    with b:

        st.title(f"Hello, {user_name} 👋😊")
        st.subheader("How're you feeling today?")
        st.write("It looks like you have not completed your exercise today. 🏋️‍♂️")
    
    
    st.markdown("### :rainbow[Your Live fee]:")
    
    # User Profile Section (Photo, Name, Age, Weight, Height, BMI)
    st.markdown("### :rainbow[Your Profile]:")
    a,b,c,d,f = st.columns(5)
    with a:
        ui.metric_card(title="Age", content=age,description="Great 😎",key="card")
    with b:
        ui.metric_card(title="Age", content=age,description="Great 😎",key="card2")
    with c:
        ui.metric_card(title="Weight", content=f"{weight} kg", description="Great 😎",key="card3")
    with d:
        ui.metric_card(title="Height", content=f"{height} cm",description="Great 😎", key="card4")
    with f:
        ui.metric_card(title="BMI", content=bmi, description="Great 😎",key="card5")


    # Health Metrics Section (Heart Rate, Steps, Blood Pressure, Sleep Condition, Overall Health)
    st.markdown("### :rainbow[Health Metrics]:")
     # Graph of health trends (e.g., steps over the past week)
    col1, col2, col3 = st.columns([1,1,1])
    st.markdown("### :rainbow[Health Trends]:")
    with col1:
        df=pd.DataFrame(np.random.rand(10,2),columns=["Mental Presure","Screan Time"])
        st.line_chart(df,y=["Mental Presure"])
    with col2:
        st.area_chart(df,y=["Mental Presure"])
    with col3:
        st.bar_chart(df)
    a,b,c,d,f = st.columns([1.5,1,2,1,1])
    with a:
        ui.metric_card(title="Heart Rate", content=heart_rate,description="Great 😎",key="card6")
    with b:
        ui.metric_card(title="Steps", content=steps,description="Great 😎",key="card7")
    with c:
        ui.metric_card(title="Blood Pressure", content=blood_pressure, description="Great 😎",key="card8")
    with d:
        ui.metric_card(title="Sleep Condition", content=sleep_condition,description="Great 😎", key="card9")
    with f:
        ui.metric_card(title="Overall Health", content=overall_health, description="Great 😎",key="card10")  

if selected=="Know Your Tablet":
    link="https://lottie.host/c7bac778-bec9-492f-ac66-7183002db6b5/XA1fPfJbhU.json"
    l=load_lottieurl(link)
    col1, col2 = st.columns([1.3,9])  # Create two columns
    with col1:
        st.lottie(l, height=100, width=100)
    with col2:
        st.header(f":rainbow[Know About Your Medicine]👨‍⚕️🩺", divider='rainbow')
    t= [
    "Paracetamol",
    "Ibuprofen",
    "Aspirin",
    "Amoxicillin",
    "Metformin",
    "Atorvastatin",
    "Losartan",
    "Amlodipine",
    "Simvastatin",
    "Omeprazole",
    "Lisinopril",
    "Furosemide",
    "Levothyroxine",
    "Azithromycin",
    "Clopidogrel",
    "Montelukast",
    "Cetirizine",
    "Salbutamol",
    "Metoprolol",
    "Esomeprazole",
    "Ranitidine",
    "Warfarin",
    "Albuterol",
    "Prednisone",
    "Hydrochlorothiazide",
    "Naproxen",
    "Diclofenac",
    "Insulin Glargine",
    "Sildenafil",
    "Doxycycline",
    "Tramadol",
    "Loratadine",
    "Miconazole",
    "Fluoxetine",
    "Diazepam",
    "Gabapentin",
    "Enalapril",
    "Codeine",
    "Rosuvastatin",
    "Pantoprazole",
    "Fluconazole",
    "Sertraline",
    "Venlafaxine",
    "Risperidone",
    "Olanzapine",
    "Quetiapine",
    "Ziprasidone",
    "Aripiprazole",
    "Lithium",
    "Valproic Acid",
    "Carbamazepine",
    "Lamotrigine",
    "Topiramate",
    "Levetiracetam",
    "Phenytoin",
    "Gabapentin",
    "Pregabalin",
    "Baclofen",
    "Botulinum Toxin",
    "Donepezil",
    "Rivastigmine",
    "Galantamine",
    "Memantine",
    "Levosimendan",
    "Milrinone",
    "Digoxin",
    "Amiodarone",
    "Sotalol",
    "Dronedarone",
    "Class I Antiarrhythmics",
    "Class II Antiarrhythmics",
    "Class III Antiarrhythmics",
    "Class IV Antiarrhythmics",
    "Sodium Bicarbonate",
    "Calcium Chloride",
    "Potassium Chloride",
    "Magnesium Sulfate",
    "Sodium Thiosulfate",
    "Activated Charcoal",
    "N-acetylcysteine",
    "Fomepizole",
    "Naloxone",
    "Flumazenil",
    "Physostigmine",
    "Pralidoxime",
    "Atropine",
    "Neostigmine",
    "Pyridostigmine",
    "Succinylcholine",
    "Pancuronium",
    "Vecuronium",
    "Cisatracurium",
    "Atracurium",
    "Mivacurium",
    "Rocuronium",
    "Dexmedetomidine",
    "Etomidate",
    "Ketamine",
    "Propofol",
    "Sevoflurane",
    "Desflurane",
    "Isoflurane",
    "Nitrous Oxide",
    "Remifentanil",
    "Alfentanil",
    "Fentanyl",
    "Sufentanil",
    "Morphine",
    "Hydromorphone",
    "Oxymorphone",
    "Levorphanol",
    "Meperidine",
    "Fentanyl",
    "Methadone",
    "Buprenorphine",
    "Naltrexone",
    "Clonidine",
    "Guanfacine",
    "Methyldopa",
    "Reserpine",
    "Prazosin",
    "Terazosin",
    "Doxazosin",
    "Tamsulosin",
    "Alfuzosin",
    "Phentolamine",
    "Yohimbine",
    "Phenylephrine",
    "Pseudoephedrine",
    "Oxymetazoline",
    "Xylometazoline",
    "Naphazoline",
    "Tetrahydrozoline",
    "Cromolyn Sodium",
    "Nedocromil Sodium",
    "Beclomethasone Dipropionate",
    "Fluticasone Propionate",
    "Mometasone Furoate",
    "Triamcinolone Acetonide",
    "Budesonide",
    "Flunisolide",
    "Salmeterol",
    "Formoterol",
    "Vilanterol",
    "Indacaterol",
    "Olodaterol",
    "Tiotropium Bromide",
    "Umeclidinium Bromide",
    "Glycopyrronium Bromide",
    "Acetylcysteine",
    "Ambroxol",
    "Guaifenesin",
    "Phenylephrine",
    "Pseudoephedrine",
    "Oxymetazoline",
    "Xylometazoline",
    "Naphazoline",
    "Tetrahydrozoline",
    "Diphenhydramine",
    "Chlorpheniramine",
    "Cetirizine",
    "Fexofenadine",
    "Loratadine",
    "Desloratadine",
    "Astemizole",
    "Terfenadine",
    "Promethazine",
    "Meclizine",
    "Dimenhydrinate",
    "Scopolamine",
    "Ondansetron",
    "Granisetron",
    "Palonosetron",
    "Metoclopramide",
    "Domperidone",
    "Cisapride",
    "Prucalopride",
    "Linaclotide",
    "Lubiprostone",
    "Rifaximin",
    "Metronidazole",
    "Tinidazole",
    "Secnidazole",
    "Ornidazole",
    "Nifurtimox",
    "Benznidazole",
    "Amphotericin B",
    "Fluconazole",
    "Voriconazole",
    "Posaconazole",
    "Itraconazole",
    "Ketoconazole",
    "Terbinafine",
    "Griseofulvin",
    "Flucytosine",
    "5-Fluorouracil",
    "Methotrexate",
    "Cyclophosphamide",
    "Doxorubicin",
    "Bleomycin",
    "Vincristine",
    "Vinblastine",
    "Paclitaxel",
    "Docetaxel",
    "Etoposide",
    "Ifosfamide",
    "Temozolomide",
    "Carboplatin",
    "Cisplatin",
    "Oxaliplatin",
    "L-Asparaginase",
    "Rituximab",
    "Trastuzumab",
    "Bevacizumab",
    "Cetuximab",
    "Panitumumab",
    "Ramucirumab",
    "Afatinib",
    "Erlotinib",
    "Gefitinib",
    "Imatinib",
    "Dasatinib",
    "Nilotinib",
    "Sorafenib",
    "Sunitinib",
    "Vemurafenib",
    "Dabrafenib",
    "Trametinib",
    "Cobimetinib",
    "Encorafenib",
    "Binimetinib",
    "Nivolumab",
    "Pembrolizumab",
    "Atezolizumab",
    "Durvalumab",
    "Avelumab",
    "Ipilimumab",
    "Nivolumab",
    "Pembrolizumab",
    "Atezolizumab",
    "Durvalumab",
    "Avelumab",
    "Ipilimumab",
    "Tremelimumab",
    "Talimogene Laherparepvec",
    "Tecartus",
    "Yescarta",
    "Kymriah",
    "Blincyto",
    "Idecabtagene Viclustel",
    "Brexucabtagene Autoleucel",
    "Lisocabtagene Maraleucel",
    "Tisagenlecleucel"
]
    col1,col2,col3=st.columns([2,1,1])
    with col1:
        selected_options = st.multiselect('Select one or more options:', t)
    with col2:
        selected_options2 = st.multiselect('Select one or more options:', ["English","Hindi","Telgue","Tamil"])
    with col3:
        s=st.button("Button clicked!")
    
    promt="""        Please provide a comprehensive overview of """+str(selected_options )+"""Include information on the following:
        
        Purpose: What conditions is it used to treat?
        Mechanism of action: How does it work in the body?
        Dosage: Typical dosage and frequency of administration.
        Side effects: Common and rare side effects.
        Interactions: Does it interact with other medications or substances?
        Precautions: Are there any specific precautions or warnings associated with its use?
        Contraindications: When should it not be used?
        Availability: Is it available over-the-counter or by prescription?
        Generic name: What is the generic name for this medication?
        Brand names: What are some common brand names?
        Please also provide any additional information you think might be relevant or helpful."

        Replace """+str(selected_options)+""" with the specific medication you're interested in.

        """
    if s:
        if selected_options and selected_options2:  
        
            k=get_gemini_response(promt)
            translator = Translator()
            text = k
            if str(selected_options2)==str(['Telgue']):              
                translated_text = translator.translate(text, dest="te")
                st.write(translated_text.text)
                output_file = text_to_speech(translated_text.text[:len(translated_text.text)//2], "te")
            
                audio_file = open(output_file, 'rb')
                audio_bytes = audio_file.read()

                st.audio(audio_bytes, format='audio/mp3')
            elif str(selected_options2)==str(['Hindi']):
                translated_text = translator.translate(text, dest="hi")
                st.write(translated_text.text)
                output_file = text_to_speech(translated_text.text[:len(translated_text.text)//2], "hi")
            
                audio_file = open(output_file, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')

            else:
                st.write(k)
                output_file = text_to_speech(k[:len(k)//2], "en")
            
                audio_file = open(output_file, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/mp3')
            try:
                
                videos = search_youtube(str("tell me about")+str(selected_options), max_results=4)
            
                cols = st.columns(len(videos))  # Create as many columns as there are videos
            
                # Display each video in a column
                for idx, (video_id, video_title) in enumerate(videos):
                    with cols[idx]:
                        st.write(f"**{video_title}**")
                        st.video(f"https://www.youtube.com/watch?v={video_id}")
            except Exception as e:
                st.error(f"Error fetching videos: {e}")
if selected=="Chat Bot":
    example()
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "Hi 😊 Tell Me Your Name?"}]
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        msg=get_gemini_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)        
if selected=="Bot Asistance":
    def get_predicted_value(patient_symptoms):
        input_vector = np.zeros(len(symptoms_dict))
        for item in patient_symptoms:
            input_vector[symptoms_dict[item]] = 1
        return diseases_list[svc.predict([input_vector])[0]]  
if selected=="Bmibot":
    import requests


    st.title("Chat API Integration")

    api_key = "C65pUxaPOwq0leUdFdKSy3c8dvuqJWMK"
    external_user_id = "rehanmittal"
    query = st.text_input("Query")

    if st.button("Submit Query"):
        # Create Chat Session
        session_url = "https://api.on-demand.io/chat/v1/sessions"
        session_headers = {"apikey": api_key}
        session_body = {"pluginIds": ["plugin-1726251310"], "externalUserId": external_user_id}

        session_response = requests.post(session_url, headers=session_headers, json=session_body)
        session_data = session_response.json()
        session_id = session_data['data']['id']

        # Submit Query
        query_url = f"https://api.on-demand.io/chat/v1/sessions/{session_id}/query"
        query_headers = {"apikey": api_key}
        query_body = {
            "endpointId": "predefined-openai-gpt4o",
            "query": query,
            "pluginIds": ["plugin-1712327325", "plugin-1713962163", "plugin-1726251310"],
            "responseMode": "sync"
        }

        query_response = requests.post(query_url, headers=query_headers, json=query_body)
        query_result = query_response.json()

        # Display the result
        st.write(query_result["data"]["answer"])
