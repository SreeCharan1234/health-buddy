from pathlib import Path
import speech_recognition as sr
import pdf2image
import PyPDF2
import gtts
import playsound
from streamlit_ace import st_ace
from PIL import Image
import base64
import streamlit as st
from streamlit_extras.let_it_rain import rain
from tempfile import NamedTemporaryFile
from streamlit_option_menu import option_menu
from streamlit_extras.mandatory_date_range import date_range_picker
import datetime
import os
import textwrap
import mediapipe as mp
import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from streamlit_lottie import st_lottie
import requests 
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import py_avataaars as pa
import sys
import io
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from youtube_transcript_api import YouTubeTranscriptApi
import time
import cv2
import pathlib
import numpy
class VideoProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")
global s
k=0
os.getenv("AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE")
genai.configure(api_key="AIzaSyBIBVb-0Z0QwaucMGOGy8-j_RM22X-4-lE")
st.set_page_config(page_title="Ai-Consultant", page_icon='chart_with_upwards_trend', layout="wide", initial_sidebar_state="auto", menu_items=None)
EXAMPLE_NO = 1
is_listening = False
recognizer = sr.Recognizer()
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Midpoint
    c = np.array(c)  # Endpoint

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle
def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
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
                
                menu_title="PhysicalHealth💪💪",  
                options=["Excersise","Report-annalyzer","Body-Movement-Tracking","3d-Study"],  
                icons=["bi bi-border-all","bi bi-card-list","bi bi-person-arms-up","bi bi-badge-3d-fill"],  
                menu_icon="cast",  
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
def start_webcam():
    # Path to the Haar cascade for face detection
    cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
    clf = cv2.CascadeClassifier(str(cascade_path))
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Could not open webcam.")
        return
    frame_placeholder = st.empty()

    while True:
        # Capture frame-by-frame from the webcam
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame.")
            break

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = clf.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the first detected face and print age
        if len(faces) > 0:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            age_text = "Age: 24"    
            cv2.putText(frame, age_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(frame_rgb, channels="RGB")
    cap.release() 
selected = streamlit_menu(example=EXAMPLE_NO)
if 'questions' not in st.session_state:
    st.session_state.questions = []
if selected=="Excersise":
    link = "https://lottie.host/d28e4979-6d47-4af3-8223-f9bed7ceade7/HC0N1Ol5Bz.json"
    lottie_animation = load_lottieurl(link)
    col1, col2 = st.columns([1.3, 9])
    with col1:
        st.lottie(lottie_animation, height=100, width=100)
    with col2:
        st.header(f":rainbow[Mind-Body Fitness]💪🧠", divider='rainbow')

    # Dropdown for exercise selection
    exercise = st.selectbox("Choose an exercise", ["Push-up", "Squat", "Lunge"])

    # Initialize session state variables
    if 'run' not in st.session_state:
        st.session_state['run'] = False

    # Start/Stop buttons for the selected exercise
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f'Start {exercise}'):
            st.session_state['run'] = True
    with col3:
        if st.button(f'Stop {exercise}'):
            st.session_state['run'] = False

    # Placeholder for video feed
    stframe = st.empty()

    # Start video capture and exercise-specific logic
    cap = cv2.VideoCapture(0)
    counter = 0
    stage = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            if not st.session_state['run']:
                break

            ret, frame = cap.read()
            if not ret:
                st.write("Failed to capture video")
                break

            # Convert frame to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(image)

            # Convert back to BGR for rendering with OpenCV
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = result.pose_landmarks.landmark

                if exercise == "Push-up":
                    # Push-up logic (shoulder-elbow-wrist)
                    shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                                landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    elbow = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    wrist = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle = calculate_angle(shoulder, elbow, wrist)
                elif exercise == "Squat":
                    # Squat logic (hip-knee-ankle)
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle = calculate_angle(hip, knee, ankle)
                elif exercise == "Lunge":
                    # Lunge logic (hip-knee-ankle, but different leg positioning)
                    hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x,
                           landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
                    knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x,
                            landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
                    ankle = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]

                    angle = calculate_angle(hip, knee, ankle)

                # Display angle
                cv2.putText(image, f"Angle: {angle:.2f}",
                            tuple(np.multiply(elbow if exercise == "Push-up" else knee,
                                              [frame.shape[1], frame.shape[0]]).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                # Repetition counting logic
                if angle > 160:
                    stage = 'up'
                if angle < 90 and stage == 'up':
                    stage = 'down'
                    counter += 1

            except:
                pass

            # Display the counter and stage
            cv2.rectangle(image, (0, 0), (250, 80), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'STAGE', (65, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage if stage else '', (60, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # Convert the frame to PIL Image and display it in Streamlit
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            stframe.image(image_pil, caption=f"{exercise} Pose Detection", use_column_width=True)

    cap.release()
if selected=="Report-annalyzer":
    def extract_text_from_pdf(file):
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    uploaded_file = st.file_uploader("Choose a file")
    
    if uploaded_file is not None:
        pdf_content=input_pdf_setup(uploaded_file)
        input_prompt1="""
            Prompt:

            Please analyze my document and provide the following:

            A summary of the main points and arguments.
            The most significant weakness or flaw in the document.
            The strengths or positive aspects of the document.
            Suggestions for improvement or further research.
            Recommendations for nearby doctors who specialize in [relevant field].
            Please note that the fifth point may require additional information about your specific needs or the topic of your document.
        
        Out put formate is :-
        1:- analyses my report
        2:- tell the worst thing in my  doucment
        3:- tell the good things in my docment
        4:- tell the some execersis 
        5:- some neary by doctures to help 
        
        """
        response=get_gemini_response1(input_prompt1,pdf_content,"Tell that in Proper")
        st.write(response)
if selected=="Body-Movement-Tracking":
    pass
if selected=="3d-Study":
    pass
