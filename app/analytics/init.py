import nltk
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import os 
from app.analytics.utils.facial_emotion_recognition import Model
import cv2
import dlib
from app.analytics.utils.valence_arousal_detection import load_valence_arousal_models
from app.analytics.utils.facs import load_custom_model
from app.analytics.utils.speech_util import AudioAnalysis
import google.generativeai as genai
from app.config import MODEL_CONFIG, API_CONFIG


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

def load_models(model_paths: dict):
		if not all(model_paths.values()):
				return {}
		genai.configure(api_key=API_CONFIG["GEMINI_API_KEY"])
		gem_model = genai.GenerativeModel(model_name="gemini-1.5-flash")
		device = "cuda:0" if torch.cuda.is_available() else "cpu"
		torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
		
		# Load Whisper model and processor
		model_id = "openai/whisper-small"
		model = AutoModelForSpeechSeq2Seq.from_pretrained(
				model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True 
		)
		model.to(device)
		processor = AutoProcessor.from_pretrained(model_id)
		sentipipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment", device=device)


		#Face Detection
		dnn_net = cv2.dnn.readNetFromCaffe(model_paths['FACE_DETECTION_PROTOTXT'], model_paths['FACE_DETECTION_CAFFE_MODEL'])
		predictor = dlib.shape_predictor(model_paths['FACE_LANDMARKS_MODEL'])


		#FER Model
		fer_model=Model(fps=30,fer_model=model_paths['FER_MODEL'])
		
		print(f"Loading model: {model_paths['FACS_MODEL']}")
		#facs model 
		facs_model=load_custom_model(model_paths['FACS_MODEL'])

		#Valence Arousal Model
		resnet,emotion_model=load_valence_arousal_models(model_paths['VALENCE_AROUSAL_MODEL'], model_paths['VALENCE_AROUSAL_FEATURES'])

		#Haar Cascade Models
		smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

		audio_analysis = AudioAnalysis()



		return {
			'asrmodel':model,
			'asrproc':processor,
			'sentipipe':sentipipe,
			'fer':fer_model,
      'facs':facs_model,
			"valence_fer":(resnet,emotion_model),
			'smile_cascade':smile_cascade,
			'face':(dnn_net,predictor),
			'gem':gem_model,
			"audio_analysis":audio_analysis,
		}   
