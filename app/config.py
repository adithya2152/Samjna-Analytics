from dotenv import load_dotenv
import os

load_dotenv(override=True)


API_CONFIG = {
	"DATA_FOLDER": "/tmp/interview-analytics",
	"URL": f"{os.getenv('API_URL') or ''}/core-service",
	"HEADERS": {
    'x-api-key': os.getenv("API_KEY") or ""
	},
	"GEMINI_API_KEY": os.getenv("GEMINI_API_KEY"),
}

MODEL_CONFIG = {
	"FACS_MODEL": os.getenv("FACS_MODEL"),
	"FACE_DETECTION_PROTOTXT": os.getenv("FACE_DETECTION_PROTOTXT"),
	"FACE_DETECTION_CAFFE_MODEL": os.getenv("FACE_DETECTION_CAFFE_MODEL"),
	"FACE_LANDMARKS_MODEL": os.getenv("FACE_LANDMARKS_MODEL"),
	"FER_MODEL": os.getenv("FER_MODEL"),
	"VALENCE_AROUSAL_FEATURES": os.getenv("VALENCE_AROUSAL_FEATURES"),
	"VALENCE_AROUSAL_MODEL": os.getenv("VALENCE_AROUSAL_MODEL"),
}

