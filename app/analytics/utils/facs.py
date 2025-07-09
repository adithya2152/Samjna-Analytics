import numpy as np
import cv2
import tensorflow as tf
import keras

au_columns = [  #action units with  their respective labels
	"au1 - Inner Brow Raiser",
	"au2 - Outer Brow Raiser",
	"au4 - Brow Lowerer",
	"au5 - Upper Lid Raiser",
	"au6 - Cheek Raiser",
	"au9 - Nose Wrinkler",
	"au12 - Lip Corner Puller",
	"au15 - Lip Corner Depressor",
	"au17 - Chin Raiser",
	"au20 - Lip Stretcher",
	"au25 - Lips Part",
	"au26 - Jaw Drop"
]
def binary_focal_loss(y_true, y_pred):
	gamma = 2.0
	alpha = 0.25
	epsilon = tf.keras.backend.epsilon()
	y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
	y_true = tf.cast(y_true, tf.float32)
	alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
	p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
	fl = -alpha_t * tf.pow(1 - p_t, gamma) * tf.math.log(p_t)
	return tf.reduce_sum(fl, axis=-1)

def load_custom_model(model_path):

	model = keras.models.load_model(model_path,
                                       custom_objects={'binary_focal_loss': binary_focal_loss})
	return model


def predict_au_from_video(model, faces):
	threshold = [0.4, 0.65, 0.55, 0.5, 0.63, 0.58, 0.54, 0.60, 0.56, 0.51, 0.62, 0.5]
	processed_faces = []
	valid_indices = []

	# facs_faces=[cv2.resize(face, (48, 48)) for face in faces if face is not None]

	for i, face in enumerate(faces):
		if face is not None:
			processed_faces.append(face)
			valid_indices.append(i)

	if not processed_faces:
		return np.zeros((len(faces), 12), dtype=int)

	processed_faces = np.array(processed_faces)
	preds = model.predict(processed_faces, batch_size=32)
	binary_preds = (preds >= threshold).astype(int)

	results = np.zeros((len(faces), 12), dtype=int)
	for idx, pred in zip(valid_indices, binary_preds):
		results[idx] = pred

	return results

def calculate_au_percentages(predictions):
	"""
	predictions: 2D numpy array of shape (num_frames, 12)
	returns: list of AU occurrence percentages
	"""
	if len(predictions) == 0:
		return [0.0] * 12
	return (np.mean(predictions, axis=0) * 100).tolist()