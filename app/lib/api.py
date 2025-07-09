import requests
from app.config import API_CONFIG

class API:
		def __init__(self, uid: str):
				self.uid = uid

		def save_file(self, file_name: str, file_path: str):
			try:
				with open(file_path, 'rb') as file:
						files = {'file': (file_path, file, 'application/octet-stream')}
						data = {"uid": self.uid, "name": file_name, "module": "STRESS" }								
						response = requests.post(
								f'{API_CONFIG["URL"]}/v1/analytics/file',
								files=files,
								headers=API_CONFIG["HEADERS"],
								data=data,
								timeout=60
						)
				if response.status_code != 200:
						print("Failed to save file. Status code:", response.status_code)
			except Exception as e:
				print("Error sending file:", e)
	
		def save_log(self, message: str):
			try:
				response = requests.post(
						f'{API_CONFIG["URL"]}/v1/analytics/log',
						json={"uid": self.uid, "message": message},
						headers=API_CONFIG['HEADERS'],
						timeout=60
				)
				if response.status_code != 200:
						print("Failed to save log. Status code:", response.status_code)
			except Exception as e:
				print("Error saving log:", e)
		
		def save_error(self, question_id: str, count: str, error: dict):
			try:
				response = requests.post(
						f'{API_CONFIG["URL"]}/v1/analytics/error',
						json={"uid": self.uid, "question_id": question_id, "count": str(count), "error": error},
						headers=API_CONFIG['HEADERS'],
						timeout=60
				)
				if response.status_code != 200:
						print("Failed to save error. Status code:", response.status_code)
			except Exception as e:
				print("Error saving error:", e)
						
		def save_metadata(self, question_id: str, count: str, metadata: dict):
			try:
				response = requests.post(
						f'{API_CONFIG["URL"]}/v1/analytics/metadata',
						json={"uid": self.uid, "question_id": question_id, "count": str(count), "metadata": metadata},
						headers=API_CONFIG['HEADERS'],
						timeout=60
				)
				if response.status_code != 200:
						print("Failed to save metadata. Status code:", response.status_code)
			except Exception as e:
					print("Error saving metadata:", e)

		def close_session(self):
			try:
				response = requests.post(
						f'{API_CONFIG["URL"]}/v1/analytics/close',
						json={"uid": self.uid},
						headers=API_CONFIG['HEADERS'],
						timeout=60
				)
				if response.status_code != 200:
						print("Failed to save metadata. Status code:", response.status_code)
			except Exception as e:
					print("Error saving metadata:", e)