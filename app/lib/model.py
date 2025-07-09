import os
import requests
from app.config import API_CONFIG, MODEL_CONFIG

def download_models(save_directory=os.path.expanduser("/home/adithya-bharadwaj/Downloads/models")):
		model_paths = {}
		os.makedirs(save_directory, exist_ok=True)
		for model_name, model_id in MODEL_CONFIG.items():
				if not model_id:
						model_paths[model_name] = ""
						print(f"{model_name} model id not found in config, skipping download")
						continue
				existing_file = next(
						(f for f in os.listdir(save_directory) if f.startswith(f"{model_id}-")), 
						None
				)
				if existing_file:
					file_path = os.path.join(save_directory, existing_file)
					model_paths[model_name] = file_path
					print(f"{model_name} model exists, skipping download")
					continue
				print(f"Downloading {model_name} model...")
				url = f'{API_CONFIG["URL"]}/v1/models/{model_id}'
				response = requests.get(url, stream=True, headers=API_CONFIG["HEADERS"])
				if response.status_code == 200:
						if 'Content-Disposition' in response.headers:
								content_disposition = response.headers['Content-Disposition']
								filename = content_disposition.split('filename=')[-1].strip('"')
						else:
								filename = url.split('/')[-1]
						
						file_path = os.path.join(save_directory, f'{model_id}-{filename}')
						
						with open(file_path, "wb") as file:
								for chunk in response.iter_content(chunk_size=1024):
										file.write(chunk)
						
						model_paths[model_name] = file_path
						print(f"{model_name} model downloaded successfully")
				else:
						raise Exception(
								f"Failed to download {model_name}. "
								f"Status code: {response.status_code}, URL: {url}"
						)
		return model_paths
