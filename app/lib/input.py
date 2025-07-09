from fastapi import UploadFile
from pathlib import Path
from typing import List
import os
import shutil
from app.config import API_CONFIG


def load_inputs(files: List[UploadFile], session_id: str, video_count: str):
		inputs = {}

		INPUT_FOLDER = Path(f'{API_CONFIG["DATA_FOLDER"]}/{session_id}/inputs/{video_count}')
		INPUT_FOLDER.mkdir(parents=True, exist_ok=True)

		for file in files:
				file_path = os.path.join(INPUT_FOLDER, file.filename)
				with open(file_path, "wb") as buffer:
						shutil.copyfileobj(file.file, buffer)
				inputs[file.filename] = file_path
				
		return inputs