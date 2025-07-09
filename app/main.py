# from fastapi import FastAPI, File, UploadFile, WebSocket, BackgroundTasks, Form, WebSocketDisconnect
# from fastapi.responses import JSONResponse
# import uvicorn
# import concurrent.futures
# from app.lib.model import download_models
# from app.analytics.init import load_models
# from app.lib.input import load_inputs
# from app.lib.socket import ConnectionManager
# from app.lib.api import API
# from app.analytics.controller import analyze_interview
# from contextlib import asynccontextmanager
# from typing import List
# from app.config import API_CONFIG
# from datetime import datetime
# import pytz
# from app.lib.validation import 	validate_all_videos
# import logging

# executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
# models = {}

# @asynccontextmanager
# async def lifespan(_: FastAPI):
# 		global models
# 		model_paths = download_models()
# 		print(f"✅ Downloaded {model_paths.__len__()} models")
# 		models = load_models(model_paths)
# 		print(f"✅ Loaded {models.__len__()} models")
# 		yield
# 		models.clear()
		
# app = FastAPI(lifespan=lifespan)
# connection_manager = ConnectionManager()

# @app.get('/health')
# async def health_controller():
# 		print("Health check")
# 		return {"ok": "true" }

# @app.websocket(f'/ws/{{uid}}')
# async def websocket_endpoint(websocket: WebSocket, uid: str):
# 		await connection_manager.connect(websocket, uid)
# 		print(f"Connected to {uid}")
# 		try:
# 				while True:
# 						await websocket.receive_text()
# 		except WebSocketDisconnect:
# 				connection_manager.disconnect(uid)

# @app.post('/')
# async def analysis_controller(
# 		files: List[UploadFile] = File(...),
# 		session_id: str = Form(...), question_id: str = Form(...), video_count: str = Form(...), is_last_video: str = Form(...), 
# 		baseline_questions: list[str] = Form([]), save: str = Form("true")
# ):
# 		if models.__len__() == 0:
# 				return JSONResponse(status_code=400, content={"message": "Models not loaded"})
		

# 		video_count = int(video_count)

# 		expected_count = 10 if is_last_video.lower() == "true" else int(video_count)
# 		valid, error_message = validate_all_videos(files, session_id, expected_count=expected_count)
# 		if not valid:
# 			logging.error(f"[ValidationError] {session_id}: {error_message}")
# 			return JSONResponse(status_code=400, content={"message": error_message})


# 		inputs = load_inputs(files, session_id, video_count)
# 		is_last_video = is_last_video.lower() == "true"
# 		save = save.lower() == "true"
# 		api = API(session_id)
										
# 		def log(message: str):
# 				ist = pytz.timezone('Asia/Kolkata')
# 				now = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
# 				print(f"[{now}] [Session: {session_id}] {message}")
# 				if save:
# 					connection_manager.send_log_background(session_id, message)
# 					api.save_log(message)

# 		log(f"Received {files.__len__()} files for session - {video_count}")

# 		executor.submit(analyze_interview, models, inputs, session_id, question_id, baseline_questions, video_count, is_last_video, save, log)
		

# 		return JSONResponse(status_code=200, content={"message": "Sent for analysis", "data": {"session_id": session_id}})

from fastapi import FastAPI, File, UploadFile, WebSocket, BackgroundTasks, Form, WebSocketDisconnect
from fastapi.responses import JSONResponse
import uvicorn
import concurrent.futures
from app.lib.model import download_models
from app.analytics.init import load_models
from app.lib.input import load_inputs
from app.lib.socket import ConnectionManager
from app.lib.api import API
from app.analytics.controller import analyze_interview
from contextlib import asynccontextmanager
from typing import List
from app.config import API_CONFIG
from datetime import datetime
import pytz
from app.lib.validation import validate_all_videos
import logging

executor = concurrent.futures.ThreadPoolExecutor(max_workers=10)
models = {}

@asynccontextmanager
async def lifespan(_: FastAPI):
    global models
    model_paths = download_models()
    print(f"\u2705 Downloaded {len(model_paths)} models")
    models = load_models(model_paths)
    print(f"\u2705 Loaded {len(models)} models")
    yield
    models.clear()

app = FastAPI(lifespan=lifespan)
connection_manager = ConnectionManager()

@app.get('/health')
async def health_controller():
    print("Health check")
    return {"ok": "true"}

@app.websocket(f'/ws/{{uid}}')
async def websocket_endpoint(websocket: WebSocket, uid: str):
    await connection_manager.connect(websocket, uid)
    print(f"Connected to {uid}")
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(uid)

@app.post('/')
async def analysis_controller(
    files: List[UploadFile] = File(...),
    session_id: str = Form(...),
    question_id: str = Form(...),
    video_count: str = Form(...),
    is_last_video: str = Form(...),
    baseline_questions: List[str] = Form([]),
    save: str = Form("true")
):
    if not models:
        return JSONResponse(status_code=400, content={"message": "Models not loaded"})

    # try:
    #     valid, error_message = validate_all_videos(files, session_id, expected_count=10)
    #     if not valid:
    #         logging.error(f"[ValidationError] {session_id}: {error_message}")
    #         return JSONResponse(status_code=400, content={"message": error_message})
    # except Exception as e:
    #     logging.exception(f"[CriticalValidationFailure] {session_id}: {str(e)}")
    #     return JSONResponse(status_code=500, content={"message": "Critical error during video validation."})
    

    valid_files, failed_files = validate_all_videos(files, session_id, expected_count=10)

    if failed_files:
        for fname, reason in failed_files:
            logging.error(f"[ValidationFailed] {session_id} | {fname} - {reason}")
        if not valid_files:
            return JSONResponse(status_code=400, content={"message": "All videos invalid", "errors": failed_files})

    # Continue only with valid files
    def extract_video_index(name: str) -> int:
        try:
            return int(name.split("-")[1])
        except:
            return 0
    valid_files.sort(key=lambda f: extract_video_index(f.filename))

    # Update video count to match actual valid ones
    # video_count = len(valid_files)
    video_count = int(video_count)
    # Load inputs for only valid files
    inputs = load_inputs(valid_files, session_id, video_count)
    is_last_video = is_last_video.lower() == "true"
    save = save.lower() == "true"
    api = API(session_id)

    def log(message: str):
        ist = pytz.timezone('Asia/Kolkata')
        now = datetime.now(ist).strftime('%Y-%m-%d %H:%M:%S')
        full_msg = f"[{now}] [Session: {session_id}] {message}"
        print(f"[{now}] [Session: {session_id}] {message}")
        print(full_msg)
        if save:
            try:
                connection_manager.send_log_background(session_id, message)
                response = api.save_log(message)
                if response is not None and hasattr(response, "status_code") and response.status_code != 200:
                    logging.error(f"[APIError] {session_id}: Failed to save log - {getattr(response, 'text', '')}")
            except Exception as e:
                logging.warning(f"[LoggingError] {session_id}: {str(e)}")
                
		

    log(f"Received {len(files)} files for session - {video_count}")

    executor.submit(analyze_interview, models, inputs, session_id, question_id, baseline_questions, video_count, is_last_video, save, log)

    return JSONResponse(status_code=200, content={"message": "Sent for analysis", "data": {"session_id": session_id}})