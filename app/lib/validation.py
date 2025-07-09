# # import os
# # import logging
# # import subprocess
# # from moviepy.editor import VideoFileClip
# # from pydub import AudioSegment
# # from fastapi import UploadFile
# # from tempfile import NamedTemporaryFile
# # from typing import List, Tuple

# # # Logger setup
# # logging.basicConfig(filename='input_validation.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# # ALLOWED_EXTS = [".webm", ".mp4", ".mov"]
# # MAX_FILE_SIZE_MB = 100
# # MIN_FPS = 15
# # MAX_FPS = 60
# # MIN_SAMPLE_RATE = 16000  # Hz
# # MIN_AUDIO_BITRATE = 64000  # bps

# # def validate_file(file: UploadFile) -> bool:
# #     try:
# #         filename = file.filename or ""
# #         suffix = os.path.splitext(filename)[1].lower()
# #         if suffix not in ALLOWED_EXTS:
# #             raise ValueError(f"Invalid file format: {suffix}")

# #         content = file.file.read()
# #         if not content:
# #             raise ValueError("Uploaded file is empty or corrupted")

# #         with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
# #             size_mb = len(content) / (1024 * 1024)
# #             if size_mb <= 0 or size_mb > MAX_FILE_SIZE_MB:
# #                 raise ValueError("File size must be >0 and <=100MB")
# #             temp.write(content)
# #             temp_path = temp.name

# #         try:
# #             clip = VideoFileClip(temp_path)
# #             if clip.fps < MIN_FPS or clip.fps > MAX_FPS:
# #                 raise ValueError(f"Frame rate out of bounds: {clip.fps}")
# #         except Exception as e:
# #             raise ValueError(f"Video load error: {e}")

# #         audio_temp = temp_path.replace(suffix, ".wav")
# #         command = ['ffmpeg', '-i', temp_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_temp, '-y']
# #         subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# #         audio = AudioSegment.from_wav(audio_temp)
# #         if audio.frame_rate < MIN_SAMPLE_RATE:
# #             raise ValueError("Sample rate too low")
# #         if audio.frame_width * 8 * audio.frame_rate < MIN_AUDIO_BITRATE:
# #             raise ValueError("Bitrate too low")

# #         os.remove(temp_path)
# #         os.remove(audio_temp)
# #         return True
# #     except Exception as e:
# #         logging.error(f"Validation failed for {file.filename}: {e}")
# #         return False

# # def validate_all_videos(files: List[UploadFile], session_id: str, expected_count: int = 10) -> Tuple[bool, str]:
# #     seen = set()
# #     for file in files:
# #         name = file.filename or ""
# #         if not name.startswith(f"{session_id}-"):
# #             logging.error(f"Filename does not start with session_id: {name}")
# #             return False, f"Filename '{name}' is not labeled properly."

# #         try:
# #             idx = int(name.split("-")[1])
# #             if idx < 1 or idx > expected_count:
# #                 raise ValueError
# #             if idx in seen:
# #                 return False, f"Duplicate video index: {idx}"
# #             seen.add(idx)
# #         except Exception:
# #             return False, f"Filename '{name}' does not follow expected pattern '{{session_id}}-{{index}}-recording.webm'"

# #         if not validate_file(file):
# #             return False, f"Video {idx} is invalid. Please reupload."

# #     return True, ""

# import os
# import logging
# from moviepy.editor import VideoFileClip
# from fastapi import UploadFile
# from tempfile import NamedTemporaryFile
# from typing import List, Tuple

# # Logger setup
# logging.basicConfig(filename='input_validation.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

# ALLOWED_EXTS = [".webm", ".mp4", ".mov"]
# MAX_FILE_SIZE_MB = 100

# def check_format_and_size(file: UploadFile) -> Tuple[bool, str]:
#     """Check format and file size only."""
#     try:
#         filename = file.filename or ""
#         suffix = os.path.splitext(filename)[1].lower()

#         if suffix not in ALLOWED_EXTS:
#             return False, f"Invalid file format: {suffix}"

#         content = file.file.read()
#         if not content:
#             return False, "File is empty or corrupted."

#         size_mb = len(content) / (1024 * 1024)
#         if size_mb <= 0 or size_mb > MAX_FILE_SIZE_MB:
#             return False, f"File size {size_mb:.2f}MB is out of bounds."

#         # Write temporarily to validate video load
#         with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
#             temp.write(content)
#             temp_path = temp.name

#         try:
#             VideoFileClip(temp_path)  # Just test loading
#         except Exception as e:
#             return False, f"Video load failed: {e}"

#         os.remove(temp_path)
#         return True, ""
#     except Exception as e:
#         logging.error(f"[ValidationError] {file.filename}: {e}")
#         return False, f"Critical error: {e}"

# def validate_filename(file: UploadFile, session_id: str, seen_indices: set, expected_count: int) -> Tuple[bool, str, int]:
#     """Validate filename pattern."""
#     try:
#         name = file.filename or ""
#         if not name.startswith(f"{session_id}-"):
#             return False, f"Filename '{name}' doesn't match session_id", -1

#         parts = name.split("-")
#         if len(parts) < 3:
#             return False, f"Filename '{name}' must be in format {{session_id}}-{{index}}-recording.webm", -1

#         idx = int(parts[1])
#         if idx < 1 or idx > expected_count:
#             return False, f"Invalid video index {idx}", -1
#         if idx in seen_indices:
#             return False, f"Duplicate video index {idx}", -1
#         seen_indices.add(idx)
#         return True, "", idx
#     except Exception:
#         return False, "Invalid filename or index", -1

# def validate_all_videos(files: List[UploadFile], session_id: str, expected_count: int = 10) -> Tuple[List[UploadFile], List[Tuple[str, str]]]:
#     """Validate all videos and return valid + failed files."""
#     seen = set()
#     valid_files = []
#     failed = []

#     for file in files:
#         name = file.filename or ""
        
#         # Validate filename
#         name_ok, msg, idx = validate_filename(file, session_id, seen, expected_count)
#         if not name_ok:
#             failed.append((name, msg))
#             continue

#         # Check format & size
#         ok, msg = check_format_and_size(file)
#         if not ok:
#             failed.append((name, msg))
#             continue

#         valid_files.append(file)

#     return valid_files, failed


import os
import logging
from moviepy.editor import VideoFileClip
from fastapi import UploadFile
from tempfile import NamedTemporaryFile
from typing import List, Tuple

# Logger setup
logging.basicConfig(
    filename='input_validation.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

ALLOWED_EXTS = [".webm", ".mp4", ".mov"]
MAX_FILE_SIZE_MB = 100

def check_format_and_size(file: UploadFile) -> Tuple[bool, str]:
    """Check format and file size only."""
    try:
        filename = file.filename or ""
        suffix = os.path.splitext(filename)[1].lower()

        if suffix not in ALLOWED_EXTS:
            return False, f"Invalid file format: {suffix}"
        
        content = file.file.read()
        if not content:
            return False, "File is empty or corrupted."

        size_mb = len(content) / (1024 * 1024)
        if size_mb <= 0 or size_mb > MAX_FILE_SIZE_MB:
            return False, f"File size {size_mb:.2f}MB is out of bounds."

        # Write temporarily to validate video load
        with NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(content)
            temp_path = temp.name

        try:
            clip = VideoFileClip(temp_path)
            duration = clip.duration  # Trigger lazy load
        except Exception as e:
            return False, f"Video load failed: {e}"

        os.remove(temp_path)
        logging.info(f"[ValidationPassed] {filename} ✅ Format: {suffix}, Size: {size_mb:.2f}MB, Duration: {duration:.2f}s")
        return True, ""
    except Exception as e:
        logging.error(f"[ValidationError] {file.filename}: {e}")
        return False, f"Critical error: {e}"

def validate_filename(file: UploadFile, session_id: str, seen_indices: set, expected_count: int) -> Tuple[bool, str, int]:
    """Validate filename pattern."""
    try:
        name = file.filename or ""
        if not name.startswith(f"{session_id}-"):
            return False, f"Filename '{name}' doesn't match session_id", -1

        parts = name.split("-")
        if len(parts) < 3:
            return False, f"Filename '{name}' must be in format {{session_id}}-{{index}}-recording.webm", -1

        idx = int(parts[1])
        if idx < 1 or idx > expected_count:
            return False, f"Invalid video index {idx}", -1
        if idx in seen_indices:
            return False, f"Duplicate video index {idx}", -1
        seen_indices.add(idx)

        logging.info(f"[FilenameValid] {name} ✅ index: {idx}")
        return True, "", idx
    except Exception as e:
        return False, f"Invalid filename or index: {e}", -1

def validate_all_videos(files: List[UploadFile], session_id: str, expected_count: int = 10) -> Tuple[List[UploadFile], List[Tuple[str, str]]]:
    """Validate all videos and return valid + failed files."""
    seen = set()
    valid_files = []
    failed = []

    for file in files:
        name = file.filename or ""
        
        # Validate filename
        name_ok, msg, idx = validate_filename(file, session_id, seen, expected_count)
        if not name_ok:
            failed.append((name, msg))
            logging.warning(f"[FilenameInvalid] {name} ❌ {msg}")
            continue

        # Check format & size
        ok, msg = check_format_and_size(file)
        if not ok:
            failed.append((name, msg))
            logging.warning(f"[FileCheckFailed] {name} ❌ {msg}")
            continue

        valid_files.append(file)
        logging.info(f"[VideoValid] {name} ✅ Successfully validated")

    print(f"\n📤 Returning from validate_all_videos:\n  ✅ valid_files: {[f.filename for f in valid_files]}\n  ❌ failed_files: {failed}")
    return valid_files, failed
