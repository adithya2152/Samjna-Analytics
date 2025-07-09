# import requests

# ANALYTICS_API_URL = 'http://127.0.0.1:8003' 


# def test_analysis_api(files: list, data: dict):
# 		response = requests.post(ANALYTICS_API_URL, files=files, data=data)
# 		print("Response Status Code: ", response.status_code)
# 		try:
# 				print("Response JSON: ", response.json())
# 		except requests.exceptions.JSONDecodeError:
# 				print("Response Text: ", response.text)

# if __name__ == "__main__":
# 		# RECORDING FILE NAME SHOULD BE IN THIS FORMAT: '{session_id}-{video_count}-recording.webm' 
# 		session_id = "1750939090073"
# 		base_path = "/home/adithya-bharadwaj/Downloads/1750926481037/"
# 		videos = [
# 				(f"{session_id}-{1}-recording.webm", open(f"{base_path}/{session_id}-{2}-recording.webm", "rb"), "video/webm"),
# 				(f"{session_id}-{2}-recording.webm", open(f"{base_path}/{session_id}-{3}-recording.webm", "rb"), "video/webm"),
# 				(f"{session_id}-{3}-recording.webm", open(f"{base_path}/{session_id}-{5}-recording.webm", "rb"), "video/webm"),
# 				(f"{session_id}-{3}-recording.webm", open(f"{base_path}/{session_id}-{6}-recording.webm", "rb"), "video/webm"),
# 				(f"{session_id}-{3}-recording.webm", open(f"{base_path}/{session_id}-{8}-recording.webm", "rb"), "video/webm")


# 		]
# 		for index, video in enumerate(videos):
# 				test_analysis_api([("files", video)], {
# 						"session_id": session_id,
# 						"question_id": "Q1",
# 						"video_count": index + 1,
# 						"is_last_video": "true" if index == len(videos) - 1 else "false", 
# 						"save": "true"
# 				})
import requests
import re

ANALYTICS_API_URL = 'http://127.0.0.1:8003'

# Your actual question IDs
QUESTION_IDS = [
    "6aK7Ww3OwbXC", "6hjTDwxwGv8p", "9ngRzBI1DvwN",
    "ABdiktwYCxuN", "Ai7asECflMZ6", "BJ7zKM8oNxdQ",
    "CcD0OgH9TmxP", "Dwe70Vvtx9NL", "E1234567890Z", "F0987654321A"
]

def extract_video_count(filename: str) -> str:
    match = re.search(r'-(\d+)-recording\.webm$', filename)
    return match.group(1) if match else "0"

def test_analysis_api(files: list, data: list):
    try:
        response = requests.post(ANALYTICS_API_URL, files=files, data=data)
        print("Response Status Code:", response.status_code)
        try:
            print("Response JSON:", response.json())
        except requests.exceptions.JSONDecodeError:
            print("Response Text:", response.text)
    except Exception as e:
        print(f"Error occurred during API call: {e}")

if __name__ == "__main__":
    session_id = "1751964715323"
    base_path = "/home/adithya-bharadwaj/Documents/1751344506544/"

    filenames = [
        f"{session_id}-1-recording.webm",
        f"{session_id}-2-recording.webm",
        f"{session_id}-3-recording.webm",
        f"{session_id}-4-recording.webm",
        f"{session_id}-5-recording.webm",
        f"{session_id}-6-recording.webm",
        f"{session_id}-7-recording.webm",
        f"{session_id}-8-recording.webm",
        f"{session_id}-9-recording.webm",
        f"{session_id}-10-recording.webm"
    ]

    videos = [
        (fname, open(f"{base_path}/{fname}", "rb"), "video/webm")
        for fname in filenames
    ]

    for idx, video in enumerate(videos):
        filename = video[0]
        video_count = extract_video_count(filename)

        data = [
            ("session_id", session_id),
            ("video_count", video_count),
            ("question_id", QUESTION_IDS[idx]),
            ("is_last_video", "true" if idx == len(videos) - 1 else "false"),
            ("save", "true")
        ]

        test_analysis_api([("files", video)], data)
