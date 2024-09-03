from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import os
import shutil
import uuid
import subprocess
from typing import Dict
from pydub import AudioSegment
from loguru import logger  # Ensure you have loguru installed for logging
import importlib
import tqdm
import copy
import cv2
import math
import base64

# common utils
from utils.commons.hparams import hparams, set_hparams
from utils.commons.tensor_utils import move_to_cuda, convert_to_tensor
from utils.commons.ckpt_utils import load_ckpt, get_last_checkpoint
# 3DMM-related utils
from deep_3drecon.deep_3drecon_models.bfm import ParametricFaceModel
from data_util.face3d_helper import Face3DHelper
from data_gen.utils.process_image.fit_3dmm_landmark import fit_3dmm_for_a_image
from data_gen.utils.process_video.fit_3dmm_landmark import fit_3dmm_for_a_video
from deep_3drecon.secc_renderer import SECC_Renderer
from data_gen.eg3d.convert_to_eg3d_convention import get_eg3d_convention_camera_pose_intrinsic
from data_gen.utils.process_image.extract_lm2d import extract_lms_mediapipe_job

# Face Parsing 
from data_gen.utils.mp_feature_extractors.mp_segmenter import MediapipeSegmenter
from data_gen.utils.process_video.extract_segment_imgs import inpaint_torso_job, extract_background
# other inference utils
from inference.infer_utils import mirror_index, load_img_to_512_hwc_array, load_img_to_normalized_512_bchw_tensor
from inference.infer_utils import smooth_camera_sequence, smooth_features_xd
from inference.edit_secc import blink_eye_for_secc

from inference.real3d_infer import read_first_frame_from_a_video, analyze_weights_img,crop_img_on_face_area_percent,GeneFace2Infer
from Authentication import auth
from upload import upload_to_imgkit
import concurrent.futures
import requests
import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Body, HTTPException
from PIL import Image

import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()
# Set up logger

from fastapi import FastAPI, UploadFile, File, APIRouter,Body
import shutil
import json
from pathlib import Path
import concurrent.futures
#from Authentication.authentication  import Authenticator, FaceSimilarity, ocr_space
from pydantic import BaseModel
import time
import ast
import subprocess
import os
import random
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
app = FastAPI()
UPLOAD_DIR = "uploads"
RESULT_DIR = "result"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)

# Dictionary to store file statuses
file_statuses: Dict[str, str] = {}
file_paths: Dict[str, str] = {}
output_files: Dict[str, str] = {}

class StatusResponse(BaseModel):
    id: str
    status: str

class WebhookPayload(BaseModel):
    url: str
    webhook_id: str

def save_file(file, filename):
    file_location = Path(UPLOAD_DIR)
    with (file_location / filename).open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return str(file_location / filename)  # Return the file path

@app.post("/user/authentication")
async def authenticate_user(user_image: UploadFile = File(...), user_government_id: UploadFile = File(...), user_names: str = Body(...)):
    try:
        # Parse the string to a dictionary
        try:
            print("Parsing user names...")
            user_names_dict = json.loads(user_names)
            print(f"User names dict: {user_names_dict}")
        except json.JSONDecodeError as e:
            print(f"Error parsing user names: {e}")
            raise HTTPException(status_code=400, detail=f"Error parsing user names: {e}")

        file_location = Path(UPLOAD_DIR)
        file_location.mkdir(parents=True, exist_ok=True)

        start_time = time.time()  # Record the start time

        # Save files concurrently
        with concurrent.futures.ThreadPoolExecutor() as executor:
            user_image_path = executor.submit(save_file, user_image, "face.jpg").result()
            govt_id_path = executor.submit(save_file, user_government_id, "govt_id.jpg").result()

        print(f"User image saved to: {user_image_path}")
        print(f"Government ID saved to: {govt_id_path}")

        # Load and process the files here
        authenticator = Authenticator(
            user_image=user_image_path,
            user_names=user_names_dict,
            govt_id=govt_id_path
        )

        response = authenticator.authenticate()
        print(f"Authentication response: {response}")

        os.remove(user_image_path)
        os.remove(govt_id_path)
        return response

    except Exception as e:
        print(f"Error during authentication: {e}")
        return {"Status": "Failure", "Error": str(e)}


@app.post("/user/generate_avatar")
async def upload_file(image_file: UploadFile = File(...), drv_audio: UploadFile = File(...), webhook_id: str = Body(...)):
    file_id = str(uuid.uuid4())
    src_image_location = os.path.join(UPLOAD_DIR, f"{file_id}.png")
    drv_audio_location = os.path.join(UPLOAD_DIR, f"{file_id}.wav")
    output_file = os.path.join(RESULT_DIR, f"{file_id}_result.mp4")

    try:
        # Save the source image
        with open(src_image_location, "wb") as f:
            shutil.copyfileobj(image_file.file, f)

        # Convert the uploaded audio to WAV format and save it
        audio = AudioSegment.from_file(drv_audio.file)
        audio.export(drv_audio_location, format="wav")

        # Set file status to "in progress"
        file_statuses[file_id] = "in progress"
        file_paths[file_id] = (src_image_location, drv_audio_location)

        # Process the file synchronously
        url = process_file(file_id, src_image_location, drv_audio_location, output_file)

        # Send the webhook request
        test_url ="12413453234512"
        call_webhook(webhook_id, url)

        return JSONResponse(content={"success": True, "message": "Files uploaded and being processed", "file_id": file_id})

    except Exception as e:
        print(f"Error in generating avatar: {e}")
        return JSONResponse(content={"success": False, "message": "Failed to generate avatar"}, status_code=500)
#data/raw/examples/bg_image.jpeg'
def process_file(file_id: str, src_image_location: str, drv_audio_location: str, output_file: str) -> str:
    try:
        # Process the file

        # Specify the directory you want to list
        directory = 'data/raw/examples/bg_images'
        
        # Create an empty list to store the file paths
        file_paths = []
        
        # Walk through the directory
        for root, dirs, files in os.walk(directory):
            for file in files:
                # Join the root directory and the file name to get the full path
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
        #file_paths.append('')
        # Output the list of file paths
        random_bg_image = random.choice(file_paths)
        src_image_location = resize_to_reference(src_image_location,'data/raw/examples/Macron.png')
        inp = {
            'a2m_ckpt': 'checkpoints/240210_real3dportrait_orig/audio2secc_vae',
            'head_ckpt': '',
            'torso_ckpt': 'checkpoints/240210_real3dportrait_orig/secc2plane_torso_orig',
            'src_image_name': src_image_location,
            'bg_image_name': random_bg_image,
            'drv_audio_name': drv_audio_location,
            'drv_pose_name': 'data/raw/examples/May_5s.mp4',
            'blink_mode': 'period',
            'temperature': 0.2,
            'mouth_amp': 0.45,
            'out_name': output_file,
            'out_mode': 'concat_debug',
            'map_to_init_pose': 'True',
            'head_torso_threshold': None,
            'seed': None,
            'min_face_area_percent': 0.2,
            'low_memory_usage': True
        }

        result = GeneFace2Infer.example_run(inp)
        print(f"Saved file avatar in {result}")
        res = upload_to_imgkit(
            user_id=file_id,
            path=result
        )

        print(f"url is {res}")
        os.remove(src_image_location)
        os.remove(drv_audio_location)
        print(f"Removed drive audio and input image")

        # Update file status and output file path
        file_statuses[file_id] = "completed"
        output_files[file_id] = res

        dir = 'uploads'
        # Walk through the directory
        for root, dirs, files in os.walk(dir):
            for file in files:
                # Join the root directory and the file name to get the full path
                full_path = os.path.join(root, file)
                os.remove(full_path)
        
        return res

    except Exception as e:
        print(f"Error in processing file: {e}")
        file_statuses[file_id] = "failed"
        raise
        
def resize_to_reference(image_path, reference_image_path):
    print(f"Opening image: {image_path}")
    print(f"Opening reference image: {reference_image_path}")
    
    # Open the original and reference images
    img = Image.open(image_path)
    ref_img = Image.open(reference_image_path)
    
    # Get the size of the reference image
    ref_width, ref_height = ref_img.size
    print(f"Reference image size: {ref_width}x{ref_height}")
    
    # Resize the original image to match the reference image size
    resized_img = img.resize((ref_width, ref_height))
    print(f"Resized original image to match reference size: {ref_width}x{ref_height}")
    
    # Save the resized image back to the same location with the same name
    resized_img.save(image_path)
    return image_path
    
def call_webhook(webhook_id: str, url: str):
    webhook_url = f"https://ntgnx4sm-3000.euw.devtunnels.ms/v1/friend/webook"
    try:
        test_id = "66a667b81047f422b425597c"
        response = requests.post(webhook_url, json={"webhook_id": webhook_id, "url":url})
        response.raise_for_status()
        print(f"Webhook sent successfully: {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Error sending webhook: {e}")


def FaceSimilarity(face1, face2, is_file=True, threshold=0.55):
    try:
        # Request data
        data = {
            "threshold": f"{threshold}",
        }

        # Endpoint URL
        url = "https://api.luxand.cloud/photo/similarity"

        # Request headers
        headers = {
            "token": f"{os.getenv('LUX_API_KEY')}",
        }


        if is_file:
            # If face1 and face2 are file paths
            print(f"Sending request with files {face1} and {face2}")
            files = {
                "face1": open(face1, "rb"),
                "face2": open(face2, "rb"),
            }
            response = requests.request("POST", url, headers=headers, data=data, files=files)
        else:
            # If face1 and face2 are URLs
            print(f"Sending request with URLs {face1} and {face2}")
            data["face1"] = face1
            data["face2"] = face2
            response = requests.request("POST", url, headers=headers, data=data)

        # Get the Python dictionary from the response
        response_dict = response.json()

        print(f"Received response: {response_dict}")
        
        response_dict["status"] = "success"

        return response_dict

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"status": "error", "message": "An error occurred"}

def ocr_space(gvt_id, user_names, input_type='file', overlay=False, api_key=os.getenv("FREE_OCR_API_KEY"), language='eng'):
    """
    Sends a request to the OCR.space API with a local file or URL.

    Parameters:
    input (str): The file path & name or URL to be processed.
    user_names (str or dict): The user names to be found in the parsed text. Can be a JSON string or a dictionary.
    input_type (str, optional): 'file' or 'url'. Defaults to 'file'.
    overlay (bool, optional): Whether the OCR.space overlay is required in the response. Defaults to False.
    api_key (str, optional): The OCR.space API key. Defaults to the value of the "FREE_OCR_API_KEY" environment variable.
    language (str, optional): The language code to be used in OCR. Defaults to 'eng'.

    Returns:
    dict: A dictionary containing the status of the request, the parsed text, a success message, and a validation result.
    """
    try:
        payload = {
            'isOverlayRequired': overlay,
            'apikey': api_key,
            'language': language,
        }

        if input_type == 'file':
            with open(gvt_id, 'rb') as f:
                r = requests.post('https://api.ocr.space/parse/image',
                                  files={gvt_id: f},
                                  data=payload,
                                  )
        elif input_type == 'url':
            payload['url'] = gvt_id
            r = requests.post('https://api.ocr.space/parse/image',
                              data=payload,
                              )

        data = json.loads(r.content.decode())
        parsed_text = data["ParsedResults"][0]["ParsedText"]

        if isinstance(user_names, str):
            user_names = json.loads(user_names)

        names = list(user_names.values())
        found_names = [name for name in names if name.lower() in parsed_text.lower()]
        is_valid = 'yes' if len(found_names) >= 2 else 'no'

        result = {
            "isValid": is_valid,
            "foundNames": found_names
        }
        print(f"OCR completed")
        print(f"Parsed text: {parsed_text}")

        return {"status": "success", "parsed_text": parsed_text, "message": "Text parsed successfully", 'validation': result}

    except Exception as e:
        print(f"An error occurred: {e}")
        return {"status": "error", "message": "An error occurred"}

class Authenticator:

    def __init__(self, user_names: dict, govt_id: str, user_image: str):
        self.user_names = user_names
        self.govt_id = govt_id
        self.user_image = user_image

    def authenticate(self):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            face_future = executor.submit(FaceSimilarity, face1=self.user_image, face2=self.govt_id, is_file=True)
            ocr_future = executor.submit(ocr_space, user_names=self.user_names, gvt_id=self.govt_id, input_type="file")

            face_response = face_future.result()
            ocr_response = ocr_future.result()

        if face_response["status"] == "error" or ocr_response["status"] == "error":
            response = {
                "status": "error",
                "message": "Error in Authenticating user"
            }
            return response

        elif face_response["status"] == "success" and ocr_response["status"] == "success":
            if face_response.get("similar") and len(ocr_response["validation"]["foundNames"]) >= 1:
                response = {
                    "status": "success",
                    "authentication": "successful",
                    "score": face_response.get("score"),
                    "user_name":  f"{self.user_names}",
                    "found_names": ocr_response["validation"]["foundNames"]
                }
        elif face_response["status"] == "success" and ocr_response["status"] == "success":
            if face_response.get("similar") and len(ocr_response["validation"]["foundNames"]) < 1:
                if face_response.get("score") > 0.66:
                    response = {
                        "status": "success",
                        "authentication": "successful",
                        "score": face_response.get("score"),
                        "user_name":  f"{self.user_names}",
                        "found_names": ocr_response["validation"]["foundNames"]
                    }
            else:
                response = {
                    "status": "success",
                    "authentication": "failed",
                    "score": face_response.get("score"),
                    "user_name": f"{self.user_names}" ,
                    "found_names": ocr_response["validation"]["foundNames"]
                }
        return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=4000)
