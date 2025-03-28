import asyncio
import logging
import os
import tempfile
from typing import Dict
from fastapi import APIRouter, File, Form, UploadFile, status, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from json import loads, dumps

from typing import Annotated

from detectors.YoloV11BarbellDetection import YoloV11BarbellDetection

# Different possible responses from GET method of video_status
STATUS_PROCESSING = "Processing"
STATUS_ERROR = "Error"
STATUS_FINISHED = "Finished"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("FastAPIApp")


# A new router object that we can add endpoints to.
# Note that the prefix is /yolo, so all endpoints from
# here on will be relative to /yolo
router = APIRouter(tags=["Video Upload and analysis"], prefix="/yolo")

# A cache of annotated images. Note that this would typically
# be some sort of persistent storage (think maybe postgres + S3)
# but for simplicity, we can keep things in memory
videos = []

# Initialize the model object only once,
# Then call detector.init_video() for each video
detector = YoloV11BarbellDetection()


@router.post("/",
             status_code=status.HTTP_201_CREATED,
             responses={
                 201: {"description": "Successfully Analyzed Video."}
             })
async def yolo_video_upload(
        file: UploadFile = File(..., description="The video to analyze"),
        lift_type: str = Form(
            ..., description="Type of lift in the video (Squat, Bench, or Deadlift)")
) -> dict:
    """Takes a multi-part upload video, analyzes each frame, and returns an annotated video.

    Arguments:
        file (UploadFile): The multi-part upload file
        lift_type (str): The type of lift in the video. "Bench", "Squat", or "Deadlift"

    Returns:
        dict: The video ID and the download URL

    Example Curl:
        curl -X 'POST' \
        'http://localhost:8080/yolo/' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'file=@IMG_6723.MOV;type=video/quicktime' \
        -F 'lift_type=Bench'
    """

    video_id = len(videos)
    logger.info(f"Receiving file {file.filename} with ID: {video_id}")
    try:

        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")

        logger.info(f"Uploading {temp_input.name}")

        # open file synchronously
        with open(temp_input.name, "wb") as f:
            f.write(await file.read())

        logger.info(f"Uploaded {temp_input.name}")

        detector.init_video(
            temp_input.name, temp_output.name, video_id, lift_type)

        # background_tasks.add_task(detector.process_video)
        asyncio.create_task(detector.process_video(video_id))

        videos.append(temp_output.name)

        return {"message": "Video uploaded successfully. Processing has started",
                "lift_type": lift_type,
                "video_id": video_id}

    except Exception as e:
        detector.update_state(video_id, f"{STATUS_ERROR}: {e}")

        if len(videos) - 1 < video_id:
            # to ensure length of arr correct
            videos.append("ENCOUNTERED ERROR")

        return {"message": f"Encountered error while uploading video. Processing cancelled. Error: {e}",
                "video_id": video_id}


@router.get("/video/{video_id}/data")
async def yolo_video_data(video_id: int) -> JSONResponse:
    """Get the data for a video by its ID.

    Arguments:
        video_id (int): The video ID to get data for

    Returns:
        JSONResponse: The JSON data for the video, as a string

    Example Curl:
        curl -X 'GET' \
        'http://localhost/yolo/video/{video_id}/data' \
        -H 'accept: application/json'
    """
    try:
        return JSONResponse(content=detector.data[video_id])
    except KeyError:
        raise HTTPException(status_code=404, detail="Video not found")


@router.get("/video/{video_id}/status")
async def yolo_video_status(video_id: int) -> dict:
    """Check the processing status of a video

    Args:
        video_id (int): The video to check the status of

    Raises:
        HTTPException: Video ID not in use

    Returns:
        dict: {"state": str "progress": float}

    Example Curl:
        curl -X 'GET' \
        'http://localhost/yolo/video/{video-id}/status' \
        -H 'accept: application/json'
    """
    if video_id not in detector.status:
        raise HTTPException(status_code=404, detail="Video not found.")

    return detector.get_status(video_id)


@router.get("/video/{video_id}",
            status_code=status.HTTP_200_OK)
async def yolo_video_download(video_id: int) -> FileResponse:
    """Download an annotated video by its ID.

    Arguments:
        video_id (int): The video ID to download

    Returns:
        Response: The annotated video file in MP4 format

    Example Curl:
        curl -X 'GET' \
        'http://localhost/yolo/video/{video-id}' \
        -H 'accept: application/json'
    """
    try:
        video_path = videos[video_id]
        video_name = os.path.basename(video_path)
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=video_name,
            headers={
                "Content-Disposition": f"attachment; filename={video_name}"}
        )
    except IndexError:
        raise HTTPException(status_code=404, detail="Video not found")
