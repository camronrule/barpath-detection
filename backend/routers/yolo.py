# For API operations and standards
import asyncio
import logging
import os
import tempfile
from typing import Dict
from fastapi import APIRouter, UploadFile, status, HTTPException
from fastapi.responses import FileResponse
# Our detector objects
from detectors.YoloV11BarbellDetection import YoloV11BarbellDetection


STATUS_UPLOADED = "Uploaded"
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

# Keep track of the status of processing for each video ID
# so that the front end knows when to update the UI for the user
# to download the video or view results
# id: status => ("Uploaded", "Processing", "Error ...", "Finished")
video_status = {}

# Initialize the model object only once,
# Then call detector.init_video() for each video
detector = YoloV11BarbellDetection()


@router.post("/",
             status_code=status.HTTP_201_CREATED,
             responses={
                 201: {"description": "Successfully Analyzed Video."}
             })
async def yolo_video_upload(file: UploadFile) -> dict:
    """Takes a multi-part upload video, analyzes each frame, and returns an annotated video.

    Arguments:
        file (UploadFile): The multi-part upload file

    Returns:
        dict: The video ID and the download URL

    Example Curl:
        curl -X 'POST' \
        'http://localhost/yolo/' \
        -H 'accept: application/json' \
        -H 'Content-Type: multipart/form-data' \
        -F 'file=@IMG_6860.MOV;type=video/quicktime'
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

        video_status[video_id] = STATUS_UPLOADED

        detector.init_video(temp_input.name, temp_output.name, video_id)

        # background_tasks.add_task(detector.process_video)
        asyncio.create_task(detector.process_video(video_id))

        videos.append(temp_output.name)

        video_status[video_id] = STATUS_PROCESSING
        return {"message": "Video uploaded successfully. Processing has started",
                "video_id": video_id}

    except Exception as e:
        video_status[video_id] = f"STATUS_ERROR: {e}"
        videos.append("ENCOUNTERED ERROR")  # to ensure length of arr correct
        return {"message": f"Encountered error while uploading video. Processing cancelled. Error: {e}",
                "video_id": video_id}


@router.get("/video/{video_id}/data")
async def yolo_video_data(video_id: int) -> str:
    """Get the data for a video by its ID.

    Arguments:
        video_id (int): The video ID to get data for

    Returns:
        str: The JSON data for the video, as a string
    """
    try:
        return detector.data[video_id]
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
        dict: Video status
    """
    if video_id not in video_status:
        raise HTTPException(status_code=404, detail="Video not found.")

    elif detector.data[video_id] == "N/A":
        video_status[video_id] = STATUS_PROCESSING

    elif video_status[video_id] == STATUS_PROCESSING and detector.data[video_id] != "N/A":
        video_status[video_id] = STATUS_FINISHED

    return {"status": video_status[video_id]}


@router.get("/video/{video_id}",
            status_code=status.HTTP_200_OK)
async def yolo_video_download(video_id: int) -> FileResponse:
    """Download an annotated video by its ID.

    Arguments:
        video_id (int): The video ID to download

    Returns:
        Response: The annotated video in MP4 format
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
    except KeyError:
        raise HTTPException(status_code=404, detail="Video not found")
