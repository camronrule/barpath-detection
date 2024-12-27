# For API operations and standards
import os
import tempfile
from fastapi import APIRouter, UploadFile, Response, status, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
# Our detector objects
from detectors.YoloV11BarbellDetection import YoloV11BarbellDetection
# For encoding images
import cv2


# A new router object that we can add endpoints to.
# Note that the prefix is /yolo, so all endpoints from
# here on will be relative to /yolo
router = APIRouter(tags=["Video Upload and analysis"], prefix="/yolo")

# A cache of annotated images. Note that this would typically
# be some sort of persistent storage (think maybe postgres + S3)
# but for simplicity, we can keep things in memory
videos = []


@router.post("/",
             status_code=status.HTTP_201_CREATED,
             responses={
                 201: {"description": "Successfully Analyzed Video."}
             })
async def yolo_video_upload(file: UploadFile):
    """Takes a multi-part upload video, analyzes each frame, and returns an annotated video.

    Arguments:
        file (UploadFile): The multi-part upload file

    Returns:
        dict: The video ID and the download URL
    """
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    try:
        contents = await file.read()
        temp_input.write(contents)
        temp_input.close()

        dt = YoloV11BarbellDetection(temp_input.name)
        out_path, data = dt.process_video()
        videos.append(out_path)
        video_id = len(videos) - 1
        return {"id": video_id, "download_url": f"/yolo/video/{video_id}"}
    finally:
        os.unlink(temp_input.name)


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
