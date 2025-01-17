import React, { useState, useEffect } from "react";
import axios from "axios";

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [videoId, setVideoId] = useState(null);
  const [message, setMessage] = useState(null);
  const [downloadLink, setDownloadLink] = useState(null);
  const [resultsLink, setResultsLink] = useState(null);
  const [isUploading, setIsUploading] = useState(null);
  const [isProcessing, setIsProcessing] = useState(null);
  const [videoProgress, setVideoProgress] = useState(null);
  const [error, setError] = useState(null);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
  };

  const handleUpload = async () => {
    setIsProcessing(false)
    setDownloadLink(null)
    if (!file) {
      setError("Please select a file");
      return;
    }
    const formData = new FormData();
    formData.append("file", file);

    setIsUploading(true)

    try {
      const response = await axios.post("http://localhost:8080/yolo/", formData);

      if (response.status != 201)
        throw new Error("Failed to upload");

      setMessage(response.data.message);
      setVideoId(response.data.video_id);
      setIsProcessing(true)
      setError(null);

    } catch (err) {
      setError(err);
      alert(err);
      console.error(err);
    }
    finally{
      setIsUploading(false);
    }
  };

  /* 
   * While a video is being processed:
   * Repeatedly poll the status endpoint to get processing status.
   * Update progress bar as needed. If processing has finished, post 
   * annotated video and results. If error reached during processing,
   * elevate the error to the UI.
   */
  async function fetchAPIData(){
    if (isProcessing && (videoId != null && videoId != undefined)){
      const response = await fetch(`http://localhost:8080/yolo/video/${videoId}/status`);

      try {

        if (response.ok){
          const data = await response.json();
          setVideoProgress(data.progress);
  
          if (data.state === "Finished"){
            setDownloadLink(
              `http://localhost:8080/yolo/video/${videoId}`
            );
            setResultsLink(
              `http://localhost:8080/yolo/video/${videoId}/data`
            );
            // Stop polling
            setIsProcessing(false); 
            return;
          }
  
          else if (data.state.includes("Error")){
            throw new Error(`Error processing video: ${data.state}`);
          }
        }
  
        else{ // fetch failed
          throw new Error("Failed to poll status.");
        }
        // poll for new status in 1 second
        setTimeout(fetchAPIData, 1000);
      }

      // error when parsing status
      catch (e) {
        setError(e);
        console.error(e);
        return;
      } 
    }
  }
  fetchAPIData();

  return (
    <div>
      <h2>Upload Your Video</h2>
      <input type="file" accept="video/*" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload</button>

      {isUploading && <p>Uploading your video, please wait...</p>}
      {isProcessing && <p>Processing your video, please wait...</p>}

      {message && !isProcessing && !downloadLink && <p>{message}</p>}
      {videoId != null && videoId != undefined && <p> Video ID: {videoId}</p>}

      {(isUploading || isProcessing) && (
        <progress value={videoProgress} />
      )}

      {(!isProcessing && !isUploading) && downloadLink && (
            <a href={downloadLink} download>
              <button>Download Processed Video ({videoId}) </button>
            </a>
          )}
      {(!isProcessing && !isUploading) && resultsLink && (
          <a href={resultsLink} download>
            <button>Download Results of Video ({videoId}) </button>
          </a>
        )}

      {error && <p style={{ color: "red" }}>{JSON.stringify(error)}</p>}
    </div>
  );
};

export default VideoUpload;
