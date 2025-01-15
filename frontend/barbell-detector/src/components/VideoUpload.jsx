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
      const response = await axios.post("http://localhost/yolo/", formData);

      console.log(response);
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

  useEffect(() => {
    if (isProcessing && (videoId != null && videoId != undefined)) {
      let timeout;
      timeout = setTimeout(async () => {
        try {
          const response = await fetch(
            `http://localhost/yolo/video/${videoId}/status`
          );

          if (!response.ok) {
            throw new Error("Failed to check status");
          }

          const data = await response.json();

          

          setVideoProgress(data.progress);

          console.log(data)

          if (data.state === "Finished") {
            setDownloadLink(
              `http://localhost/yolo/video/${videoId}`
            );
            setResultsLink(
              `http://localhost/yolo/video/${videoId}/data`
            );
            setIsProcessing(false); // Stop polling
            //TODO Get results
          }
          else if (data.state.includes("Error")){
            throw new Error(data.state)
          }
        } catch (error) {
          console.error("Error checking status:", error);
          setError(error)
        }
      }, 1000); // Poll every second

      return () => clearTimeout(timeout); // Cleanup on component unmount
    }
  }, [isProcessing, videoId]);

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
