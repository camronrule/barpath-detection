import React, { useState, useEffect } from "react";
import axios from "axios";

const VideoUpload = () => {
  const [file, setFile] = useState(null);
  const [lift_type, setLiftType] = useState(null);
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

  const handleRadioChange = (e) => {
    setLiftType(e.target.value);
    console.log(e.target.value);
  };

  const handleUpload = async () => {

    // prevent user from uploading another video while one is processing
    if (isProcessing || isUploading){
      alert(`Please wait for video ${videoId} to finish before uploading another.`);
      return;
    }

    // lift_type must be selected before uploading
    if (!lift_type){
      alert("Please select a lift type before uploading your video.")
      return;
    }

    if (!file) {
      setError("Please select a file");
      return;
    }

    setIsProcessing(false);
    setDownloadLink(null);
    setIsUploading(true);

    // pass lift type and file to backend in a multipart form
    const formData = new FormData();
    formData.append("lift_type", lift_type);
    formData.append("file", file);

    try {
      const response = await axios.post("http://localhost:8080/yolo/", formData);

      if (response.status != 201)
        throw new Error("Failed to upload");

      // upload to API successful
      setMessage(response.data.message);
      setVideoId(response.data.video_id);
      setIsProcessing(true)
      setError(null);

      // unsuccessful upload to API
    } catch (err) {
      setError(err);
      alert(err);
      console.error(err);
    }
    // finished working with this video
    finally{
      setIsUploading(false);
      setLiftType(null); // reset lift type for next upload
    }
  };

  /* 
   * While a video is being processed:
   * Repeatedly poll the status endpoint to get processing status.
   * Update progress bar as needed. If processing has finished, post 
   * annotated video and results. If error reached during processing,
   * elevate the error to the UI.
   */
  let timeout;
  var keep_polling = true;
  useEffect(() => {
    if (isProcessing && 
      (videoId != null && videoId != undefined)) {
      timeout = setTimeout(function() {poll();}, 500); 
    }
    return () => clearTimeout(timeout);
  }, [isProcessing, videoId, error, videoProgress, downloadLink, resultsLink, setIsProcessing, setError, setVideoProgress, setDownloadLink, setResultsLink]);

  /*
   * Repeatedly poll the status endpoint to get processing status.
   * `data` received from the status endpoint has two fields:
   * `state` and `progress`. `state` is either "Processing", "Finished", or includes "Error".
   * `progress` is a float between 0 and 1, representing the percentage of the video 
   * that has been processed.
   */
  const poll = async () => {
    try {
      const response = await fetch(
        `http://localhost:8080/yolo/video/${videoId}/status`
      );
  
      if (!response.ok) {
        throw new Error("Failed to check status");
      }
  
      const data = await response.json();
      console.log(`Poll response: ${JSON.stringify(data)}`);
  
      // update progress
      setVideoProgress(data.progress);
  
      // and then update state

      // if state is OK,
      if (data.state === "Finished") {
        setDownloadLink(
          `http://localhost:8080/yolo/video/${videoId}`
        );
        setResultsLink(
          `http://localhost:8080/yolo/video/${videoId}/data`
        );
        setIsProcessing(false);
        keep_polling = false;
      } 
      // and if state is NOT OK,
      else if (data.state.includes("Error")) {
        throw new Error(data.state);
      }
    } 
    // just in case we reached an error when polling
    catch (e) {
      setError(e.message);
      setIsProcessing(false);
      console.error(e.message);
      keep_polling = false;
    }
    // reset the poll to continue polling every 500ms
    if (keep_polling){
      clearTimeout(timeout);
      timeout = setTimeout(function() {poll();}, 500);
    }
    // if we are done polling, ensure the poll will not continue    
    else
      clearTimeout(timeout);
  };
  

  return (
    <div>

      {(!isUploading && !isProcessing) && 
        <div>
          <h2>Upload Your Video</h2>
          <legend>Select lift type:</legend>
          <div>
            <input type="radio" id="radio_bench" name="lift_type" value="Bench" onClick={handleRadioChange}></input>
            <label htmlFor="radio_bench">Bench</label>

            <input type="radio" id="radio_squat" name="lift_type" value="Squat" onClick={handleRadioChange}></input>
            <label htmlFor="radio_squat">Squat</label>

            <input type="radio" id="radio_deadlift" name="lift_type" value="Deadlift" onClick={handleRadioChange}></input>
            <label htmlFor="radio_deadlift">Deadlift</label>
          </div>
          <input type="file" accept="video/*" onChange={handleFileChange}/>
          <button onClick={handleUpload}>Upload</button>
        </div>
        }
      
      {isUploading && <p>Uploading your video, please wait...</p>}
      {isProcessing && !error && <p>Processing your video, please wait...</p>}

      {message && !isProcessing && !downloadLink && !error && <p>{message}</p>}
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

      {error && <div style={{ color: "red" }}>{error}</div>}
    </div>
  );
};

export default VideoUpload;
