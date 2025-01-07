import React, { useState } from "react";
import VideoUpload from "./components/VideoUpload";
import VideoResults from "./components/VideoResults";

const App = () => {
  const [videoId, setVideoId] = useState(null);

  return (
    <div>
      <h1>Barbell Velocity Tracker</h1>
      <VideoUpload setVideoId={setVideoId} />
      {videoId && <VideoResults videoId={videoId} />}
    </div>
  );
};

export default App;

