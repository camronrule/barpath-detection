import React, { useState } from "react";
import VideoUpload from "./components/VideoUpload";

const App = () => {
  const [videoId, setVideoId] = useState(null);

  return (
    <div>
      <h1>Barbell Velocity Tracker</h1>
      <VideoUpload setVideoId={setVideoId} />
    </div>
  );
};

export default App;

