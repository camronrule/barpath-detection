import React, { useState } from "react";
import axios from "axios";

const VideoResults = ({ videoId }) => {
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);

  const fetchResults = async () => {
    try {
      const response = await axios.get(`http://localhost/video/${videoId}/data`);
      setResults(response.data);
      setError(null);
    } catch (err) {
      setError("Failed to fetch results. Video might still be processing.");
    }
  };

  return (
    <div>
      <h2>Video Results</h2>
      <button onClick={fetchResults}>Fetch Results</button>
      {results && (
        <div>
          results
        </div>
      )}
      {error && <p style={{ color: "red" }}>{error}</p>}
    </div>
  );
};

export default VideoResults;
