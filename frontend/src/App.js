import React, { useState } from "react";
import axios from "axios";

function App() {
  const [message, setMessage] = useState("");
  const [uploadedImage, setUploadedImage] = useState(null);

  // Handle file upload
  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setMessage(response.data.message);
      setUploadedImage(URL.createObjectURL(file)); // Display the uploaded image
    } catch (error) {
      console.error("Error uploading image:", error);
      setMessage("Failed to upload image");
    }
  };

  return (
    <div style={{ textAlign: "center", marginTop: "50px" }}>
      <h1>{message}</h1>

      {/* File Input */}
      <input type="file" accept="image/*" onChange={handleImageUpload} />

      {/* Display Uploaded Image */}
      {uploadedImage && (
        <div style={{ marginTop: "20px" }}>
          <h3>Uploaded Image:</h3>
          <img src={uploadedImage} alt="Uploaded" style={{ width: "300px" }} />
        </div>
      )}
    </div>
  );
}

export default App;