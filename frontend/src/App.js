import React, { useState } from "react";
import axios from "axios";

function App() {
  const [message, setMessage] = useState("");
  const [uploadedImage, setUploadedImage] = useState(null);
  const [chatMessages, setChatMessages] = useState([]); // State to hold chat messages
  const [outputImage, setOutputImage] = useState(null); // State to hold the output image

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

      // After uploading the image, fetch the chat messages
      fetchChatMessages();
      setOutputImage(`/outputs/${response.data.processed_image.split('/').pop()}`);
    } catch (error) {
      console.error("Error uploading image:", error);
      setMessage("Failed to upload image");
    }
  };

  // Fetch chat messages from the backend
  const fetchChatMessages = async () => {
    try {
      const response = await axios.get("/api/chat");
      setChatMessages(response.data.messages);
    } catch (error) {
      console.error("Error fetching chat messages:", error);
    }
  };

  return (
    <>
    <div style={{ display: "flex", justifyContent: "space-between", marginTop: "50px" }}>
      {/* Left Column - Image Upload */}
      <div style={{ textAlign: "center", width: "45%" }}>
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

      {/* Right Column - Image Output */}
      <div style={{ textAlign: "center", width: "45%" }}>
        {outputImage && (
          <div style={{ marginTop: "20px" }}>
            <h3>Processed Image:</h3>
            <img src={outputImage} alt="Processed" style={{ width: "300px" }} />
          </div>
        )}
      </div>
    </div>

    {/* Chat Box - Message Display*/}
    <div style={{ marginTop: "30px", border: "1px solid #ccc", borderRadius: "10px", width: "80%", margin: "30px auto", padding: "10px", backgroundColor: "#f9f9f9", height: "300px", overflowY: "scroll" }}>
      <h3>LLM Group Chat:</h3>
      {chatMessages.length > 0 ? (
        chatMessages.map((msg, index) => (
          <div key={index} style={{ textAlign: msg.name === "supervisor" ? "right" : "left", margin: "10px" }}>
            <strong>{msg.name}:</strong> {msg.content}
          </div>
        ))
      ) : (
        <p>Generating Translation...</p>
      )}
    </div>
    </>
  );
}

export default App;