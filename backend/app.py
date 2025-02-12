import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from pipeline import Pipeline

app = Flask(__name__, static_folder="build")  # Serve React app from the 'build' folder
CORS(app)  # Enable CORS for frontend communication

# Directory to store uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
OUTPUT_FOLDER = "outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)  # Create the folder if it doesn't exist
chat_history = []  # Initialize chat history

pipeline_instance = Pipeline()
print("Pipeline created!") # for testing purposes

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Helper function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# API route for uploading images
@app.route("/api/upload", methods=["POST"])
def upload_image():
    # Check if a file was included in the request
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files["file"]

    # Check if the file has a valid name and extension
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type. Only PNG, JPG, JPEG, and GIF are allowed."}), 400

    # Save the file securely
    filename = secure_filename(file.filename)
    input_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(input_path)

    # Process the image using the Pipeline class
    try:
        output_filename = f"processed_{filename}"  # Generate a unique name for the processed image
        output_path = os.path.join(OUTPUT_FOLDER, output_filename)
        global chat_history
        output_name, chat_history = pipeline_instance.process_translate_typeset(input_path, output_path)  # Call the processing function
    except Exception as e:
        return jsonify({"error": f"Image processing failed: {str(e)}"}), 500

    # Return the paths to the original and processed images
    return jsonify({
        "message": "File uploaded and processed successfully",
        "original_image": f"/uploads/{filename}",
        "chat_messages": chat_history,
        "processed_image": f"../backend/outputs/{output_filename}"
    }), 200

@app.route("/outputs/<filename>")
def processed_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

# Serve uploaded files
@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

# API route to fetch chat messages
@app.route("/api/chat", methods=["GET"])
def get_chat_messages():
    # This would return the current chat messages stored in memory
    return jsonify({"messages": chat_history}), 200

# Fallback route to serve React app
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    else:
        return send_from_directory(app.static_folder, "index.html")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)