from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
from dotenv import load_dotenv
from werkzeug.utils import secure_filename
from check import generate_embeddings_and_save, load_and_clean_documents  # Import functions
from main import initialize_chatbot, get_chatbot_response

# Load environment variables
load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable CORS for frontend communication

UPLOAD_FOLDER = "text_files"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize chatbot globally
chatbot = initialize_chatbot()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_input = data.get("message", "")
    if not user_input:
        return jsonify({"error": "No input provided"}), 400

    try:
        # Call get_chatbot_response with only user_input; the global chatbot is used inside.
        response = get_chatbot_response(user_input)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def upload_file():
    global chatbot  # Ensure chatbot is updated

    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        print("📂 Cleaning and processing uploaded file...")
        cleaned_docs = load_and_clean_documents(app.config['UPLOAD_FOLDER'])
        generate_embeddings_and_save(cleaned_docs)

        print("🔄 Reloading chatbot with new embeddings...")
        chatbot = initialize_chatbot()  # RELOAD CHATBOT AFTER NEW FILE UPLOAD
        
        return jsonify({"message": "File uploaded and processed successfully. Chatbot updated!"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/uploaded_files", methods=["GET"])
def list_files():
    files = os.listdir(app.config['UPLOAD_FOLDER'])  # List files in 'text_files'
    return jsonify({"files": files})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render ka default PORT
    app.run(debug=True,host='0.0.0.0', port=port)
    