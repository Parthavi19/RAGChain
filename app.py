import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template
import json
from werkzeug.utils import secure_filename
from specterragchain1 import ResearchPaperRAG  
import logging

app = Flask(__name__,static_url_path="/static",static_folder="static")
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize RAG (will be set after file upload)
rag = None
API_KEY = "AIzaSyDw-MBI6oRRLNGEz8LksrgkPnAj0vSZeV4"                                   

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def allowed_file(filename):
    """Check if the file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global rag
    logger.info("Received upload request")
    if 'file' not in request.files:
        logger.error("No file part in request")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"Saved file to {file_path}")
        try:
            rag = ResearchPaperRAG(api_key=API_KEY)
            logger.info("Initializing RAG with paper")
            rag.load_research_paper(file_path)
            logger.info("RAG initialized successfully")
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return jsonify({'error': f"Error processing file: {str(e)}"}), 500
    logger.error("Invalid file type")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/query', methods=['POST'])
def query():
    global rag
    logger.info("Received query request")
    if rag is None:
        logger.error("No document loaded")
        return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
    try:
        data = request.get_json(force=True, silent=False)
        logger.info(f"Request JSON: {data}")
        question = data.get('question', '')
        if not question.strip():
            logger.error("Invalid or empty question")
            return jsonify({'error': 'Please provide a valid question'}), 400
        result = rag.query(question)
        logger.info("Query processed successfully")
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        return jsonify({'error': f"Error processing query: {str(e)}"}), 500

@app.route('/results', methods=['GET'])
def results():
    question = request.args.get('question', '')
    result = request.args.get('result', '{}')
    try:
        result = json.loads(result)
        
    except json.JSONDecodeError:
        result = {'answer': '', 'confidence': 0, 'metadata': {}, 'evaluation': {}, 'sources': []}
    return render_template('results.html', question=question, result=result)

if __name__ == '__main__':
    app.run(debug=True)