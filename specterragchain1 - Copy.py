import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from specterragchain1 import ResearchPaperRAG  
import logging

app = Flask(__name__, static_url_path='/static', static_folder='static')
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
    """Handle PDF upload and initialize RAG."""
    global rag
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        try:
            rag = ResearchPaperRAG(api_key=API_KEY)
            rag.load_research_paper(file_path)
            logger.info(f"Successfully loaded {file_path}")
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({'error': f"Error processing file: {str(e)}"}), 500
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/query', methods=['POST'])
def query():
    """Handle query and return results."""
    global rag
    if rag is None:
        return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
    data = request.get_json()
    question = data.get('question', '')
    if not question.strip():
        return jsonify({'error': 'Please provide a valid question'}), 400
    try:
        result = rag.query(question)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': f"Error processing query: {str(e)}"}), 500

@app.route('/results', methods=['GET'])
def results():
    """Render results page (for manual navigation, if needed)."""
    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
    
    
#web: gunicorn wsgi:app --timeout 180 --workers 2 --bind 0.0.0.0:$PORT