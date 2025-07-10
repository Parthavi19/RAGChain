import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import logging
import time
import uuid
import tempfile
from research_rag import ResearchPaperRAG

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration - Use environment variables for production
API_KEY = os.environ.get('GOOGLE_API_KEY', "AIzaSyDw-MBI6oRRLNGEz8LksrgkPnAj0vSZeV4")
QDRANT_URL = os.environ.get('QDRANT_URL', "https://004fed81-613d-49f3-a9d4-159a745114b0.europe-west3-0.gcp.cloud.qdrant.io:6333")
QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY', "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4jW6kmZPwAbF-O_wIZ8xqlMBcVGSpfc8GrwPkXGo2fE")

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to store RAG instances per session
rag_instances = {}

def allowed_file(filename):
    """Check if the file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'

def get_session_id():
    """Get or create session ID."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

# Add startup health check
@app.route('/startup', methods=['GET'])
def startup_check():
    """Startup readiness check."""
    try:
        # Basic health check - don't load models here
        return jsonify({
            'status': 'ready',
            'timestamp': time.time(),
            'message': 'Application started successfully'
        }), 200
    except Exception as e:
        logger.error(f"Startup check failed: {e}")
        return jsonify({'status': 'not_ready', 'error': str(e)}), 500

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and initialize RAG."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        try:
            session_id = get_session_id()
            
            # Clean up previous RAG instance if exists
            if session_id in rag_instances:
                rag_instances[session_id].cleanup()
                del rag_instances[session_id]
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                file.save(tmp_file.name)
                file_path = tmp_file.name
            
            # Initialize RAG (models will be loaded lazily)
            rag = ResearchPaperRAG(
                api_key=API_KEY,
                qdrant_url=QDRANT_URL,
                qdrant_api_key=QDRANT_API_KEY
            )
            
            # Load the research paper
            rag.load_research_paper(file_path)
            
            # Store RAG instance
            rag_instances[session_id] = rag
            
            # Clean up temporary file
            os.unlink(file_path)
            
            logger.info(f"Successfully loaded file for session {session_id}")
            return jsonify({'message': 'File uploaded and processed successfully'}), 200
            
        except Exception as e:
            logger.error(f"Error processing file: {e}")
            return jsonify({'error': f"Error processing file: {str(e)}"}), 500
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/query', methods=['POST'])
def query():
    """Handle query and return results."""
    session_id = get_session_id()
    
    if session_id not in rag_instances:
        return jsonify({'error': 'No document loaded. Please upload a PDF first.'}), 400
    
    data = request.get_json()
    question = data.get('question', '')
    
    if not question.strip():
        return jsonify({'error': 'Please provide a valid question'}), 400
    
    try:
        rag = rag_instances[session_id]
        result = rag.query(question)
        return jsonify(result), 200
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        return jsonify({'error': f"Error processing query: {str(e)}"}), 500

@app.route('/results', methods=['GET'])
def results():
    """Render results page."""
    return render_template('results.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'timestamp': time.time()}), 200

@app.route('/cleanup', methods=['POST'])
def cleanup_session():
    """Clean up session resources."""
    session_id = get_session_id()
    
    if session_id in rag_instances:
        rag_instances[session_id].cleanup()
        del rag_instances[session_id]
        return jsonify({'message': 'Session cleaned up successfully'}), 200
    
    return jsonify({'message': 'No active session found'}), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large'}), 413

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Get port from environment variable, default to 8080 for Cloud Run
    port = int(os.environ.get('PORT', 5000))
    
    # Log startup info
    logger.info(f"Starting application on port {port}")
    
    # Remove debug=True for production
    app.run(host='0.0.0.0', port=port, debug=False)