import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
import google.generativeai as genai
import numpy as np
import PyPDF2
import re
import nltk
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModel
import torch
from datasets import Dataset
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import time
import google.api_core.exceptions
import pkg_resources
from ragas.metrics import context_precision, context_recall, answer_relevancy, faithfulness
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import tempfile
import shutil

# Initialize Flask app
app = Flask(__name__, static_url_path='/static', static_folder='static')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configuration
API_KEY = "AIzaSyDw-MBI6oRRLNGEz8LksrgkPnAj0vSZeV4"
QDRANT_URL = "https://004fed81-613d-49f3-a9d4-159a745114b0.europe-west3-0.gcp.cloud.qdrant.io:6333"  # Replace with your Qdrant URL
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4jW6kmZPwAbF-O_wIZ8xqlMBcVGSpfc8GrwPkXGo2fE"  # Replace with your Qdrant API key

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download NLTK resources
def download_nltk_resources():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab')
    try:
        nltk.data.find('taggers/maxent_ne_chunker')
    except LookupError:
        nltk.download('maxent_ne_chunker')
    try:
        nltk.data.find('corpora/words')
    except LookupError:
        nltk.download('words')

# Download NLTK resources at startup
download_nltk_resources()

@dataclass
class ResearchChunk:
    text: str
    page_number: int
    section: str
    chunk_id: str
    embedding: np.ndarray = None

class ResearchPaperRAG:
    def __init__(self, api_key: str, qdrant_url: str, qdrant_api_key: str, chunk_size: int = 600, overlap: int = 50):
        # Initialize Gemini
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=30,
            )
        )
        
        # Initialize LangChain components for RAGAS
        self.langchain_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        ))
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Generate unique collection name for this session
        self.collection_name = f"research_papers_{uuid.uuid4().hex[:8]}"
        
        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading embedding model...")
        
        # Initialize embedding model
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load all-MiniLM-L6-v2: {e}. Falling back to bert-base-uncased.")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.embedding_model = AutoModel.from_pretrained('bert-base-uncased')
        
        self.embedding_model.eval()

        self.chunks: List[ResearchChunk] = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.paper_metadata = {}
        self.api_quota_exceeded = False
        self.collection_created = False

        # Setup RAGAS metrics
        ragas_version = pkg_resources.get_distribution("ragas").version
        self.logger.info(f"Detected ragas version: {ragas_version}")
        use_embeddings = ragas_version >= "0.2.0"

        self.metrics = [
            context_precision,
            context_recall,
            answer_relevancy,
            faithfulness
        ]
        
        for metric in self.metrics:
            try:
                metric.llm = self.langchain_llm
                if use_embeddings and hasattr(metric, 'embeddings'):
                    metric.embeddings = self.embeddings
                self.logger.debug(f"Assigned LLM and embeddings to metric: {metric.name}")
            except Exception as e:
                self.logger.error(f"Failed to assign LLM/embeddings to metric {metric.name}: {e}")

    def clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        return text.strip()

    def extract_metadata_and_text(self, pdf_path: str) -> List[Dict]:
        """Extract text and metadata from PDF."""
        self.logger.info(f"Extracting text from PDF: {pdf_path}")
        pages_text = []
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    text = page.extract_text()
                    if text:
                        cleaned_text = self.clean_text(text)
                        section = self._infer_section(cleaned_text, page_num)
                        pages_text.append({
                            'text': cleaned_text,
                            'page_number': page_num + 1,
                            'section': section
                        })
        except Exception as e:
            self.logger.error(f"Error reading PDF: {e}")
            raise
        return pages_text

    def _infer_section(self, text: str, page_num: int) -> str:
        """Infer section type based on text content."""
        text_lower = text.lower()
        if page_num <= 2 and 'abstract' in text_lower:
            return 'Abstract'
        elif 'introduction' in text_lower[:200]:
            return 'Introduction'
        elif any(keyword in text_lower[:200] for keyword in ['method', 'experiment', 'approach']):
            return 'Methodology'
        elif any(keyword in text_lower[:200] for keyword in ['result', 'finding', 'evaluation']):
            return 'Results'
        elif any(keyword in text_lower[:200] for keyword in ['conclusion', 'discussion']):
            return 'Conclusion'
        return 'Other'

    def chunk_text(self, pages_text: List[Dict]) -> List[ResearchChunk]:
        """Split text into chunks."""
        chunks = []
        for page_data in pages_text:
            text = page_data['text']
            page_num = page_data['page_number']
            section = page_data['section']
            
            if not text.strip():
                continue
                
            sentences = nltk.sent_tokenize(text)
            current_chunk = ""
            
            for sentence in sentences:
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                
                if len(potential_chunk) <= self.chunk_size:
                    current_chunk = potential_chunk
                else:
                    if current_chunk.strip():
                        chunk_id = str(uuid.uuid4())
                        chunks.append(ResearchChunk(
                            text=current_chunk.strip(),
                            page_number=page_num,
                            section=section,
                            chunk_id=chunk_id
                        ))
                    
                    current_chunk = sentence[-self.overlap:] + " " + sentence if len(sentence) > self.overlap else sentence
            
            if current_chunk.strip():
                chunk_id = str(uuid.uuid4())
                chunks.append(ResearchChunk(
                    text=current_chunk.strip(),
                    page_number=page_num,
                    section=section,
                    chunk_id=chunk_id
                ))
        
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, chunks: List[ResearchChunk]) -> np.ndarray:
        """Create embeddings for text chunks."""
        self.logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        texts = [f"{chunk.section}: {chunk.text}" for chunk in chunks]
        embeddings = []

        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.embedding_model(**inputs)
                batch_embeddings = outputs.last_hidden_state[:, 0, :]
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
            
            embeddings.append(batch_embeddings.cpu().numpy())

        embeddings = np.vstack(embeddings)
        
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
            
        return embeddings

    def create_qdrant_collection(self, embedding_dim: int):
        """Create Qdrant collection for storing vectors."""
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=embedding_dim,
                    distance=Distance.COSINE
                )
            )
            self.collection_created = True
            self.logger.info(f"Created Qdrant collection: {self.collection_name}")
        except Exception as e:
            self.logger.error(f"Error creating Qdrant collection: {e}")
            raise

    def store_in_qdrant(self, chunks: List[ResearchChunk]):
        """Store chunks and embeddings in Qdrant."""
        if not self.collection_created:
            embedding_dim = chunks[0].embedding.shape[0]
            self.create_qdrant_collection(embedding_dim)
        
        points = []
        for chunk in chunks:
            points.append(PointStruct(
                id=chunk.chunk_id,
                vector=chunk.embedding.tolist(),
                payload={
                    "text": chunk.text,
                    "page_number": chunk.page_number,
                    "section": chunk.section
                }
            ))
        
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        self.logger.info(f"Stored {len(points)} points in Qdrant")

    def extract_metadata(self, pages_text: List[Dict]):
        """Extract paper metadata."""
        first_page = pages_text[0]['text'] if pages_text else ""
        abstract = ""
        title = ""
        authors = ""
        
        for page in pages_text[:2]:
            text_lower = page['text'].lower()
            if 'abstract' in text_lower:
                abstract_start = text_lower.find('abstract')
                abstract = page['text'][abstract_start:abstract_start + 300].strip()
            
            if page['page_number'] == 1:
                lines = page['text'].split('. ')
                title = lines[0].strip() if lines else ""
                authors = lines[1].strip() if len(lines) > 1 else ""
        
        self.paper_metadata = {
            'title': title,
            'authors': authors,
            'abstract': abstract
        }

    def load_research_paper(self, pdf_path: str):
        """Load and process research paper."""
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        self.logger.info(f"Loading research paper: {pdf_path}")
        
        # Extract text and metadata
        pages_text = self.extract_metadata_and_text(pdf_path)
        self.extract_metadata(pages_text)
        
        # Create chunks and embeddings
        self.chunks = self.chunk_text(pages_text)
        embeddings = self.create_embeddings(self.chunks)
        
        # Store in Qdrant
        self.store_in_qdrant(self.chunks)
        
        self.logger.info(f"Loaded research paper with {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[ResearchChunk, float]]:
        """Retrieve relevant chunks from Qdrant."""
        if not self.collection_created:
            self.logger.error("No document loaded")
            raise ValueError("No document loaded")
        
        # Create query embedding
        query_inputs = self.tokenizer(
            [query],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            query_outputs = self.embedding_model(**query_inputs)
            query_embedding = query_outputs.last_hidden_state[:, 0, :]
            query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        
        query_embedding = query_embedding.cpu().numpy()[0]
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2
        )
        
        results = []
        for result in search_results:
            payload = result.payload
            if "publication date" not in payload["text"].lower():
                chunk = ResearchChunk(
                    text=payload["text"],
                    page_number=payload["page_number"],
                    section=payload["section"],
                    chunk_id=result.id
                )
                results.append((chunk, result.score))
            
            if len(results) >= top_k:
                break
        
        self.logger.debug(f"Retrieved {len(results)} chunks for query: {query}")
        return results

    def generate_response(self, query: str, context_chunks: List[ResearchChunk]) -> str:
        """Generate response using Gemini."""
        context = "\n\n".join([f"[Section: {chunk.section}, Page {chunk.page_number}]\n{chunk.text}"
                               for chunk in context_chunks])
        
        prompt = f"""You are a research assistant specializing in academic papers. Answer the question based only on the provided context and metadata, using technical terms as they appear in the paper.

PAPER METADATA:
Title: {self.paper_metadata.get('title', 'Unknown')}
Authors: {self.paper_metadata.get('authors', 'Unknown')}
Abstract: {self.paper_metadata.get('abstract', 'Not extracted')}

CONTEXT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide a complete and detailed answer, including all relevant information from the context.
2. Use bullet points for key points or lists to ensure clarity.
3. Cite sections and pages from the context to support your answer.
4. If information is missing, state: "Information not found in the provided context."
5. Avoid speculation or external knowledge.
6. Preserve technical terms and definitions from the paper.
7. Do not truncate the response; provide the full answer regardless of length.

ANSWER:"""
        
        retries = 3
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                return response.text.strip()
            except google.api_core.exceptions.ResourceExhausted as e:
                self.logger.warning(f"API quota exceeded on attempt {attempt + 1}: {e}")
                self.api_quota_exceeded = True
                if attempt < retries - 1:
                    time.sleep(60)
                else:
                    self.logger.error(f"Failed to generate response after {retries} attempts: {e}")
                    return f"Error generating response: API quota exceeded."
            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                return f"Error generating response: {str(e)}"

    def evaluate_response(self, query: str, answer: str, context_chunks: List[ResearchChunk]) -> Dict:
        """Evaluate the LLM response using RAGAs metrics."""
        contexts = [chunk.text for chunk in context_chunks if chunk.text.strip()]
        ground_truth = " ".join(contexts) if contexts else "No relevant context found."
        
        if not contexts or all("publication date" in ctx.lower() for ctx in contexts):
            self.logger.warning("Retrieved contexts are uninformative")
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "response_relevancy": 0.0,
                "faithfulness": 0.0
            }
        
        data = {
            "question": [query],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth]
        }
        
        try:
            dataset = Dataset.from_dict(data)
            self.logger.debug(f"Evaluation dataset: {data}")
        except Exception as e:
            self.logger.error(f"Error creating dataset: {e}")
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "response_relevancy": 0.0,
                "faithfulness": 0.0
            }

        try:
            result = evaluate(
                dataset=dataset,
                metrics=self.metrics,
                llm=self.langchain_llm,
                embeddings=self.embeddings if pkg_resources.get_distribution("ragas").version >= "0.2.0" else None,
                raise_exceptions=True
            )
            
            eval_result = {
                "context_precision": float(result.get("context_precision", 0.0)),
                "context_recall": float(result.get("context_recall", 0.0)),
                "response_relevancy": float(result.get("answer_relevancy", 0.0)),
                "faithfulness": float(result.get("faithfulness", 0.0))
            }
            
            self.logger.info(f"Evaluation metrics: {eval_result}")
            return eval_result
            
        except Exception as e:
            self.logger.error(f"Error evaluating response: {str(e)}")
            return {
                "context_precision": 0.0,
                "context_recall": 0.0,
                "response_relevancy": 0.0,
                "faithfulness": 0.0
            }

    def query(self, question: str, top_k: int = 10) -> Dict:
        """Process query and return results."""
        if not question.strip():
            self.logger.warning("Empty question provided")
            return {
                'answer': "Please provide a valid question.",
                'sources': [],
                'confidence': 0.0,
                'metadata': {},
                'evaluation': {
                    'context_precision': 0.0,
                    'context_recall': 0.0,
                    'response_relevancy': 0.0,
                    'faithfulness': 0.0
                }
            }
        
        relevant_chunks = self.retrieve(question, top_k)
        
        if not relevant_chunks:
            self.logger.info("No relevant chunks found")
            return {
                'answer': "No relevant information found in the paper.",
                'sources': [],
                'confidence': 0.0,
                'metadata': self.paper_metadata,
                'evaluation': {
                    'context_precision': 0.0,
                    'context_recall': 0.0,
                    'response_relevancy': 0.0,
                    'faithfulness': 0.0
                }
            }
        
        avg_similarity = sum(sim for _, sim in relevant_chunks) / len(relevant_chunks)
        confidence = avg_similarity * 100
        
        answer = self.generate_response(question, [chunk for chunk, _ in relevant_chunks])
        
        sources = [{
            'rank': i + 1,
            'section': chunk.section,
            'page': chunk.page_number,
            'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            'relevance_score': float(round(sim, 3))
        } for i, (chunk, sim) in enumerate(relevant_chunks)]
        
        evaluation = self.evaluate_response(question, answer, [chunk for chunk, _ in relevant_chunks])
        
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': float(round(confidence, 1)),
            'metadata': self.paper_metadata,
            'evaluation': evaluation
        }
        
        self.logger.info(f"Query result: {result}")
        return result

    def cleanup(self):
        """Clean up Qdrant collection."""
        if self.collection_created:
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                self.logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            except Exception as e:
                self.logger.error(f"Error deleting collection: {e}")


# Global variable to store RAG instances per session
rag_instances = {}

def allowed_file(filename):
    """Check if the file is a PDF."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_session_id():
    """Get or create session ID."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
    return session['session_id']

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
            
            # Initialize RAG
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
    return jsonify({'status': 'healthy'}), 200

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
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)