import os
import google.generativeai as genai
import numpy as np
import PyPDF2
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import time
import google.api_core.exceptions
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid
import sys

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for lazy loading
_embedding_model = None
_tokenizer = None
_model_loading_error = None

def get_embedding_model():
    """Lazy load embedding model to avoid startup delays."""
    global _embedding_model, _tokenizer, _model_loading_error
    
    if _model_loading_error:
        raise _model_loading_error
    
    if _embedding_model is None:
        try:
            logger.info("Loading embedding model...")
            _tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            _embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            _embedding_model.eval()
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load all-MiniLM-L6-v2: {e}. Falling back to bert-base-uncased.")
            try:
                _tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
                _embedding_model = AutoModel.from_pretrained('bert-base-uncased')
                _embedding_model.eval()
                logger.info("Fallback embedding model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load any embedding model: {e}")
                _model_loading_error = Exception(f"Could not load embedding model: {e}")
                raise _model_loading_error
    
    return _embedding_model, _tokenizer

@dataclass
class ResearchChunk:
    text: str
    page_number: int
    section: str
    chunk_id: str
    embedding: np.ndarray = None
    
def regex_sentence_tokenize(text: str) -> List[str]:
    """
    Tokenize text into sentences using regex patterns.
    Handles common abbreviations and edge cases.
    """
    # Common abbreviations that shouldn't end sentences
    abbreviations = {
        'dr', 'mr', 'mrs', 'ms', 'prof', 'inc', 'ltd', 'corp', 'co',
        'vs', 'etc', 'fig', 'e.g', 'i.e', 'al', 'et', 'cf', 'p',
        'pp', 'vol', 'no', 'ed', 'eds', 'min', 'max', 'approx',
        'dept', 'univ', 'assoc', 'dev', 'eng', 'tech', 'sci',
        'int', 'nat', 'comp', 'gen', 'spec', 'std', 'ref'
    }
    
    # First, protect abbreviations by temporarily replacing periods
    protected_text = text
    for abbr in abbreviations:
        # Match abbreviation followed by period (case insensitive)
        pattern = rf'\b{re.escape(abbr)}\.'
        replacement = f'{abbr}<PERIOD>'
        protected_text = re.sub(pattern, replacement, protected_text, flags=re.IGNORECASE)
    
    # Handle decimal numbers (protect periods in numbers)
    protected_text = re.sub(r'(\d+)\.(\d+)', r'\1<PERIOD>\2', protected_text)
    
    # Handle citations like (Smith et al. 2020)
    protected_text = re.sub(r'\bet al\. (\d{4})\)', r'et al<PERIOD> \1)', protected_text)
    
    # Split on sentence-ending punctuation followed by whitespace and capital letter
    # or end of string
    sentence_pattern = r'[.!?]+(?:\s+(?=[A-Z])|$)'
    sentences = re.split(sentence_pattern, protected_text)
    
    # Clean up and restore periods
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.replace('<PERIOD>', '.').strip()
        if sentence:  # Only add non-empty sentences
            cleaned_sentences.append(sentence)
    
    return cleaned_sentences

class ResearchPaperRAG:
    def __init__(self, api_key: str, qdrant_url: str, qdrant_api_key: str, chunk_size: int = 600, overlap: int = 50):
        logger.info("Initializing ResearchPaperRAG...")
        
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

        # Initialize Qdrant client
        self.qdrant_client = QdrantClient(
            url=qdrant_url,
            api_key=qdrant_api_key,
        )
        
        # Generate unique collection name for this session
        self.collection_name = f"research_papers_{uuid.uuid4().hex[:8]}"
        
        self.logger = logging.getLogger(__name__)
        
        # Don't load embedding model here - use lazy loading
        self.embedding_model = None
        self.tokenizer = None

        self.chunks: List[ResearchChunk] = []
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.paper_metadata = {}
        self.api_quota_exceeded = False
        self.collection_created = False

    def _ensure_embedding_model(self):
        """Ensure embedding model is loaded."""
        if self.embedding_model is None:
            self.embedding_model, self.tokenizer = get_embedding_model()

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
        """Split text into chunks using regex sentence tokenization."""
        chunks = []
        for page_data in pages_text:
            text = page_data['text']
            page_num = page_data['page_number']
            section = page_data['section']
            
            if not text.strip():
                continue
            
            # Use regex sentence tokenization
            sentences = regex_sentence_tokenize(text)
            
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
        self._ensure_embedding_model()
        
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
        
        self._ensure_embedding_model()
        
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

    def calculate_simple_confidence(self, similarities: List[float]) -> float:
        """Calculate a simple confidence score based on similarity scores."""
        if not similarities:
            return 0.0
        
        # Use weighted average with higher weight for top results
        weights = [1.0 / (i + 1) for i in range(len(similarities))]
        weighted_sum = sum(sim * weight for sim, weight in zip(similarities, weights))
        total_weight = sum(weights)
        
        return min(100.0, (weighted_sum / total_weight) * 100)

    def query(self, question: str, top_k: int = 10) -> Dict:
        """Process query and return results."""
        if not question.strip():
            self.logger.warning("Empty question provided")
            return {
                'answer': "Please provide a valid question.",
                'sources': [],
                'confidence': 0.0,
                'metadata': {}
            }
        
        relevant_chunks = self.retrieve(question, top_k)
        
        if not relevant_chunks:
            self.logger.info("No relevant chunks found")
            return {
                'answer': "No relevant information found in the paper.",
                'sources': [],
                'confidence': 0.0,
                'metadata': self.paper_metadata
            }
        
        # Calculate confidence based on similarity scores
        similarities = [sim for _, sim in relevant_chunks]
        confidence = self.calculate_simple_confidence(similarities)
        
        answer = self.generate_response(question, [chunk for chunk, _ in relevant_chunks])
        
        sources = [{
            'rank': i + 1,
            'section': chunk.section,
            'page': chunk.page_number,
            'text': chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
            'relevance_score': float(round(sim, 3))
        } for i, (chunk, sim) in enumerate(relevant_chunks)]
        
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': float(round(confidence, 1)),
            'metadata': self.paper_metadata
        }
        
        self.logger.info(f"Query processed successfully with confidence: {confidence}%")
        return result

    def cleanup(self):
        """Clean up Qdrant collection."""
        if self.collection_created:
            try:
                self.qdrant_client.delete_collection(collection_name=self.collection_name)
                self.logger.info(f"Deleted Qdrant collection: {self.collection_name}")
            except Exception as e:
                self.logger.error(f"Error deleting collection: {e}")