import os
import google.generativeai as genai
import numpy as np
import faiss
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
from docling.document_converter import DocumentConverter
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import PdfFormatOption
import json

# Download NLTK resources with better error handling
def download_nltk_resources():
    """Download required NLTK resources with proper error handling."""
    resources_to_download = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab')
    ]
    
    for resource_path, resource_name in resources_to_download:
        try:
            nltk.data.find(resource_path)
            print(f"NLTK resource {resource_name} already available")
        except LookupError:
            try:
                print(f"Downloading NLTK resource: {resource_name}")
                nltk.download(resource_name, quiet=True)
                print(f"Successfully downloaded {resource_name}")
            except Exception as e:
                print(f"Warning: Could not download {resource_name}: {e}")
                # For punkt_tab, try alternative if it fails
                if resource_name == 'punkt_tab':
                    try:
                        nltk.download('punkt', quiet=True)
                        print("Downloaded punkt as alternative to punkt_tab")
                    except Exception as e2:
                        print(f"Warning: Could not download punkt alternative: {e2}")

# Initialize NLTK resources
download_nltk_resources()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def convert_to_json_serializable(obj):
    """Convert numpy types to JSON serializable types."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    else:
        return obj

@dataclass
class ResearchChunk:
    text: str
    page_number: int
    section: str
    heading: str = ""
    element_type: str = ""
    embedding: np.ndarray = None
    context_window: str = ""  # Additional context from surrounding text

class ResearchPaperRAG:
    def __init__(self, api_key: str, chunk_size: int = 1000, overlap: int = 200):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=30,
            )
        )
        self.langchain_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=api_key,
            temperature=0.1
        ))
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )

        self.logger = logging.getLogger(__name__)
        self.logger.info("Loading embedding model...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
            self.embedding_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load all-MiniLM-L6-v2: {e}. Falling back to bert-base-uncased.")
            self.tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.embedding_model = AutoModel.from_pretrained('bert-base-uncased')
        self.embedding_model.eval()

        # Initialize Docling converter
        self.logger.info("Initializing Docling document converter...")
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = False
        pipeline_options.do_table_structure = True
        pipeline_options.table_structure_options.do_cell_matching = True
        
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            }
        )

        self.chunks: List[ResearchChunk] = []
        self.raw_text_elements: List[Dict] = []  # Store raw elements for context
        self.index = None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.paper_metadata = {}
        self.api_quota_exceeded = False

        # Initialize RAGAS metrics
        ragas_version = pkg_resources.get_distribution("ragas").version
        self.logger.info(f"Detected ragas version: {ragas_version}")
        use_embeddings = ragas_version >= "0.2.0"

        self.metrics = [context_precision, context_recall, answer_relevancy, faithfulness]
        for metric in self.metrics:
            try:
                metric.llm = self.langchain_llm
                if use_embeddings and hasattr(metric, 'embeddings'):
                    metric.embeddings = self.embeddings
            except Exception as e:
                self.logger.error(f"Failed to assign LLM/embeddings to metric {metric.name}: {e}")

    def clean_text(self, text: str) -> str:
        """Enhanced text cleaning with better preservation of structure."""
        if not text:
            return ""
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        
        # Fix hyphenated words at line breaks
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        
        # Remove page numbers and headers/footers
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\d+\s*$', '', text)  # Remove trailing page numbers
        
        # Fix spacing around punctuation
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        text = re.sub(r'([.!?])\s*([A-Z])', r'\1 \2', text)
        
        return text.strip()

    def extract_paper_metadata(self, document) -> Dict:
        """Extract comprehensive metadata from the document."""
        metadata = {
            'title': 'Unknown',
            'authors': 'Unknown',
            'abstract': 'Not extracted',
            'keywords': [],
            'sections': []
        }
        
        try:
            # Try to extract title from document metadata
            if hasattr(document, 'title') and document.title:
                metadata['title'] = document.title
            
            # Try to extract authors
            if hasattr(document, 'authors') and document.authors:
                metadata['authors'] = ', '.join(document.authors)
            
            # Extract from text elements if metadata is not available
            if metadata['title'] == 'Unknown' or metadata['authors'] == 'Unknown':
                for element in document.texts[:10]:  # Check first 10 elements
                    if hasattr(element, 'text') and element.text:
                        text = element.text.strip()
                        element_type = getattr(element, 'label', '')
                        
                        # Try to identify title
                        if (element_type == 'title' or 
                            (len(text) < 200 and not text.endswith('.') and 
                             any(word in text.lower() for word in ['analysis', 'study', 'approach', 'method', 'framework']))):
                            if metadata['title'] == 'Unknown':
                                metadata['title'] = text
                        
                        # Try to identify abstract
                        if ('abstract' in text.lower()[:20] or element_type == 'abstract'):
                            metadata['abstract'] = text
                        
                        # Try to identify keywords
                        if 'keywords' in text.lower() or 'key words' in text.lower():
                            keywords_text = re.sub(r'keywords?:?\s*', '', text.lower())
                            metadata['keywords'] = [kw.strip() for kw in keywords_text.split(',')]
            
        except Exception as e:
            self.logger.warning(f"Error extracting metadata: {e}")
        
        return metadata

    def extract_metadata_and_text_with_docling(self, source: str) -> List[Dict]:
        """Enhanced text extraction with better structure preservation."""
        self.logger.info(f"Extracting text using Docling: {source}")
        
        try:
            result = self.converter.convert(source)
            document = result.document
            
            # Extract comprehensive metadata
            self.paper_metadata = self.extract_paper_metadata(document)
            
            # Process document structure
            text_elements = []
            current_section = "Introduction"
            current_heading = ""
            
            for element in document.texts:
                if hasattr(element, 'text') and element.text:
                    text = self.clean_text(element.text)
                    if not text or len(text) < 10:  # Skip very short texts
                        continue
                    
                    element_type = getattr(element, 'label', 'text')
                    page_number = getattr(element, 'page', 1)
                    
                    # Update section and heading based on element type
                    if element_type in ['title', 'section-header', 'heading']:
                        current_heading = text
                        new_section = self._categorize_section(text)
                        if new_section != "Other":
                            current_section = new_section
                    
                    # For regular text, try to infer section if not already set
                    if element_type == 'text':
                        inferred_section = self._infer_section_from_content(text)
                        if inferred_section and inferred_section != "Other":
                            current_section = inferred_section
                    
                    text_elements.append({
                        'text': text,
                        'page_number': page_number,
                        'section': current_section,
                        'heading': current_heading,
                        'element_type': element_type
                    })
            
            # Fallback to markdown if no structured elements
            if not text_elements:
                self.logger.warning("No structured elements found, using markdown export")
                markdown_content = document.export_to_markdown()
                text_elements = self._parse_markdown_content(markdown_content)
            
            # Store raw elements for context building
            self.raw_text_elements = text_elements
            
            self.logger.info(f"Extracted {len(text_elements)} text elements")
            return text_elements
            
        except Exception as e:
            self.logger.error(f"Error extracting with Docling: {e}")
            raise

    def _categorize_section(self, heading: str) -> str:
        """Enhanced section categorization."""
        heading_lower = heading.lower()
        
        # Define section keywords with priorities
        section_keywords = {
            'Abstract': ['abstract', 'summary'],
            'Introduction': ['introduction', 'intro', 'background', 'motivation'],
            'Related Work': ['related work', 'literature review', 'prior work', 'previous work'],
            'Methodology': ['method', 'approach', 'technique', 'framework', 'architecture', 'design', 'implementation'],
            'Results': ['result', 'experiment', 'evaluation', 'finding', 'performance', 'analysis'],
            'Discussion': ['discussion', 'interpretation', 'implication', 'limitation'],
            'Conclusion': ['conclusion', 'future work', 'summary', 'final', 'closing']
        }
        
        for section, keywords in section_keywords.items():
            if any(keyword in heading_lower for keyword in keywords):
                return section
        
        return 'Other'

    def _infer_section_from_content(self, text: str) -> Optional[str]:
        """Infer section from content with improved heuristics."""
        text_lower = text.lower()
        first_100 = text_lower[:100]
        
        # Check for section indicators in the beginning of text
        if any(keyword in first_100 for keyword in ['this paper', 'we present', 'in this work']):
            return 'Introduction'
        elif any(keyword in first_100 for keyword in ['previous work', 'related work', 'literature']):
            return 'Related Work'
        elif any(keyword in first_100 for keyword in ['our method', 'our approach', 'algorithm', 'framework']):
            return 'Methodology'
        elif any(keyword in first_100 for keyword in ['experiment', 'evaluation', 'result', 'performance']):
            return 'Results'
        elif any(keyword in first_100 for keyword in ['discussion', 'analysis', 'interpretation']):
            return 'Discussion'
        elif any(keyword in first_100 for keyword in ['conclusion', 'summary', 'future work']):
            return 'Conclusion'
        
        return None

    def _parse_markdown_content(self, markdown_content: str) -> List[Dict]:
        """Parse markdown content as fallback."""
        # This is a placeholder implementation
        # You would need to implement actual markdown parsing logic here
        return [{
            'text': markdown_content,
            'page_number': 1,
            'section': 'Other',
            'heading': '',
            'element_type': 'text'
        }]

    def _get_context_window(self, element_index: int, window_size: int = 2) -> str:
        """Get surrounding context for better chunk understanding."""
        context_parts = []
        
        # Get preceding context
        start_idx = max(0, element_index - window_size)
        for i in range(start_idx, element_index):
            if i < len(self.raw_text_elements):
                context_parts.append(self.raw_text_elements[i]['text'][:100])
        
        # Get following context
        end_idx = min(len(self.raw_text_elements), element_index + window_size + 1)
        for i in range(element_index + 1, end_idx):
            if i < len(self.raw_text_elements):
                context_parts.append(self.raw_text_elements[i]['text'][:100])
        
        return " ... ".join(context_parts)

    def chunk_text(self, text_elements: List[Dict]) -> List[ResearchChunk]:
        """Enhanced chunking with better context preservation."""
        chunks = []
        
        for idx, element in enumerate(text_elements):
            text = element['text']
            page_num = element['page_number']
            section = element['section']
            heading = element.get('heading', '')
            element_type = element.get('element_type', 'text')
            
            if len(text) <= self.chunk_size:
                # For smaller texts, keep as single chunk with context
                context_window = self._get_context_window(idx)
                chunks.append(ResearchChunk(
                    text=text,
                    page_number=page_num,
                    section=section,
                    heading=heading,
                    element_type=element_type,
                    context_window=context_window
                ))
            else:
                # For larger texts, use sliding window approach
                try:
                    sentences = nltk.sent_tokenize(text)
                except Exception as e:
                    # Fallback to simple splitting if NLTK fails
                    self.logger.warning(f"NLTK tokenization failed, using simple splitting: {e}")
                    sentences = text.split('. ')
                    sentences = [s.strip() + '.' for s in sentences if s.strip()]
                
                current_chunk = ""
                sentence_buffer = []
                
                for sentence in sentences:
                    sentence_buffer.append(sentence)
                    potential_chunk = " ".join(sentence_buffer)
                    
                    if len(potential_chunk) > self.chunk_size:
                        if current_chunk:
                            context_window = self._get_context_window(idx)
                            chunks.append(ResearchChunk(
                                text=current_chunk,
                                page_number=page_num,
                                section=section,
                                heading=heading,
                                element_type=element_type,
                                context_window=context_window
                            ))
                        
                        # Create overlap
                        overlap_sentences = sentence_buffer[-(self.overlap // 50):] if len(sentence_buffer) > 2 else sentence_buffer
                        current_chunk = " ".join(overlap_sentences)
                        sentence_buffer = overlap_sentences
                    else:
                        current_chunk = potential_chunk
                
                # Add remaining chunk
                if current_chunk.strip():
                    context_window = self._get_context_window(idx)
                    chunks.append(ResearchChunk(
                        text=current_chunk,
                        page_number=page_num,
                        section=section,
                        heading=heading,
                        element_type=element_type,
                        context_window=context_window
                    ))
        
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, chunks: List[ResearchChunk]) -> np.ndarray:
        """Create embeddings with enhanced context."""
        self.logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        
        texts = []
        for chunk in chunks:
            # Create rich context for embedding
            context_parts = []
            
            # Add section and heading context
            if chunk.section != "Other":
                context_parts.append(f"Section: {chunk.section}")
            if chunk.heading:
                context_parts.append(f"Heading: {chunk.heading}")
            
            # Add element type if meaningful
            if chunk.element_type and chunk.element_type not in ['text', 'paragraph']:
                context_parts.append(f"Type: {chunk.element_type}")
            
            # Combine context with main text
            if context_parts:
                context_prefix = " | ".join(context_parts) + ": "
            else:
                context_prefix = ""
            
            # Include context window for better understanding
            enhanced_text = context_prefix + chunk.text
            if chunk.context_window:
                enhanced_text += f" [Context: {chunk.context_window[:200]}]"
            
            texts.append(enhanced_text)
        
        # Create embeddings in batches
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
        
        # Assign embeddings to chunks
        for i, chunk in enumerate(chunks):
            chunk.embedding = embeddings[i]
        
        return embeddings

    def build_index(self, embeddings: np.ndarray):
        """Build FAISS index for similarity search."""
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        self.logger.info(f"Built index with {self.index.ntotal} vectors")

    def load_research_paper(self, source: str):
        """Load research paper with enhanced processing."""
        self.logger.info(f"Loading research paper: {source}")
        
        if not source.startswith('http') and not os.path.exists(source):
            raise FileNotFoundError(f"PDF file not found: {source}")
        
        # Extract text and metadata
        text_elements = self.extract_metadata_and_text_with_docling(source)
        
        # Create chunks
        self.chunks = self.chunk_text(text_elements)
        
        # Create embeddings and build index
        embeddings = self.create_embeddings(self.chunks)
        self.build_index(embeddings)
        
        self.logger.info(f"Loaded research paper with {len(self.chunks)} chunks")
        self.logger.info(f"Paper metadata: {self.paper_metadata}")

    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[ResearchChunk, float]]:
        """Retrieve relevant chunks with improved filtering."""
        if not self.index:
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
        
        query_embedding = query_embedding.cpu().numpy().astype('float32')
        faiss.normalize_L2(query_embedding)
        
        # Search for similar chunks
        similarities, indices = self.index.search(query_embedding, top_k * 3)
        
        # Filter and rank results
        results = []
        seen_texts = set()
        
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                
                # Skip duplicate or uninformative chunks
                if (chunk.text not in seen_texts and 
                    len(chunk.text) > 50 and
                    not chunk.text.lower().startswith('page ') and
                    'publication date' not in chunk.text.lower()):
                    
                    results.append((chunk, sim))
                    seen_texts.add(chunk.text)
                    
                    if len(results) >= top_k:
                        break
        
        return results

    def generate_response(self, query: str, context_chunks: List[ResearchChunk]) -> str:
        """Generate enhanced response with better context utilization."""
        if not context_chunks:
            return "No relevant information found in the document."
        
        # Organize chunks by section for better structure
        sections = {}
        for chunk in context_chunks:
            if chunk.section not in sections:
                sections[chunk.section] = []
            sections[chunk.section].append(chunk)
        
        # Build structured context
        context_parts = []
        for section, chunks in sections.items():
            section_text = f"\n=== {section} ===\n"
            for chunk in chunks:
                chunk_info = f"[Page {chunk.page_number}"
                if chunk.heading:
                    chunk_info += f", {chunk.heading}"
                chunk_info += "]"
                section_text += f"{chunk_info}\n{chunk.text}\n\n"
            context_parts.append(section_text)
        
        context = "\n".join(context_parts)
        
        prompt = f"""You are an expert research assistant. Analyze the provided academic paper content and answer the question comprehensively.

PAPER INFORMATION:
Title: {self.paper_metadata.get('title', 'Unknown')}
Authors: {self.paper_metadata.get('authors', 'Unknown')}
Abstract: {self.paper_metadata.get('abstract', 'Not provided')}

RELEVANT CONTENT:
{context}

QUESTION: {query}

INSTRUCTIONS:
1. Provide a comprehensive, well-structured answer based on the content above
2. Use specific information from the paper, citing page numbers and sections
3. Organize your response with clear headings and bullet points where appropriate
4. If the information is incomplete, clearly state what aspects are missing
5. Focus on the most relevant and substantive information
6. Maintain academic tone and precision

ANSWER:"""

        try:
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            self.logger.error(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"

    def query(self, question: str, top_k: int = 5) -> Dict:
        """Enhanced query method with better result formatting and JSON serialization fix."""
        if not question.strip():
            return {
                'answer': "Please provide a valid question.",
                'sources': [],
                'confidence': 0.0,
                'metadata': self.paper_metadata,
                'evaluation': {}
            }
        
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve(question, top_k)
        
        if not relevant_chunks:
            return {
                'answer': "No relevant information found in the paper.",
                'sources': [],
                'confidence': 0.0,
                'metadata': self.paper_metadata,
                'evaluation': {}
            }
        
        # Generate response
        answer = self.generate_response(question, [chunk for chunk, _ in relevant_chunks])
        
        # Calculate confidence - FIXED: Convert to Python float
        avg_similarity = sum(float(sim) for _, sim in relevant_chunks) / len(relevant_chunks)
        confidence = min(avg_similarity * 100, 95.0)  # Cap at 95%
        
        # Prepare enhanced source information
        sources = []
        for i, (chunk, sim) in enumerate(relevant_chunks):
            source_info = {
                'rank': i + 1,
                'section': chunk.section,
                'page': int(chunk.page_number),  # Ensure it's a Python int
                'heading': chunk.heading if chunk.heading else 'No heading',
                'text_preview': chunk.text[:300] + "..." if len(chunk.text) > 300 else chunk.text,
                'relevance_score': float(sim),  # FIXED: Convert numpy float32 to Python float
                'element_type': chunk.element_type
            }
            sources.append(source_info)
        
        # Prepare the result dictionary and ensure all values are JSON serializable
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': float(confidence),  # FIXED: Ensure it's a Python float
            'metadata': self.paper_metadata,
            'total_chunks': len(self.chunks),
            'retrieved_chunks': len(relevant_chunks)
        }
        
        # Apply the conversion function to ensure everything is JSON serializable
        return convert_to_json_serializable(result)