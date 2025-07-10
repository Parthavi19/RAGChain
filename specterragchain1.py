import os
import google.generativeai as genai
import numpy as np
import faiss
import PyPDF2
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from transformers import AutoTokenizer, AutoModel
import torch
import time
import google.api_core.exceptions

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@dataclass
class ResearchChunk:
    text: str
    page_number: int
    section: str
    embedding: np.ndarray = None

class ResearchPaperRAG:
    def __init__(self, api_key: str, chunk_size: int = 600, overlap: int = 50):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            'gemini-1.5-flash',
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                top_p=0.8,
                top_k=30,
            )
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

        self.chunks: List[ResearchChunk] = []
        self.index = None
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.paper_metadata = {}
        self.api_quota_exceeded = False

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
        text = re.sub(r'Page\s+\d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s+([,.!?;:])', r'\1', text)
        return text.strip()

    def tokenize_sentences(self, text: str) -> List[str]:
        """
        Simple regex-based sentence tokenization to replace NLTK's sent_tokenize.
        Handles most common sentence endings and abbreviations.
        """
        # Common abbreviations that shouldn't end sentences
        abbreviations = [
            'Mr', 'Mrs', 'Ms', 'Dr', 'Prof', 'Sr', 'Jr', 'vs', 'etc', 'Fig', 'fig', 
            'Table', 'table', 'al', 'e.g', 'i.e', 'cf', 'viz', 'Inc', 'Corp', 'Ltd', 
            'Co', 'LLC', 'Ph.D', 'M.D', 'B.A', 'M.A', 'B.S', 'M.S', 'CEO', 'CFO', 
            'CTO', 'USA', 'UK', 'EU', 'UN', 'NASA', 'IEEE', 'ACM', 'MIT', 'UCLA', 
            'USC', 'NYU', 'IBM', 'HP', 'AI', 'ML', 'NLP', 'CV', 'GPU', 'CPU', 'API',
            'URL', 'HTTP', 'HTTPS', 'XML', 'JSON', 'PDF', 'HTML', 'CSS', 'JS', 'SQL',
            'DB', 'OS', 'UI', 'UX', '3D', '2D', 'RGB', 'CMYK', 'JPEG', 'PNG', 'GIF',
            'SVG', 'MP3', 'MP4', 'AVI', 'MOV', 'ZIP', 'RAR', 'TAR', 'GZ', 'TXT',
            'DOC', 'DOCX', 'XLS', 'XLSX', 'PPT', 'PPTX', 'CSV', 'TSV', 'YAML',
            'TOML', 'INI', 'CFG', 'LOG', 'BAK', 'TMP', 'TEMP', 'SRC', 'BIN', 'LIB',
            'DLL', 'EXE', 'APP', 'DMG', 'ISO', 'IMG', 'VHD', 'VMDK', 'OVA', 'OVF'
        ]
        
        # Create a set for faster lookup
        abbrev_set = set(abbreviations)
        
        # Simple sentence splitting on periods, exclamation marks, and question marks
        # followed by whitespace and capital letters
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
        
        # Post-process to handle abbreviations
        final_sentences = []
        current_sentence = ""
        
        for sentence in sentences:
            if current_sentence:
                current_sentence += " " + sentence
            else:
                current_sentence = sentence
                
            # Check if this sentence ends with an abbreviation
            words = current_sentence.strip().split()
            if words:
                last_word = words[-1].rstrip('.!?')
                if last_word not in abbrev_set:
                    # This looks like a real sentence end
                    final_sentences.append(current_sentence.strip())
                    current_sentence = ""
        
        # Add any remaining sentence
        if current_sentence.strip():
            final_sentences.append(current_sentence.strip())
        
        # Clean up sentences and filter out empty ones
        cleaned_sentences = []
        for sentence in final_sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:  # Filter out very short fragments
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences

    def extract_metadata_and_text(self, pdf_path: str) -> List[Dict]:
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
        chunks = []
        chunk_id = 0
        for page_data in pages_text:
            text = page_data['text']
            page_num = page_data['page_number']
            section = page_data['section']
            if not text.strip():
                continue
                
            # Use regex-based sentence tokenization instead of NLTK
            sentences = self.tokenize_sentences(text)
            current_chunk = ""
            
            for sentence in sentences:
                potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
                if len(potential_chunk) <= self.chunk_size:
                    current_chunk = potential_chunk
                else:
                    if current_chunk.strip():
                        chunks.append(ResearchChunk(
                            text=current_chunk.strip(),
                            page_number=page_num,
                            section=section
                        ))
                        chunk_id += 1
                    current_chunk = sentence[-self.overlap:] + " " + sentence if len(sentence) > self.overlap else sentence
            
            if current_chunk.strip():
                chunks.append(ResearchChunk(
                    text=current_chunk.strip(),
                    page_number=page_num,
                    section=section
                ))
                chunk_id += 1
        
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks

    def create_embeddings(self, chunks: List[ResearchChunk]) -> np.ndarray:
        self.logger.info(f"Creating embeddings for {len(chunks)} chunks...")
        texts = [f"{chunk.section}: {chunk.text}" for chunk in chunks]
        embeddings = []

        batch_size = 8  # Reduced batch size for faster CPU processing
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

    def build_index(self, embeddings: np.ndarray):
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings.astype('float32'))
        self.index.add(embeddings.astype('float32'))
        self.logger.info(f"Built index with {self.index.ntotal} vectors")

    def extract_metadata(self, pages_text: List[Dict]):
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
        self.paper_metadata = {'title': title, 'authors': authors, 'abstract': abstract}

    def load_research_paper(self, pdf_path: str):
        if not os.path.exists(pdf_path):
            self.logger.error(f"PDF file not found: {pdf_path}")
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        self.logger.info(f"Loading research paper: {pdf_path}")
        pages_text = self.extract_metadata_and_text(pdf_path)
        self.extract_metadata(pages_text)
        self.chunks = self.chunk_text(pages_text)
        embeddings = self.create_embeddings(self.chunks)
        self.build_index(embeddings)
        self.logger.info(f"Loaded research paper with {len(self.chunks)} chunks")

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[ResearchChunk, float]]:
        if not self.index:
            self.logger.error("No document loaded")
            raise ValueError("No document loaded")
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
        similarities, indices = self.index.search(query_embedding, top_k * 2)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                chunk = self.chunks[idx]
                if "publication date" not in chunk.text.lower():
                    results.append((chunk, sim))
            if len(results) >= top_k:
                break
        self.logger.debug(f"Retrieved {len(results)} chunks for query: {query}")
        return results

    def generate_response(self, query: str, context_chunks: List[ResearchChunk]) -> str:
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

    def query(self, question: str, top_k: int = 10) -> Dict:
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
        
        result = {
            'answer': answer,
            'sources': sources,
            'confidence': float(round(confidence, 1)),
            'metadata': self.paper_metadata
        }
        
        self.logger.info(f"Query result: {result}")
        return result