"""
Unified Question Generation App
Combines PDF processing, question generation, and IRT theta calculation
"""

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import re
import logging
import random
import shutil
import numpy as np
from scipy.optimize import minimize
from typing import List, Dict, Any

# LangChain imports
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# PDF processing imports
from PyPDF2 import PdfReader

app = FastAPI(title="Unified Question Generation System", version="1.0.0")

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("unified_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API key configuration
os.environ["GOOGLE_API_KEY"] = ""  
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Configuration
VECTOR_DB_FOLDER = 'static/vector_db'
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

# IRT Parameters
IRT_A = 1.0
IRT_C = 0.25  # 4 choices

# ==================== PDF PROCESSING FUNCTIONS ====================

def count_pdf_pages(pdf_path: str) -> int:
    """Count the number of pages in a PDF file."""
    try:
        pdf = PdfReader(pdf_path)
        return len(pdf.pages)
    except Exception as e:
        logger.error(f"Error counting PDF pages: {e}")
        return 0

def split_pdf_to_documents(file_path: str, chapter_title: str = "Default Chapter") -> List[Document]:
    """Read PDF and split into documents without difficulty metadata."""
    try:
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        text = "".join([p.page_content for p in pages])

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)

        docs = []
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "chapter": chapter_title
                }
            )
            docs.append(doc)

        logger.info(f"Split PDF into {len(docs)} chunks")
        return docs
    except Exception as e:
        logger.error(f"Error splitting PDF: {e}")
        return []

def save_to_faiss(documents: List[Document], index_path: str) -> bool:
    """Save documents to FAISS vector store."""
    try:
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
            logger.info(f"Existing vector DB at {index_path} has been removed.")
        
        os.makedirs(index_path, exist_ok=True)
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
        logger.info(f"✅ Vector DB saved to: {index_path}")
        return True
    except Exception as e:
        logger.error(f"❌ Error saving Vector DB: {e}")
        return False

def process_pdf_and_create_vectordb(pdf_path: str, chapter_title: str = "Default Chapter") -> tuple:
    """Process PDF and create vector database."""
    pdf_filename = os.path.basename(pdf_path)
    vector_db_path = os.path.join(VECTOR_DB_FOLDER, pdf_filename.replace('.pdf', ''))

    logger.info(f"📄 Reading and splitting PDF: {pdf_path}")
    documents = split_pdf_to_documents(pdf_path, chapter_title)

    if not documents:
        return None, "Failed to process PDF documents"

    logger.info(f"📊 Total chunks: {len(documents)}")
    logger.info("💾 Saving to vector DB...")
    success = save_to_faiss(documents, vector_db_path)

    if success:
        return vector_db_path, None
    else:
        return None, "Failed to save to vector database"

# ==================== QUESTION GENERATION FUNCTIONS ====================

def load_from_faiss(index_path: str):
    """Load FAISS vector store from path."""
    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector DB not found at {index_path}")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"✅ Vector DB loaded from: {index_path}")
        return vector_store
    except Exception as e:
        logger.error(f"❌ Error loading Vector DB: {e}")
        return None

def retrieve_random_contexts(vector_store, n_results: int = 40) -> List[str]:
    """Retrieve random contexts from vector DB."""
    try:
        all_docs = list(vector_store.docstore._dict.values())
        if len(all_docs) <= n_results:
            selected_docs = all_docs
        else:
            selected_docs = random.sample(all_docs, n_results)
        contexts = [doc.page_content for doc in selected_docs]
        logger.info(f"Number of contexts retrieved: {len(contexts)}")
        return contexts
    except Exception as e:
        logger.error(f"Error retrieving context: {e}")
        return []

def extract_json_from_text(text: str) -> str:
    """Extract JSON array from LLM response text."""
    try:
        json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        logger.warning(f"JSON pattern not found in text: {text[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return None

def generate_questions_from_context(contexts: List[str], target_difficulty: int, num_questions: int, max_attempts: int = 20) -> List[Dict]:
    """Generate questions from context using LLM."""
    try:
        llm = GoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3
        )
        context_text = "\n\n".join(contexts)
        prompt_template = """
        You are an expert in designing high-quality multiple choice questions (MCQs) for learning assessment.

        Your task is to create {num_questions} objective and conceptual multiple choice questions based on the provided text.
        These questions should assess understanding of key ideas, facts, concepts, or logic **derived from** the text - 
        but they should NOT rely on literal phrases like "According to the text".

        Here is the source material:
        ------------
        {text}
        ------------

        Instructions:
        - Difficulty level: {difficulty}/5 (1 = easiest, 5 = most difficult)
        - All questions MUST have difficulty exactly {difficulty}/5.
        - Each question should test **conceptual understanding or factual knowledge**.
        - Avoid using phrases like "According to the passage" or "As stated above".
        - Instead, write **standalone, self-contained questions** that make sense without referring back to the source.
        - Avoid trivial questions or questions that require rote copying.
        - Use paraphrasing when drawing ideas from the source.

        Each question must be in this JSON format:
        {{
            "question": "<clear and self-contained question>",
            "difficulty": <1 to 5>,
            "choices": ["<choice A>", "<choice B>", "<choice C>", "<choice D>"],
            "correct_answer": "<the exact text of the correct choice>",
            "explanation": "<why this is correct, and why the other 3 are wrong>"
        }}

        Rules:
        1. Each question must have exactly 4 distinct choices.
        2. The correct_answer must exactly match one of the choices.
        3. All questions must have "difficulty": {difficulty}.
        4. All questions must be based on the content above - do not use outside knowledge.
        5. The explanation should be concise but informative: clearly justify the answer and refute distractors.

        I need exactly {num_questions} questions, no more and no less.

        Return only a **valid JSON array** of question objects, nothing else.
        """

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["text", "difficulty", "num_questions"]
        )

        collected_questions = []
        attempts = 0
        seen_questions = set()

        while len(collected_questions) < num_questions and attempts < max_attempts:
            remaining = num_questions - len(collected_questions)
            logger.info(f"Attempt {attempts+1}: Requesting {remaining} questions at difficulty {target_difficulty}")
            formatted_prompt = PROMPT.format(
                text=context_text,
                difficulty=target_difficulty,
                num_questions=remaining
            )
            response = llm.invoke(formatted_prompt)
            json_str = extract_json_from_text(response)
            if not json_str:
                logger.warning("No valid JSON extracted from LLM response, retrying...")
                attempts += 1
                continue
            try:
                questions = json.loads(json_str)
            except json.JSONDecodeError:
                logger.warning("JSON decode error, retrying...")
                attempts += 1
                continue

            # Filter questions with correct difficulty
            filtered = [q for q in questions if q.get('difficulty', 3) == target_difficulty]

            # Remove duplicates
            new_unique = []
            for q in filtered:
                q_text = q.get('question', '').strip()
                if q_text and q_text not in seen_questions:
                    seen_questions.add(q_text)
                    new_unique.append(q)
            collected_questions.extend(new_unique)
            logger.info(f"Collected {len(collected_questions)} / {num_questions} questions so far")
            attempts += 1

        if len(collected_questions) < num_questions:
            logger.warning(f"Stopped after {attempts} attempts with only {len(collected_questions)} questions collected")

        return collected_questions[:num_questions]
    except Exception as e:
        logger.error(f"Error generating questions: {e}")
        return []

# ==================== IRT CALCULATION FUNCTIONS ====================

def difficulty_to_b(level: int) -> float:
    """Convert difficulty level to IRT b parameter."""
    mapping = {
        1: -2.5,   # Very easy
        2: -1.25,  # Easy
        3: 0.0,    # Medium
        4: 1.25,   # Hard
        5: 2.5     # Very hard
    }
    return mapping.get(level, 0.0)

def theta_to_level(theta: float) -> int:
    """Convert theta to difficulty level."""
    if theta < -2.0:
        return 1  # Very easy
    elif theta < -0.5:
        return 2  # Easy
    elif theta < 0.5:
        return 3  # Medium
    elif theta < 2.0:
        return 4  # Hard
    else:
        return 5  # Very hard

def P_theta(theta: float, a: float, b: float, c: float) -> float:
    """Calculate probability of correct answer using 3PL IRT model."""
    exp_term = np.exp(-1.7 * a * (theta - b))
    return c + (1 - c) / (1 + exp_term)

def log_posterior(theta: float, a_list: np.ndarray, b_list: np.ndarray, c_list: np.ndarray, u_list: np.ndarray) -> float:
    """Log-posterior function for MAP estimation."""
    exp_term = np.exp(-1.7 * a_list * (theta - b_list))
    p_list = c_list + (1 - c_list) / (1 + exp_term)
    p_list = np.clip(p_list, 1e-6, 1 - 1e-6)
    ll = np.sum(u_list * np.log(p_list) + (1 - u_list) * np.log(1 - p_list))
    lp = -0.5 * theta ** 2
    return -(ll + lp)  # Negative for minimization

def estimate_theta_map(questions: List[Dict[str, Any]]) -> float:
    """Estimate theta using MAP (Maximum A Posteriori)."""
    a_list = np.array([IRT_A for _ in questions])
    b_list = np.array([difficulty_to_b(q['difficulty']) for q in questions])
    c_list = np.array([IRT_C for _ in questions])
    u_list = np.array([1 if q['isCorrect'] else 0 for q in questions])
    
    res = minimize(
        log_posterior,
        x0=0.0,
        args=(a_list, b_list, c_list, u_list),
        bounds=[(-3, 3)],
        method='L-BFGS-B'
    )
    return res.x[0]

# ==================== API ENDPOINTS ====================

@app.post("/process")
async def process_pdf_endpoint(
    pdf_filename: str = Form(...), 
    chapter_title: str = Form("Default Chapter")
):
    """Process PDF and create vector database."""
    if not os.path.exists(pdf_filename):
        raise HTTPException(status_code=404, detail=f"PDF file not found: {pdf_filename}")

    vector_db_path, error = process_pdf_and_create_vectordb(pdf_filename, chapter_title)

    if error:
        raise HTTPException(status_code=500, detail=error)

    page_count = count_pdf_pages(pdf_filename)
    
    return {
        "message": "Vector DB created successfully",
        "vector_db_path": vector_db_path,
        "pdf_filename": pdf_filename,
        "chapter_title": chapter_title,
        "page_count": page_count
    }

@app.post("/generate")
async def generate_questions_endpoint(
    vector_db_path: str = Form(...),
    difficulty: int = Form(...),
    num_questions: int = Form(10)
):
    """Generate questions from vector database."""
    logger.info(f"Received request with: vector_db_path={vector_db_path}, difficulty={difficulty}, num_questions={num_questions}")
    
    try:
        num_questions = int(num_questions)
        logger.info(f"Parsed num_questions to int: {num_questions}")
    except ValueError:
        logger.error(f"Invalid num_questions value: {num_questions}")
        raise HTTPException(status_code=400, detail="num_questions must be an integer")
    
    if num_questions < 1:
        logger.warning(f"Requested fewer than 1 question: {num_questions}, setting to 1")
        num_questions = 1
    elif num_questions > 40:
        logger.warning(f"Requested too many questions: {num_questions}, limiting to 40")
        num_questions = 40

    if difficulty < 1 or difficulty > 5:
        raise HTTPException(status_code=400, detail="Difficulty must be between 1 and 5")

    vector_store = load_from_faiss(vector_db_path)
    if not vector_store:
        logger.error(f"Vector database not found at path: {vector_db_path}")
        raise HTTPException(status_code=404, detail="Vector database not found.")

    contexts = retrieve_random_contexts(vector_store, n_results=40)
    if not contexts:
        logger.error(f"No context found in vector DB")
        raise HTTPException(status_code=404, detail="No suitable context found.")

    questions = generate_questions_from_context(contexts, difficulty, num_questions, max_attempts=20)
    logger.info(f"Generated {len(questions)} questions, requested {num_questions}")

    if len(questions) < num_questions:
        logger.warning(f"Only {len(questions)} questions with difficulty {difficulty} found. Requested {num_questions}.")

    return JSONResponse(content=questions)

@app.post("/calculate_theta")
async def calculate_theta_endpoint(request: Request):
    """Calculate theta (ability level) based on question responses."""
    try:
        data = await request.json()
        logger.info(f"Received theta calculation request with {len(data)} questions")
        
        # Validate input data
        for i, question in enumerate(data):
            if 'difficulty' not in question or 'isCorrect' not in question:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Question {i} missing required fields 'difficulty' or 'isCorrect'"
                )
            if not isinstance(question['difficulty'], int) or question['difficulty'] < 1 or question['difficulty'] > 5:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Question {i} has invalid difficulty: {question['difficulty']}"
                )
            if not isinstance(question['isCorrect'], bool):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Question {i} has invalid isCorrect value: {question['isCorrect']}"
                )
        
        theta = estimate_theta_map(data)
        level = theta_to_level(theta)
        
        logger.info(f"Result: theta = {theta:.4f}, level = {level}")
        
        return JSONResponse(content={
            'theta': float(theta), 
            'level': int(level),
            'questions_analyzed': len(data)
        })
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON in request body")
    except Exception as e:
        logger.error(f"Error calculating theta: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Unified Question Generation System",
        "version": "1.0.0",
        "endpoints": {
            "process_pdf": "POST /process_pdf - Process PDF and create vector database",
            "generate_questions": "POST /generate_questions - Generate questions from vector database", 
            "calculate_theta": "POST /calculate_theta - Calculate ability level from question responses"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "vector_db_folder": VECTOR_DB_FOLDER}

if __name__ == "__main__":
    uvicorn.run(
        "app:app", 
        host='0.0.0.0', 
        port=8003, 
        reload=True,
        log_level="info"
    )