from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import os
import json
import re
import logging
import random

from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

app = FastAPI()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_debug.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# API key configuration
os.environ["GOOGLE_API_KEY"] = ""  
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

VECTOR_DB_FOLDER = 'static/vector_db'
os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)

def load_from_faiss(index_path):
    try:
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"Vector DB not found at {index_path}")
        vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        logger.info(f"✅ Vector DB loaded from: {index_path}")
        return vector_store
    except Exception as e:
        logger.error(f"❌ Error loading Vector DB: {e}")
        return None

def retrieve_random_contexts(vector_store, n_results=40):
    """Lấy ngẫu nhiên n_results context từ vector DB (không quan tâm độ khó)."""
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

def extract_json_from_text(text):
    try:
        json_match = re.search(r'\[\s*\{.*\}\s*\]', text, re.DOTALL)
        if json_match:
            return json_match.group(0)
        logger.warning(f"JSON pattern not found in text: {text[:200]}...")
        return None
    except Exception as e:
        logger.error(f"Error extracting JSON: {e}")
        return None

def generate_questions_from_context(contexts, target_difficulty, num_questions, max_attempts=20):
    """
    Gọi LLM nhiều lần, gom câu hỏi đúng độ khó, đến khi đủ số lượng hoặc đạt max_attempts.
    Tất cả câu hỏi phải có cùng 1 mức độ khó.
    """
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

            # Lọc câu hỏi đúng độ khó
            filtered = [q for q in questions if q.get('difficulty', 3) == target_difficulty]

            # Loại bỏ trùng lặp
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

@app.post("/generate")
async def generate_questions(
    vector_db_path: str = Form(...),
    difficulty: int = Form(...),
    num_questions: int = Form(10)
):
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

if __name__ == "__main__":
    uvicorn.run("gen_question:app", host='0.0.0.0', port=8002, reload=True)
