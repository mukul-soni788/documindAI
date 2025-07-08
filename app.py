import os
import logging
import json
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for, redirect
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from werkzeug.utils import secure_filename
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from art import text2art
from pdfminer.high_level import extract_text as extract_pdf_text
from docx import Document
import pandas as pd
from deep_translator import GoogleTranslator
import pytesseract
import psycopg2
from psycopg2.extras import RealDictCursor
from werkzeug.security import generate_password_hash, check_password_hash
import uuid
from authlib.integrations.flask_client import OAuth
from urllib.parse import urlparse, urljoin
import smtplib
from email.message import EmailMessage
import secrets
from datetime import datetime, timedelta
from functools import wraps
import razorpay
import hmac
import hashlib


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyBjLGqvwA1rbGFkB1IrgCPEWtR0A9y61kU"
os.environ["OPENROUTER_API_KEY"] = "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"

app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-default-dev-secret-key')


# Initialize Google Gemini model
logger.info("Initializing Google Gemini model...")
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ["GOOGLE_API_KEY"],
    temperature=0.7,
    # streaming=True
    disable_streaming = False
)
logger.info("Model initialized.")

# Prompt and chain setup
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful, concise, and intelligent AI assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])
chain = prompt | llm

# Memory for user sessions
store = {}
store.clear()
logger.info("Chat history cleared on server startup.")

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

runnable_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history"
)

# ASCII art generator
def generate_ascii_art(prompt: str) -> str:
    try:
        ascii_art = text2art(prompt, font="standard")
        return f"```\n{ascii_art}\n```"
    except Exception as e:
        logger.error(f"ASCII error: {str(e)}")
        return f"Error generating ASCII art: {str(e)}"

# Pydantic model (for reference, not used directly in Flask)
class AnalysisResponse(BaseModel):
    summary: str
    insights: list

def extract_text_from_docx(file) -> str:
    doc = Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_csv(file) -> str:
    try:
        df = pd.read_csv(file)
        df_clean = df.dropna(how='all')  # Drop rows where all elements are NaN
        return df_clean.astype(str).to_string(index=False)
    except Exception as e:
        logger.error(f"Error reading CSV file: {str(e)}")
        return ""


def extract_text_from_excel(file) -> str:
    try:
        xl = pd.read_excel(file, sheet_name=None)  # Read all sheets
        all_sheets_text = []

        for sheet_name, df in xl.items():
            df_clean = df.dropna(how='all')  # Drop rows where all elements are NaN
            sheet_text = f"--- Sheet: {sheet_name} ---\n{df_clean.astype(str).to_string(index=False)}\n"
            all_sheets_text.append(sheet_text)

        return "\n".join(all_sheets_text)
    except Exception as e:
        logger.error(f"Error reading Excel file: {str(e)}")
        return ""



def check_token_status(view_func):
    """
    Flask decorator to check if the Authorization token is valid and active using raw SQL.
    Usage: @check_token_status
    """
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        token = request.headers.get('Authorization') or request.cookies.get('token')
        if not token:
            return render_template('auth.html')
        try:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute('SELECT user_id, active_status, expired_at FROM login_info WHERE token=%s', (token,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            if not row:
                return render_template('auth.html')
            user_id, active_status, expired_at = row
            if active_status != 1:
                return render_template('auth.html')
            if expired_at and datetime.utcnow() > expired_at:
                return render_template('auth.html')
            # Optionally, set user_id in g for downstream use
            return view_func(*args, **kwargs)
        except Exception as e:
            return render_template('auth.html')
    return wrapper


# Routes
@app.route("/chat/")
@check_token_status
def get_home():
    return render_template("index.html")

@app.route("/portfolio")
def portfolio():
    return render_template("test.html")

@app.route("/")
def base():
    return render_template("base.html")

@app.route("/login")
def login():
    return render_template("auth.html")

@app.route("/document", endpoint="document")
@check_token_status
def get_document():
    return render_template("document.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
@check_token_status
def contact():
    return render_template('contact.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route('/profile')
@check_token_status
def profile():
    return render_template('profile.html')

@app.route("/analyze", methods=["POST"])
@check_token_status
def analyze_document():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400
    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.pdf'):
            file.stream.seek(0)
            content = extract_pdf_text(file.stream)
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(file.stream)
        elif filename.endswith('.txt'):
            content = file.stream.read().decode('utf-8')
        elif filename.endswith('.csv'):
            content = extract_text_from_csv(file.stream)
        elif filename.endswith(('.xls', '.xlsx')):
            content = extract_text_from_excel(file.stream)
        else:
            return jsonify({"detail": "Only PDF, DOCX, TXT, CSV, XLS, XLSX files are supported."}), 400

        cleaned = content
        print(cleaned)
        if not cleaned.strip():
            return jsonify({"detail": "Empty or unreadable file content."}), 400

        logger.info(f"Extracted PDF text preview:\n{cleaned[:500]}")

        prompt_text = f"""
        Analyze the following document text and provide:
        1. A concise summary (150-250 words) in a single paragraph.
        2. 5-7 key insights or important points, each as a concise sentence.

        Document text:
        {cleaned}

        Respond strictly in this JSON format:
        {{
            "summary": "summary text",
            "insights": ["insight 1", "insight 2", "insight 3", ...],
        }}
        Do not wrap the response in markdown code blocks (e.g., ```json). Do not use markdown syntax (e.g., **, ###) in the response. Ensure exactly 3-5 insights.
        """

        response = llm.invoke(prompt_text)
        result = response.content.strip()
        print(result)

        # Remove possible code fences
        if result.startswith("```json"):
            result = result.replace("```json", "", 1).replace("```", "", 1).strip()
        elif result.startswith("```"):
            result = result.replace("```", "", 2).strip()

        try:
            data = json.loads(result)
            summary = data.get("summary", "")
            insights = data.get("insights", [])
        except json.JSONDecodeError as e:
            logger.warning(f"Model did not return valid JSON: {str(e)}")
            return jsonify({"detail": "AI returned invalid format"}), 500

        return jsonify({"summary": summary.strip(), "insights": insights})

    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        return jsonify({"detail": f"Error processing document: {str(e)}"}), 500


@app.route("/resume-match", endpoint="resume_match")
@check_token_status
def resume_match_page():
    return render_template("resume_match.html")

@app.route("/analyze-resume", methods=["POST"])
@check_token_status
def analyze_resume():
    try:
        # Get resume
        resume_file = request.files.get("resume")
        if not resume_file or resume_file.filename == '':
            return jsonify({"detail": "No resume uploaded"}), 400

        if resume_file.filename.endswith('.pdf'):
            resume_file.stream.seek(0)
            resume_text = extract_pdf_text(resume_file.stream)
        else:
            return jsonify({"detail": "Only PDF resumes are supported."}), 400

        # Get job description (text or file)
        job_desc_text = request.form.get("job_description", "")
        job_desc_file = request.files.get("job_description_file")

        if not job_desc_text and job_desc_file:
            if job_desc_file.filename.endswith(".docx"):
                job_desc_text = extract_text_from_docx(job_desc_file.stream)
            elif job_desc_file.filename.endswith(".txt"):
                job_desc_text = job_desc_file.stream.read().decode("utf-8")
            elif job_desc_file.filename.endswith(".pdf"):
                job_desc_text = extract_pdf_text(job_desc_file.stream)
            else:
                return jsonify({"detail": "Unsupported job description file format."}), 400

        if not job_desc_text.strip():
            return jsonify({"detail": "Empty job description"}), 400

        # Prompt for job match analysis
        prompt_text = f"""
        You are an expert career assistant and recruiter.

        Your task is to analyze how well a given resume matches a specific job description.

        Based on your analysis, respond with the following in strictly valid JSON format:
        1. A resume-job match score between 0% to 100% (as a string, e.g., "78%").
        2. A list of **missing or weakly represented skills/keywords** that appear in the job description but not in the resume.
        3. 3 to 5 **specific suggestions** to improve the resume so it better aligns with the job description.

        Do not return any explanation or extra commentary—just the JSON.

        Here are the inputs:

        Resume:
        \"\"\"
        {resume_text}
        \"\"\"

        Job Description:
        \"\"\"
        {job_desc_text}
        \"\"\"

        Respond strictly in this JSON format (no markdown or code block):
        {{
        "match_score": "score %",
        "missing_keywords": ["keyword 1", "keyword 2"],
        "improvements": ["suggestion 1", "suggestion 2"]
        }}
        """


        response = llm.invoke(prompt_text)
        result = response.content.strip()

        # Sanitize JSON
        if result.startswith("```json"):
            result = result.replace("```json", "", 1).replace("```", "", 1).strip()
        elif result.startswith("```"):
            result = result.replace("```", "", 2).strip()

        data = json.loads(result)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Error analyzing resume: {str(e)}")
        return jsonify({"detail": f"Error: {str(e)}"}), 500

@app.route("/contract-analyzer", endpoint="contract_analyzer")
@check_token_status
def contract_analyzer():
    return render_template("contract_analyzer.html")

@app.route("/analyze_contract", methods=["POST"])
@check_token_status
def analyze_contract():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400
    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.pdf'):
            file.stream.seek(0)
            content = extract_pdf_text(file.stream)
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(file.stream)
        else:
            return jsonify({"detail": "Only PDF and DOCX files are supported."}), 400

        if not content.strip():
            return jsonify({"detail": "Empty or unreadable file content."}), 400

        prompt_text = f"""
        You are an expert legal assistant specializing in contract analysis.

        Your task is to analyze the contract text provided below and extract the following five categories of information with clarity and precision. Ensure responses are written in a professional and concise tone suitable for legal or business review.

        Extract and return the following information:

        1. **Key Clauses** – Identify and briefly summarize the most important legal clauses (e.g., termination, indemnification, confidentiality, governing law).
        2. **Party Obligations** – List the primary responsibilities and obligations of each party involved in the contract.
        3. **Risky Terms** – Highlight terms that could pose potential legal or financial risk to either party, such as automatic renewals, penalties, restrictive covenants, or one-sided indemnities.
        4. **Payment Terms** – Extract details related to payment amounts, due dates, methods, late fees, or refund policies.
        5. **Negotiation Points** – Suggest clauses or terms that the reviewing party should consider negotiating for better balance or clarity.

        ⚠️ **Output Format Instructions:**
        Respond strictly in **valid JSON format**, with no markdown or code fences. Ensure each field is an array of 2–5 bullet-point style sentences.

        Return structure:
        {{
            "key_clauses": ["..."],
            "obligations": ["..."],
            "risky_terms": ["..."],
            "payment_terms": ["..."],
            "negotiation_points": ["..."]
        }}

        Do not include any explanation or disclaimer text outside the JSON.

        Contract Text:
        \"\"\"{content}\"\"\"
        """


        response = llm.invoke(prompt_text)
        result = response.content.strip()

        # Cleanup markdown if any
        if result.startswith("```json"):
            result = result.replace("```json", "").replace("```", "").strip()

        data = json.loads(result)
        return jsonify(data)

    except Exception as e:
        logger.error(f"Error analyzing contract: {str(e)}")
        return jsonify({"detail": f"Error analyzing contract: {str(e)}"}), 500


def split_text(text, max_length=5000):
    """Splits long text into chunks below max_length."""
    paragraphs = text.split('\n')
    chunks, current_chunk = [], ""

    for para in paragraphs:
        if len(current_chunk) + len(para) + 1 < max_length:
            current_chunk += para + "\n"
        else:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n"
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


@app.route("/translate-doc", endpoint="translate_doc")
@check_token_status
def translate_doc_page():
    return render_template("translate.html")



@app.route("/translate", methods=["POST"])
@check_token_status
def translate_and_analyze():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400

    target_language = request.form.get("language", "en")
    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.pdf'):
            file.stream.seek(0)
            content = extract_pdf_text(file.stream)
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(file.stream)
        elif filename.endswith('.txt'):
            content = file.stream.read().decode('utf-8')
        else:
            return jsonify({"detail": "Only PDF, DOCX, TXT files are supported."}), 400

        if not content.strip():
            return jsonify({"detail": "Empty or unreadable file content."}), 400

        chunks = split_text(content, max_length=5000)
        translated_chunks = [
            GoogleTranslator(source='auto', target=target_language).translate(text=chunk)
            for chunk in chunks
        ]
        translated_text = "\n\n".join(translated_chunks)

        prompt_text = f'''
        Analyze the following translated document text and provide:
        1. A concise summary (150-250 words).
        2. 5 key insights.

        Text:
        {translated_text}

        Return strictly as:
        {{
            "summary": "summary text",
            "insights": ["insight 1", "insight 2", ...]
        }}
        '''

        response = llm.invoke(prompt_text)
        result = response.content.strip()

        if result.startswith("```json"):
            result = result.replace("```json", "", 1).replace("```", "", 1).strip()
        elif result.startswith("```"):
            result = result.replace("```", "", 2).strip()

        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return jsonify({"detail": "AI returned invalid format"}), 500

        return jsonify({
            "translated_text": translated_text,
            "summary": data.get("summary", ""),
            "insights": data.get("insights", [])
        })

    except Exception as e:
        logger.error(f"Translation/Analysis error: {str(e)}")
        return jsonify({"detail": f"Error: {str(e)}"}), 500



@app.route("/ebook-to-lessons", methods=["GET", "POST"], endpoint="ebook_to_lessons")
@check_token_status
def ebook_to_lessons():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"detail": "No file part"}), 400
        file = request.files['file']
        if file.filename == '':
            return jsonify({"detail": "No selected file"}), 400

        try:
            filename = secure_filename(file.filename)
            if filename.endswith('.pdf'):
                file.stream.seek(0)
                content = extract_pdf_text(file.stream)
            elif filename.endswith('.docx'):
                content = extract_text_from_docx(file.stream)
            else:
                return jsonify({"detail": "Only PDF and DOCX files supported."}), 400

            if not content.strip():
                return jsonify({"detail": "Empty or unreadable file content."}), 400

            prompt_text = f'''
            You are a skilled educational AI assistant. Your task is to analyze the following e-book or study material and generate a structured learning experience by dividing the content into clear, bite-sized micro-lessons.

            Please follow this exact format and structure:

            - Divide the content into micro-lessons.
            - For each lesson, provide:
            - A clear and engaging lesson title.
            - A concise summary (2–4 sentences).
            - One quiz question with:
                - Four multiple-choice options.
                - The correct answer clearly identified.
            - Two flashcards, each with:
                - A term.
                - Its definition based on the context of the material.

            ⛔ Do NOT return any markdown syntax or code block formatting (e.g., no triple backticks).
            ✅ Ensure the response is strictly valid JSON.

            Required Output JSON format:
            {{
            "lessons": [
                {{
                "title": "Lesson Title",
                "summary": "Brief summary of the lesson content.",
                "quiz": {{
                    "question": "A question to assess understanding",
                    "options": ["Option A", "Option B", "Option C", "Option D"],
                    "answer": "Correct option from the list above"
                }},
                "flashcards": [
                    {{
                    "term": "Term 1",
                    "definition": "Definition for Term 1"
                    }},
                    {{
                    "term": "Term 2",
                    "definition": "Definition for Term 2"
                    }}
                ]
                }}
                // ...additional lessons
            ]
            }}

            Here is the document text to analyze:
            {content}
            '''


            response = llm.invoke(prompt_text)
            result = response.content.strip()

            # Clean up markdown
            if result.startswith("```json"):
                result = result.replace("```json", "").replace("```", "").strip()

            data = json.loads(result)
            return jsonify(data)

        except Exception as e:
            logger.error(f"Error processing ebook: {str(e)}")
            return jsonify({"detail": f"Error processing file: {str(e)}"}), 500

    return render_template("ebook.html")


# Route to serve Business Report Analyzer UI
@app.route("/business-report", endpoint="business_report")
@check_token_status
def business_report():
    return render_template("business_report.html")

# API endpoint to analyze uploaded financial/marketing report
@app.route("/analyze-business-report", methods=["POST"])
@check_token_status
def analyze_business_report():
    if 'file' not in request.files:
        return jsonify({"detail": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "No selected file"}), 400

    try:
        filename = secure_filename(file.filename)

        # Reuse existing extractors
        if filename.endswith('.pdf'):
            file.stream.seek(0)
            content = extract_pdf_text(file.stream)
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(file.stream)
        elif filename.endswith('.txt'):
            content = file.stream.read().decode('utf-8')
        elif filename.endswith('.csv'):
            content = extract_text_from_csv(file.stream)
        elif filename.endswith(('.xls', '.xlsx')):
            content = extract_text_from_excel(file.stream)
        else:
            return jsonify({"detail": "Only PDF, DOCX, TXT, CSV, XLS, XLSX files are supported."}), 400

        if not content.strip():
            return jsonify({"detail": "Empty or unreadable file content."}), 400

        prompt_text = f"""
            You are a highly intelligent business analysis assistant.

            The user has uploaded a business report, which may contain financial statements, marketing insights, or operational summaries.

            Your task is to carefully read and analyze the document content, then respond strictly in the following JSON format:

            {{
                "key_indicators": ["List 3-5 key financial or business metrics or signals found in the document."],
                "swot": {{
                    "strengths": ["Identify strengths such as competitive advantages, internal capabilities, or positive trends."],
                    "weaknesses": ["Identify weaknesses such as internal gaps, inefficiencies, or declining metrics."],
                    "opportunities": ["Identify external opportunities for growth, market expansion, or innovation."],
                    "threats": ["Identify threats such as competition, market risks, or operational challenges."]
                }},
                "growth_opportunities": ["List 3-5 specific actionable areas for growth based on the document analysis."],
                "recommendations": ["Provide 3-5 strategic recommendations tailored to the content of the document."]
            }}

            Guidelines:
            - Do not include any markdown syntax (no backticks, no **bold**).
            - Do not summarize the document.
            - Your entire response must be strictly valid JSON and directly usable in code.
            - Be concise but informative in each list item.
            - Use only English.

            Here is the document text:
            {content}
            """


        response = llm.invoke(prompt_text)
        result = response.content.strip()

        if result.startswith("```json"):
            result = result.replace("```json", "", 1).replace("```", "", 1).strip()
        elif result.startswith("```"):
            result = result.replace("```", "", 2).strip()

        try:
            data = json.loads(result)
            return jsonify(data)
        except json.JSONDecodeError as e:
            return jsonify({"detail": "AI returned invalid format", "raw": result}), 500

    except Exception as e:
        return jsonify({"detail": f"Error processing document: {str(e)}"}), 500


@app.route("/transcript", endpoint="transcript")
@check_token_status
def transcript_page():
    return render_template("transcript.html")


@app.route("/analyze-transcript", methods=["POST"])
@check_token_status
def analyze_transcript():
    if 'file' not in request.files:
        return jsonify({"detail": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"detail": "Empty filename"}), 400

    try:
        filename = secure_filename(file.filename)
        if filename.endswith('.txt'):
            content = file.stream.read().decode('utf-8')
        elif filename.endswith('.docx'):
            content = extract_text_from_docx(file.stream)
        elif filename.endswith('.pdf'):
            file.stream.seek(0)
            content = extract_pdf_text(file.stream)
        else:
            return jsonify({"detail": "Only TXT, DOCX, and PDF supported"}), 400

        if not content.strip():
            return jsonify({"detail": "Empty or unreadable file content"}), 400

        logger.info(f"Transcript uploaded. Size: {len(content)} chars")

        prompt_text = f"""
        You are an intelligent meeting assistant tasked with analyzing business meeting transcripts.

        Your goals:
        1. Provide a professional summary of the key discussion points (within 100–150 words).
        2. Extract 3–7 clear, actionable items from the conversation, each in the form:
        - Action: [What needs to be done]
        - Responsible: [Name or role if mentioned, else say "Unassigned"]
        - Deadline: [Mention if present, else say "Not specified"]

        Input Transcript:
        \"\"\"
        {content}
        \"\"\"

        Return your response strictly in this JSON format:
        {{
        "summary": "Concise and professional summary of the meeting...",
        "action_items": [
            "Action: ..., Responsible: ..., Deadline: ...",
            "Action: ..., Responsible: ..., Deadline: ...",
            ...
        ]
        }}

        ### Constraints:
        - Do not return any markdown, bullet points, or formatting outside the JSON structure.
        - Ensure the JSON is valid and directly parsable.
        - Avoid hallucinating information that isn't in the transcript.
        - If any detail (e.g., person or deadline) is not mentioned, use the default values provided above.
        """


        response = llm.invoke(prompt_text)
        result = response.content.strip()

        if result.startswith("```json"):
            result = result.replace("```json", "", 1).replace("```", "", 1).strip()

        data = json.loads(result)
        return jsonify({"summary": data.get("summary", ""), "action_items": data.get("action_items", [])})

    except Exception as e:
        logger.error(f"Error analyzing transcript: {str(e)}")
        return jsonify({"detail": f"Processing error: {str(e)}"}), 500


@app.route("/invoice", methods=["GET", "POST"], endpoint="invoice")
@check_token_status
def upload_invoice():
    if request.method == "POST":
        if 'file' not in request.files:
            return jsonify({"detail": "No file uploaded."}), 400

        file = request.files['file']
        filename = secure_filename(file.filename)

        if not filename.lower().endswith(('.pdf', '.jpg', '.jpeg', '.png')):
            return jsonify({"detail": "Unsupported file format."}), 400

        os.makedirs("static/uploads", exist_ok=True)
        save_path = os.path.join("static/uploads", filename)
        file.save(save_path)

        try:
            if filename.lower().endswith('.pdf'):
                from pdfminer.high_level import extract_text
                extracted_text = extract_text(save_path)
            else:
                pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
                # pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
                from PIL import Image
                image = Image.open(save_path)
                extracted_text = pytesseract.image_to_string(image)
            logger.info(f"Extracted text from {filename}")
        except Exception as e:
            logger.error(f"Failed to extract text from {filename}: {str(e)}")
            return jsonify({"detail": "Failed to extract text", "error": str(e)}), 500
        finally:
            # Delete the uploaded file
            try:
                if os.path.exists(save_path):
                    os.remove(save_path)
                    logger.info(f"Deleted file: {save_path}")
                else:
                    logger.warning(f"File not found for deletion: {save_path}")
            except Exception as e:
                logger.error(f"Failed to delete file {save_path}: {str(e)}")

        # Prompt AI for invoice extraction
        
        prompt_text = f"""
        You are an intelligent and precise AI assistant trained to extract key details from invoices or receipts.

        Below is the raw text extracted from a document. Please carefully identify and extract the following fields:

        1. **Vendor Name** – The name of the company or individual who issued the invoice.
        2. **Invoice Date** – The date on which the invoice was issued.
        3. **Total Amount** – The full payable amount (include currency if available).
        4. **Tax Amount** – If a tax or VAT is mentioned, provide the amount (or write "Not mentioned").
        5. **Itemized List** – A concise list of the items or services billed. Extract item names only (ignore price or quantity unless clearly stated).
        6. **Suggested Category** – Classify the expense into a business category such as: "Travel", "Meals", "Office Supplies", "Utilities", "Software", "Marketing", "Healthcare", etc.

        Please respond **strictly in raw JSON format only**, without any markdown syntax or explanation. Follow this structure exactly:

        {{
        "vendor": "Example Vendor Name",
        "date": "DD-MM-YYYY or similar",
        "amount": "₹1234.56 or $1234.56",
        "tax": "₹123.45 or Not mentioned",
        "items": ["Item 1", "Item 2", "Item 3"],
        "category": "Office Supplies"
        }}

        Extract as accurately as possible. If some fields are missing in the text, leave them as empty strings or write "Not mentioned".

        Extracted Document Text:
        -------------------------
        {extracted_text}
        """


        response = llm.invoke(prompt_text)
        result = response.content.strip()
        if result.startswith("```json"):
            result = result.replace("```json", "", 1).replace("```", "", 1).strip()

        try:
            data = json.loads(result)
        except Exception as e:
            return jsonify({"detail": "Failed to parse AI response", "error": str(e)}), 500

        return jsonify({
            "filename": filename,
            "data": data
        })

    return render_template("invoice.html")



# WebSocket (SocketIO) endpoint
@socketio.on('connect')
def handle_connect():
    session_id = 'default'
    logger.info("WebSocket connection started")
    store[session_id] = ChatMessageHistory()
    emit('message', {"sender": "system", "message": "Welcome to the AI Chatbot! How can I assist you today?"})

@socketio.on('disconnect')
def handle_disconnect():
    logger.info("WebSocket disconnected")

@socketio.on('message')
def handle_message(message):
    session_id = 'default'
    if message.lower() in ["quit", "exit"]:
        emit('message', {"sender": "system", "message": "Goodbye!"})
        disconnect()
        return
    if message.lower().startswith("generate image:"):
        art_prompt = message[len("generate image:"):].strip()
        ascii_art = generate_ascii_art(art_prompt)
        emit('message', {"sender": "ai", "message": ascii_art})
        return
    response = ""
    try:
        result = runnable_with_history.invoke({"input": message}, config={"configurable": {"session_id": session_id}})
        # result may be a string, dict, or tuple depending on your chain
        if isinstance(result, str):
            response = result
        elif isinstance(result, dict) and "content" in result:
            response = result["content"]
        elif hasattr(result, "content"):
            response = result.content
        elif isinstance(result, tuple):
            # Try to join string parts of the tuple
            response = " ".join(str(x) for x in result)
        else:
            response = str(result)
        emit('message', {"sender": "ai", "message": response.strip()})
    except Exception as e:
        emit('message', {"sender": "system", "message": f"Error: {str(e)}"})

# Database config
DB_NAME = os.environ.get('DATABASE_NAME', 'docuai')
DB_USER = os.environ.get('DATABASE_USER', 'postgres')
DB_PASSWORD = os.environ.get('DATABASE_PASSWORD', 'Root#12345')
DB_HOST = os.environ.get('DATABASE_HOST', 'localhost')
DB_PORT = os.environ.get('DATABASE_PORT', '5432')

def get_db_connection():
    conn = psycopg2.connect(
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT
    )
    return conn

# --- AUTH API ENDPOINTS ---
from flask import session

def send_verification_email(to_email, token):
    EMAIL_USER = os.environ.get('EMAIL_USER')
    EMAIL_PASS = os.environ.get('EMAIL_PASS')
    if not EMAIL_USER or not EMAIL_PASS:
        logger.error('SMTP credentials not set')
        return False
    msg = EmailMessage()
    msg['Subject'] = 'Verify your email for DocuMind AI'
    msg['From'] = EMAIL_USER
    msg['To'] = to_email
    verify_url = f"{request.url_root.rstrip('/')}/verify-email/{token}"
    msg.set_content(f"""
    Welcome to DocuMind AI!

    Please verify your email address by clicking the link below:
    {verify_url}

    If you did not sign up, you can ignore this email.
    """)
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_USER, EMAIL_PASS)
            smtp.send_message(msg)
        logger.info(f"Verification email sent to {to_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send verification email: {str(e)}")
        return False

@app.route('/verify-email/<token>')
def verify_email(token):
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id, verification_token_expiry FROM users WHERE verification_token=%s', (token,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return 'Invalid or expired verification link.', 400
        user_id, expiry = row
        if expiry and datetime.utcnow() > expiry:
            cur.close(); conn.close()
            return 'Verification link has expired.', 400
        cur.execute('UPDATE users SET is_verified=TRUE, verification_token=NULL, verification_token_expiry=NULL WHERE id=%s', (user_id,))
        conn.commit()
        cur.close(); conn.close()
        return 'Email verified! You can now log in.'
    except Exception as e:
        logger.error(f"Email verification error: {str(e)}")
        return 'Verification failed.', 500

@app.route('/api/signup', methods=['POST'])
def api_signup():
    data = request.get_json()
    name = data.get('name')
    email = data.get('email')
    password = data.get('password')
    if not all([name, email, password]):
        return jsonify({'detail': 'Missing required fields'}), 400
    hashed_pw = generate_password_hash(password)
    token = secrets.token_urlsafe(32)
    expiry = datetime.utcnow() + timedelta(hours=24)
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('SELECT id FROM users WHERE email=%s', (email,))
        if cur.fetchone():
            return jsonify({'detail': 'Email already registered'}), 409
        cur.execute('''INSERT INTO users (full_name, email, password, is_verified, verification_token, verification_token_expiry) VALUES (%s, %s, %s, %s, %s, %s) RETURNING id''',
                    (name, email, hashed_pw, False, token, expiry))
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        send_verification_email(email, token)
        return jsonify({'detail': 'Signup successful. Please check your email to verify your account.', 'user_id': user_id}), 201
    except Exception as e:
        logger.error(f"Signup error: {str(e)}")
        return jsonify({'detail': f'Error: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def api_login():
    data = request.get_json()
    email = data.get('email')
    password = data.get('password')
    if not all([email, password]):
        return jsonify({'detail': 'Missing required fields'}), 400
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute('SELECT * FROM users WHERE email=%s', (email,))
        user = cur.fetchone()
        if not user or not user['password']:
            cur.close(); conn.close()
            return jsonify({'detail': 'Invalid credentials'}), 401
        if not check_password_hash(user['password'], password):
            cur.close(); conn.close()
            return jsonify({'detail': 'Invalid credentials'}), 401
        if not user['is_verified']:
            cur.close(); conn.close()
            return jsonify({'detail': 'Please verify your email before logging in.'}), 403
        # Generate token
        token = secrets.token_urlsafe(32)
        user_id = user['id']
        created_at = datetime.utcnow()
        expired_at = created_at + timedelta(days=7)  # Token valid for 7 days
        # Store token in login_info
        cur2 = conn.cursor()
        cur2.execute('''INSERT INTO login_info (user_id, token, active_status, created_at, expired_at) VALUES (%s, %s, %s, %s, %s)''',
                    (user_id, token, 1, created_at, expired_at))
        conn.commit()
        cur2.close()
        cur.close()
        conn.close()
        return jsonify({'detail': 'Login successful', 'token': token, 'user': {'id': user['id'], 'full_name': user['full_name'], 'email': user['email'], 'avatar_url': user.get('avatar_url')}})
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        return jsonify({'detail': f'Error: {str(e)}'}), 500

@app.route('/api/logout', methods=['POST'])
@check_token_status
def api_logout():
    token = request.headers.get('Authorization')
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('UPDATE login_info SET active_status=0, expired_at=%s WHERE token=%s', (datetime.utcnow(), token))
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({'detail': 'Logged out successfully.'})
    except Exception as e:
        logger.error(f"Logout error: {str(e)}")
        return jsonify({'detail': f'Error: {str(e)}'}), 500

@app.route('/api/change-password', methods=['POST'])
@check_token_status
def api_change_password():
    token = request.headers.get('Authorization') or request.cookies.get('token')
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    confirm_password = data.get('confirm_password')
    if not all([current_password, new_password, confirm_password]):
        return jsonify({'detail': 'All fields are required.'}), 400
    if new_password != confirm_password:
        return jsonify({'detail': 'New password and confirm password do not match.'}), 400
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Get user_id from token
        cur.execute('SELECT user_id FROM login_info WHERE token=%s AND active_status=1', (token,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return jsonify({'detail': 'Invalid or expired token.'}), 401
        user_id = row['user_id']
        # Get current password hash
        cur.execute('SELECT password FROM users WHERE id=%s', (user_id,))
        user = cur.fetchone()
        if not user or not user['password']:
            cur.close(); conn.close()
            return jsonify({'detail': 'User not found.'}), 404
        if not check_password_hash(user['password'], current_password):
            cur.close(); conn.close()
            return jsonify({'detail': 'Current password is incorrect.'}), 400
        # Update password
        new_hash = generate_password_hash(new_password)
        cur.execute('UPDATE users SET password=%s WHERE id=%s', (new_hash, user_id))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({'detail': 'Password changed successfully.'})
    except Exception as e:
        logger.error(f"Change password error: {str(e)}")
        return jsonify({'detail': f'Error: {str(e)}'}), 500

@app.route('/api/delete-account', methods=['POST'])
@check_token_status
def api_delete_account():
    token = request.headers.get('Authorization') or request.cookies.get('token')
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        # Get user_id from token
        cur.execute('SELECT user_id FROM login_info WHERE token=%s AND active_status=1', (token,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return jsonify({'detail': 'Invalid or expired token.'}), 401
        user_id = row[0]
        # Delete login_info entries
        cur.execute('DELETE FROM login_info WHERE user_id=%s', (user_id,))
        # Delete user
        cur.execute('DELETE FROM users WHERE id=%s', (user_id,))
        conn.commit()
        cur.close(); conn.close()
        return jsonify({'detail': 'Account deleted successfully.'})
    except Exception as e:
        logger.error(f"Delete account error: {str(e)}")
        return jsonify({'detail': f'Error: {str(e)}'}), 500

@app.route('/api/social-auth', methods=['POST'])
def api_social_auth():
    data = request.get_json()
    provider = data.get('provider')
    provider_id = data.get('provider_id')
    email = data.get('email')
    name = data.get('name')
    avatar_url = data.get('avatar_url')
    if not all([provider, provider_id, email, name]):
        return jsonify({'detail': 'Missing required fields'}), 400
    try:
        conn = get_db_connection()
        cur = conn.cursor(cursor_factory=RealDictCursor)
        # Check if user exists by provider_id
        cur.execute('SELECT * FROM users WHERE provider=%s AND provider_id=%s', (provider, provider_id))
        user = cur.fetchone()
        if not user:
            # If not, check by email (for first-time social login)
            cur.execute('SELECT * FROM users WHERE email=%s', (email,))
            user_by_email = cur.fetchone()
            if user_by_email:
                # Update user with provider info
                cur.execute('''UPDATE users SET provider=%s, provider_id=%s, avatar_url=%s WHERE email=%s''',
                            (provider, provider_id, avatar_url, email))
                conn.commit()
                user = user_by_email
            else:
                # Create new user
                cur.execute('''INSERT INTO users (full_name, email, provider, provider_id, avatar_url, is_verified) VALUES (%s, %s, %s, %s, %s, %s) RETURNING *''',
                            (name, email, provider, provider_id, avatar_url, True))
                user = cur.fetchone()
                conn.commit()
        # Generate token and insert into login_info
        token = secrets.token_urlsafe(32)
        user_id = user['id']
        created_at = datetime.utcnow()
        expired_at = created_at + timedelta(days=7)
        cur2 = conn.cursor()
        cur2.execute('''INSERT INTO login_info (user_id, token, active_status, created_at, expired_at) VALUES (%s, %s, %s, %s, %s)''',
                    (user_id, token, 1, created_at, expired_at))
        conn.commit()
        cur2.close()
        cur.close()
        conn.close()
        return jsonify({'detail': 'Social auth successful', 'token': token, 'user': {'id': user['id'], 'full_name': user['full_name'], 'email': user['email'], 'avatar_url': user.get('avatar_url')}})
    except Exception as e:
        logger.error(f"Social auth error: {str(e)}")
        return jsonify({'detail': f'Error: {str(e)}'}), 500

from dotenv import load_dotenv
load_dotenv()
# --- OAUTH CONFIG ---
OAUTH_CLIENTS = {
    'google': {
        'client_id': os.environ.get('GOOGLE_CLIENT_ID', ''),
        'client_secret': os.environ.get('GOOGLE_CLIENT_SECRET', ''),
        'api_base_url': 'https://www.googleapis.com/oauth2/v2/',
        'access_token_url': 'https://oauth2.googleapis.com/token',
        'authorize_url': 'https://accounts.google.com/o/oauth2/auth',
        'userinfo_endpoint': 'https://www.googleapis.com/oauth2/v2/userinfo',
        'client_kwargs': {'scope': 'openid email profile'},
        'server_metadata_url': 'https://accounts.google.com/.well-known/openid-configuration'
    }
    # 'github': {
    #     'client_id': os.environ.get('GITHUB_CLIENT_ID', ''),
    #     'client_secret': os.environ.get('GITHUB_CLIENT_SECRET', ''),
    #     'api_base_url': 'https://api.github.com/',
    #     'access_token_url': 'https://github.com/login/oauth/access_token',
    #     'authorize_url': 'https://github.com/login/oauth/authorize',
    #     'userinfo_endpoint': 'https://api.github.com/user',
    #     'client_kwargs': {'scope': 'user:email'}
    # }
    # 'microsoft': {
    #     'client_id': os.environ.get('MICROSOFT_CLIENT_ID', ''),
    #     'client_secret': os.environ.get('MICROSOFT_CLIENT_SECRET', ''),
    #     'api_base_url': 'https://graph.microsoft.com/v1.0/',
    #     'access_token_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/token',
    #     'authorize_url': 'https://login.microsoftonline.com/common/oauth2/v2.0/authorize',
    #     'userinfo_endpoint': 'https://graph.microsoft.com/v1.0/me',
    #     'client_kwargs': {'scope': 'User.Read openid email profile'}
    # }
}

oauth = OAuth(app)
for provider, conf in OAUTH_CLIENTS.items():
    oauth.register(
        name=provider,
        client_id=conf['client_id'],
        client_secret=conf['client_secret'],
        access_token_url=conf['access_token_url'],
        access_token_params=None,
        authorize_url=conf['authorize_url'],
        authorize_params=None,
        api_base_url=conf['api_base_url'],
        client_kwargs=conf['client_kwargs'],
        server_metadata_url=conf.get('server_metadata_url')
    )

def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

@app.route('/login/<provider>')
def oauth_login(provider):
    if provider not in OAUTH_CLIENTS:
        return 'Unsupported provider', 400
    redirect_uri = url_for('oauth_callback', provider=provider, _external=True)
    print("redirect_uri", redirect_uri)
    return oauth.create_client(provider).authorize_redirect(redirect_uri)

@app.route('/auth/<provider>/callback')
def oauth_callback(provider):
    if provider not in OAUTH_CLIENTS:
        return 'Unsupported provider', 400
    client = oauth.create_client(provider)
    token = client.authorize_access_token()
    print("token", token, client)
    userinfo = None
    if provider == 'google':
        userinfo = client.get('userinfo').json()
        provider_id = userinfo.get('id')
        email = userinfo.get('email')
        name = userinfo.get('name')
        avatar_url = userinfo.get('picture')
    elif provider == 'github':
        userinfo = client.get('user').json()
        provider_id = str(userinfo.get('id'))
        email = userinfo.get('email')
        if not email:
            emails = client.get('user/emails').json()
            for e in emails:
                if e.get('primary') and e.get('verified'):
                    email = e.get('email')
                    break
        name = userinfo.get('name') or userinfo.get('login')
        avatar_url = userinfo.get('avatar_url')
    elif provider == 'microsoft':
        userinfo = client.get('me').json()
        provider_id = userinfo.get('id')
        email = userinfo.get('mail') or userinfo.get('userPrincipalName')
        name = userinfo.get('displayName')
        avatar_url = None
    else:
        return 'Unsupported provider', 400
    # Call /api/social-auth to get token and user
    import requests as pyrequests
    try:
        api_url = request.url_root.rstrip('/') + '/api/social-auth'
        resp = pyrequests.post(api_url, json={
            'provider': provider,
            'provider_id': provider_id,
            'email': email,
            'name': name,
            'avatar_url': avatar_url
        })
        if resp.ok:
            data = resp.json()
            token = data.get('token')
            user = data.get('user')
            if token and user:
                import urllib.parse
                user_str = urllib.parse.quote(json.dumps(user))
                return redirect(f"/login?social=1&token={token}&user={user_str}")
        # fallback: just go to dashboard
        return redirect('/')
    except Exception as e:
        logger.error(f"OAuth error: {str(e)}")
        return 'Authentication failed', 500

RAZORPAY_KEY_ID = os.environ.get('RAZORPAY_KEY_ID', 'rzp_test_Ug9xpT5DD3kxg7')
RAZORPAY_KEY_SECRET = os.environ.get('RAZORPAY_KEY_SECRET', 'phVCS9rHFDKBULRP0mWITi9F')
razorpay_client = razorpay.Client(auth=(RAZORPAY_KEY_ID, RAZORPAY_KEY_SECRET))

# Pricing data for amount calculation
pricingData = {
    'starter': { 'monthly': 0, 'annual': 0 },
    'pro': { 'monthly': 15, 'annual': 10 },
    'business': { 'monthly': 49, 'annual': 39 }
}

def create_transaction_logs_table():
    """Create the transaction_logs table if it doesn't exist."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            CREATE TABLE IF NOT EXISTS transaction_logs (
                id SERIAL PRIMARY KEY,
                user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                plan_type VARCHAR(50) NOT NULL,
                billing_period VARCHAR(20) NOT NULL,
                amount DECIMAL(10,2) NOT NULL,
                currency VARCHAR(3) DEFAULT 'USD',
                stripe_session_id VARCHAR(255),
                stripe_payment_intent_id VARCHAR(255),
                status VARCHAR(50) DEFAULT 'pending',
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        conn.commit()
        cur.close()
        conn.close()
        logger.info("Transaction logs table created/verified successfully")
    except Exception as e:
        logger.error(f"Error creating transaction_logs table: {str(e)}")

create_transaction_logs_table()

def log_transaction(user_id, plan_type, billing_period, amount, stripe_session_id=None, status='pending', metadata=None):
    """Log a transaction to the database."""
    try:
        # Convert metadata dict to JSON string if it's a dict
        if metadata is not None and isinstance(metadata, dict):
            metadata = json.dumps(metadata)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute('''
            INSERT INTO transaction_logs 
            (user_id, plan_type, billing_period, amount, stripe_session_id, status, metadata)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (user_id, plan_type, billing_period, amount, stripe_session_id, status, metadata))
        transaction_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Transaction logged with ID: {transaction_id}")
        return transaction_id
    except Exception as e:
        logger.error(f"Error logging transaction: {str(e)}")
        return None

def update_transaction_status(stripe_session_id, status, stripe_payment_intent_id=None, metadata=None):
    """Update transaction status after payment completion."""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        update_fields = ['status = %s', 'updated_at = CURRENT_TIMESTAMP']
        params = [status]
        
        if stripe_payment_intent_id:
            update_fields.append('stripe_payment_intent_id = %s')
            params.append(stripe_payment_intent_id)
        
        if metadata:
            update_fields.append('metadata = metadata || %s')
            params.append(json.dumps(metadata))
        
        params.append(stripe_session_id)
        
        cur.execute(f'''
            UPDATE transaction_logs 
            SET {', '.join(update_fields)}
            WHERE stripe_session_id = %s
        ''', params)
        conn.commit()
        cur.close()
        conn.close()
        logger.info(f"Transaction status updated to {status} for session {stripe_session_id}")
        return True
    except Exception as e:
        logger.error(f"Error updating transaction status: {str(e)}")
        return False

@app.route('/api/create-razorpay-order', methods=['POST'])
@check_token_status
def create_razorpay_order():
    data = request.get_json()
    plan = data.get('plan')
    period = data.get('period')
    amount_usd = data.get('amount_usd')
    amount_inr = data.get('amount_inr')
    is_yearly = data.get('is_yearly', False)
    
    if plan not in pricingData or period not in pricingData[plan]:
        return jsonify({'detail': 'Invalid plan or period'}), 400

    # Use the converted INR amount from frontend, or fallback to backend calculation
    if amount_inr is not None:
        amount = int(amount_inr)  # Frontend already converts to paise
    else:
        # Fallback calculation (keeping original logic as backup)
        usd_amount = pricingData[plan][period]
        if is_yearly:
            usd_amount *= 12  # Multiply by 12 for yearly billing
        # Convert USD to INR (approximate rate)
        usd_to_inr_rate = 83.5
        amount = int(usd_amount * usd_to_inr_rate * 100)  # Convert to paise
    
    currency = 'INR'
    receipt = f"{plan}_{period}_{secrets.token_hex(8)}"
    print(f"Creating order: plan={plan}, period={period}, amount={amount}, is_yearly={is_yearly}")

    # Get user info from token
    token = request.headers.get('Authorization') or request.cookies.get('token')
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT u.id, u.email FROM users u JOIN login_info l ON u.id = l.user_id WHERE l.token=%s', (token,))
    row = cur.fetchone()
    cur.close(); conn.close()
    if not row:
        return jsonify({'detail': 'User not found'}), 404
    print("row", row)
    user_id, customer_email = row

    # Create Razorpay order
    order = razorpay_client.order.create({
        'amount': amount,
        'currency': currency,
        'receipt': receipt,
        'payment_capture': 1
    })

    print(order)

    # Log the transaction with USD amount for reference
    transaction_id = log_transaction(
        user_id=user_id,
        plan_type=plan,
        billing_period=period,
        amount=amount_usd if amount_usd is not None else pricingData[plan][period],
        stripe_session_id=order['id'],
        status='created',
        metadata={
            'plan_name': plan.title(),
            'period_name': period.title(),
            'amount_usd': amount_usd,
            'amount_inr_paise': amount,
            'is_yearly': is_yearly,
            'currency': currency
        }
    )

    return jsonify({
        'order_id': order['id'],
        'amount': amount,
        'currency': currency,
        'key_id': RAZORPAY_KEY_ID,
        'customer_email': customer_email,
        'transaction_id': transaction_id
    })

@app.route('/policy')
def policy():
    return render_template('policy.html')

@app.route('/api/razorpay-webhook', methods=['POST'])
def razorpay_webhook():
    payload = request.get_data()
    signature = request.headers.get('X-Razorpay-Signature')
    webhook_secret = os.environ.get('RAZORPAY_WEBHOOK_SECRET', 'webhook')

    try:
        generated_signature = hmac.new(
            webhook_secret.encode('utf-8'),
            msg=payload,
            digestmod=hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(generated_signature, signature):
            raise ValueError("Invalid webhook signature")
        event = request.json
        if event['event'] == 'payment.captured':
            payment = event['payload']['payment']['entity']
            order_id = payment['order_id']
            payment_id = payment['id']
            # Update transaction status
            update_transaction_status(
                stripe_session_id=order_id,  # Rename this field to razorpay_order_id in your DB for clarity
                status='completed',
                stripe_payment_intent_id=payment_id,
                metadata={'razorpay_payment_id': payment_id}
            )
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Razorpay webhook error: {str(e)}")
        return jsonify({'detail': 'Webhook verification failed'}), 400

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)
