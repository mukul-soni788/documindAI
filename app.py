import os
import logging
import json
from flask import Flask, render_template, request, jsonify, send_from_directory
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyCt7cnq5RCSdr0Ofb8qTY-1lis69pKGfDo"
os.environ["OPENROUTER_API_KEY"] = "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"

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



# Routes
@app.route("/chat/")
def get_home():
    return render_template("index.html")

@app.route("/portfolio")
def portfolio():
    return render_template("test.html")

@app.route("/")
def base():
    return render_template("base.html")

@app.route("/document", endpoint="document")
def get_document():
    return render_template("document.html")

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')


@app.route("/analyze", methods=["POST"])
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
def resume_match_page():
    return render_template("resume_match.html")

@app.route("/analyze-resume", methods=["POST"])
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
def contract_analyzer():
    return render_template("contract_analyzer.html")

@app.route("/analyze_contract", methods=["POST"])
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
def translate_doc_page():
    return render_template("translate.html")


from deep_translator import GoogleTranslator

@app.route("/translate", methods=["POST"])
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
def business_report():
    return render_template("business_report.html")

# API endpoint to analyze uploaded financial/marketing report
@app.route("/analyze-business-report", methods=["POST"])
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
def transcript_page():
    return render_template("transcript.html")


@app.route("/analyze-transcript", methods=["POST"])
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
        - Avoid hallucinating information that isn’t in the transcript.
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
                import pytesseract
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

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=8000)
