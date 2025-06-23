# DocuMind AI Chatbot

A powerful AI-powered document analysis and chat assistant web app built with Flask, Socket.IO, and LangChain.  
Supports PDF, DOCX, TXT, CSV, Excel, and image-based document analysis, translation, resume/job matching, contract review, business report analysis, and more.

## Features

- Real-time AI chat (Google Gemini, OpenAI, etc.)
- Document upload and extraction (PDF, DOCX, TXT, CSV, XLS/XLSX, images)
- Resume/job description matching
- Contract analyzer
- Business report analyzer (SWOT, KPIs, recommendations)
- Meeting transcript summarizer
- Invoice/receipt data extraction (including OCR)
- Document translation and summary
- E-book to micro-lessons generator

## Setup

1. **Clone the repository**  
   ```
   git clone <your-repo-url>
   cd <project-folder>
   ```

2. **Install dependencies**  
   ```
   pip install -r requirements.txt
   ```

3. **Set up Tesseract OCR**  
   - Download and install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` path in `app.py` if needed.

4. **Set your API keys**  
   - Edit `app.py` and set your `GOOGLE_API_KEY` and `OPENROUTER_API_KEY` as environment variables or directly in the code.

5. **Run the app**  
   ```
   python app.py
   ```
   The app will be available at [http://localhost:8000](http://localhost:8000).

## Usage

- Access `/chat/` for the main AI chat interface.
- Upload documents for analysis, translation, or extraction via the respective pages.
- Use the navigation bar to explore all features.

## Notes

- For production, use a production-ready WSGI server and secure your API keys.
- Some features require internet access for AI APIs.
- OCR and image extraction require Tesseract and Pillow.

##
