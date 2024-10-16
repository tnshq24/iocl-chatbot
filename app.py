import fitz  # PyMuPDF
import re
import requests
from io import BytesIO
from flask import Flask, render_template, request, send_file
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, DictionaryObject, FloatObject, NameObject, NumberObject
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.data.path.append('/root/nltk_data')  # Add this line to specify the NLTK data path
nltk.download('punkt', download_dir='/root/nltk_data')  # Download punkt to the specified directory

app = Flask(__name__)

# URL to PDF file
PDF_URL = "https://ioclhrchatgpt.blob.core.windows.net/documents/hrhbtb.pdf"

# Text cleaning function
def clean_text(text):
    return re.sub(r'\s+', ' ', re.sub(r'[•\n]', ' ', text)).strip()

def extract_page_numbers(chatbot_response):
    """
    Extracts page numbers from references of the format 'hrhbtb-page-X.pdf'.

    Args:
    - chatbot_response (str): The response text from the chatbot.

    Returns:
    - list: A list of extracted page numbers.
    """
    # Regular expression to match page references in the format hrhbtb-page-X.pdf
    pattern = r'hrhbtb-page-(\d+)\.pdf'
    
    # Find all matches
    matches = re.findall(pattern, chatbot_response)
    
    return list(set(map(int, matches)))  # Convert to int and return unique page numbers

# Extract sentences from PDF
def extract_sentences_from_pdf(pdf_content):
    sentences = []
    pdf = fitz.open(stream=pdf_content, filetype="pdf")
    for page_num in range(len(pdf)):
        page = pdf[page_num]
        text = clean_text(page.get_text("text"))
        sentences.extend(sent_tokenize(text))
    return sentences

# Find most similar sentence
def find_most_similar_sentence(chatbot_response, pdf_sentences):
    documents = [chatbot_response] + pdf_sentences
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
    best_match_index = similarities.argmax()
    return pdf_sentences[best_match_index], similarities[best_match_index]

# Create a highlight annotation
def highlight_annotation(bounds, color=[1, 1, 0]):
    x0, y0, x1, y1 = [list(sub_list) for sub_list in zip(*bounds)]
    rect_bbox = [min(x0), min(y0), max(x1), max(y1)]
    quad_points = []
    for bbox in bounds:
        x1, y1, x2, y2 = bbox
        quad_points.extend([x1, y2, x2, y2, x1, y1, x2, y1])
    return DictionaryObject({
        NameObject("/F"): NumberObject(4),
        NameObject("/Type"): NameObject("/Annot"),
        NameObject("/Subtype"): NameObject("/Highlight"),
        NameObject("/C"): ArrayObject([FloatObject(c) for c in color]),
        NameObject("/Rect"): ArrayObject([FloatObject(c) for c in rect_bbox]),
        NameObject("/QuadPoints"): ArrayObject([FloatObject(c) for c in quad_points]),
    })

# Highlight similar text in PDF
def highlight_text_in_page(page, search_text):
    rects = page.search_for(search_text)
    p1 = rects[0].tl  # top-left point of first rectangle
    p2 = rects[-1].br  # bottom-right point of last rectangle

    # mark text that potentially extends across multiple lines
    page.add_highlight_annot(start=p1, stop=p2)

@app.route('/view-pdf')
def view_pdf():
    # Fetch PDF content
    response = requests.get(PDF_URL)
    if response.status_code != 200:
        return "Unable to fetch the PDF.", 500

    pdf_content = BytesIO(response.content)
    chatbot_response = request.args.get('response', '').strip()
    pages = extract_page_numbers(chatbot_response)


    if not chatbot_response:
        return "No chatbot response provided.", 400

    # Extract sentences from the PDF
    pdf_sentences = extract_sentences_from_pdf(pdf_content.getvalue())
    # Find the most similar sentence to the chatbot response
    most_similar_sentence, similarity_score = find_most_similar_sentence(chatbot_response, pdf_sentences)
    print(most_similar_sentence)
    if similarity_score < 0.1:
        return "No highly similar sentence found.", 400

    try:
        # Use fitz to process pages for highlighting
        pdf_fitz = fitz.open(stream=pdf_content.getvalue(), filetype="pdf")

        # Process specified pages to highlight the found sentence
        for page_num in range(len(pdf_fitz)):  # Ensure page number is valid (use <= for 1-based indexing)
            pdf_page = pdf_fitz[page_num]  # Get the specific page
            print(len(pdf_fitz))
            print('page_num', page_num)
            # Attempt to highlight the text on the current page
            rects = pdf_page.search_for(most_similar_sentence)  # Search for the text on the current page
            print('rects',rects)
            if rects:  # If rectangles are found, highlight them
                highlight_text_in_page(pdf_page, most_similar_sentence)
                break  # Exit the loop after highlighting the first found instance


        # Save the modified PDF document
        output_stream = BytesIO()
        pdf_fitz.save(output_stream)  # Save the modified PDF into the output stream
        pdf_fitz.close()  # Close the document
        output_stream.seek(0)  # Reset stream position to the start

        return send_file(
            output_stream,
            mimetype='application/pdf',
            as_attachment=False,
            download_name="highlighted_pdf.pdf"
        )

    except Exception as e:
        return f"An error occurred: {str(e)}", 500

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
