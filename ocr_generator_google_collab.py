import importlib
import subprocess

# List of packages that need to be checked/installed
packages = [
    ("openai", "openai"),
    ("fitz", "pymupdf"),
    ("google.cloud.vision", "google-cloud-vision"),
    ("PIL", "pillow"),
    ("tqdm", "tqdm")
]

def check_and_install(package_name, install_name):
    # Check if the package is installed, if not, install it
    try:
        importlib.import_module(package_name)
        print(f"{package_name} is already installed.")
    except ImportError:
        print(f"{package_name} is not installed. Installing...")
        subprocess.check_call(["pip", "install", install_name])
        print(f"{package_name} has been installed.")

# Iterate through the list of packages and ensure they are installed
for package_name, install_name in packages:
    check_and_install(package_name, install_name)

import os
import io
import re
import time
import openai
import fitz # PyMuPDF
from google.cloud import vision
from google.colab import drive  # For Google Drive integration
from PIL import Image
from tqdm import tqdm  # For progress bars
import os

# Custom exception for invalid file types
class InvalidFileTypeError(Exception):
    pass
    
def setup_environment():
    # Mount Google Drive and set up environment variables
    print("Debug: Mounting Google Drive and setting up environment.")
    drive.mount('/content/drive')
    json_key_path = ""  # Designate the path to your Google Vision OCR JSON key
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = json_key_path

    # Define paths for PDF folder and OCR output folder
    pdf_folder_path = "" # Designate the path to the folder containing the PDFs you wish to OCR
    ocr_output_folder_path = "" # Designate the path to the output folder for the OCR'd PDFs
    os.makedirs(ocr_output_folder_path, exist_ok=True)

    print(f"Debug: PDF folder path: {pdf_folder_path}, OCR output folder path: {ocr_output_folder_path}")
    return pdf_folder_path, ocr_output_folder_path

def list_pdfs(pdf_folder):
    # List all PDF files in the specified folder
    print(f"Debug: Listing PDFs in folder: {pdf_folder}")
    pdf_files = [f for f in os.listdir(pdf_folder) if f.endswith('.pdf')]
    for idx, pdf_file in enumerate(pdf_files):
        print(f"{idx + 1}: {pdf_file}")
    
    # Allow user to select specific PDFs, a range, or all of them
    while True:
        user_input = input("Enter the file number(s) to process (e.g., '1', '1-3', 'all'): ").strip().lower()
        if user_input == 'all':
            print("Debug: User selected all PDFs.")
            return pdf_files
        elif '-' in user_input:
            try:
                start, end = map(int, user_input.split('-'))
                print(f"Debug: User selected range from {start} to {end}.")
                return pdf_files[start - 1:end]
            except ValueError:
                print("Invalid range. Please try again.")
        else:
            try:
                indices = list(map(int, user_input.split(',')))
                print(f"Debug: User selected individual files: {indices}")
                return [pdf_files[i - 1] for i in indices]
            except (ValueError, IndexError):
                print("Invalid input. Please try again.")

def main(debug=False):
    # Main function to orchestrate the OCR process
    print("Debug: Starting main function.")
    pdf_folder, ocr_output_folder = setup_environment()
    selected_pdfs = list_pdfs(pdf_folder)

    if not selected_pdfs:
        print("No PDFs selected for processing.")
        return

    # Process each selected PDF
    for pdf_file in tqdm(selected_pdfs, desc="Processing PDFs"):
        print(f"Processing {pdf_file}...")
        create_hybrid_ocr_pdf(os.path.join(pdf_folder, pdf_file), ocr_output_folder)

def get_ocr_bounding_boxes(image_path):
    # Extract OCR bounding boxes from the given image using Google Cloud Vision
    print(f"Debug: Getting OCR bounding boxes for image: {image_path}")
    client = vision.ImageAnnotatorClient()
    with io.open(image_path, 'rb') as image_file:
        content = image_file.read()
    image = vision.Image(content=content)
    response = client.document_text_detection(image=image)

    bounding_boxes = []
    print("Bounding box extraction starts")

    # Iterate through each page, block, and paragraph to extract word bounding boxes
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                words = []
                vertices = []

                for word in paragraph.words:
                    word_text = ''.join([symbol.text for symbol in word.symbols])
                    word_vertices = [(vertex.x, vertex.y) for vertex in word.bounding_box.vertices]
                    scaled_vertices = [(int(vertex[0] + 10), int(vertex[1] + 35)) for vertex in word_vertices]

                    words.append(word_text)
                    vertices.extend(scaled_vertices)

                    # If the word is punctuation, append it to the previous word
                    if word_text in ".,;!?":
                        bounding_boxes[-1] = (bounding_boxes[-1][0] + word_text, bounding_boxes[-1][1] + scaled_vertices)
                    else:
                        bounding_boxes.append((word_text, scaled_vertices))

    print("Bounding box extraction ends")
    return bounding_boxes

def create_hybrid_ocr_pdf(pdf_path, output_folder, dpi=600):
    # Create a hybrid OCR PDF by adding text overlay to the images
    print(f"Debug: Creating hybrid OCR PDF for: {pdf_path}")
    pdf_document = fitz.open(pdf_path)
    ocr_pdf_filename = os.path.join(output_folder, f"ocr_{os.path.basename(pdf_path)}")
    new_pdf = fitz.open()

    # Iterate through each page in the PDF
    for page_num in tqdm(range(len(pdf_document)), desc="Processing pages in PDF"):
        print(f"Debug: Processing page number: {page_num}")
        page = pdf_document[page_num]
        try:
            pixmap = page.get_pixmap(dpi=dpi)
        except Exception as e:
            print(f"Error generating pixmap for page {page_num} of {pdf_path}: {e}")
            continue
        image_path = "/tmp/temp_image.png"
        pixmap.save(image_path)

        # Get bounding boxes from OCR results
        bounding_boxes = get_ocr_bounding_boxes(image_path)

        # Remove the temporary image file after processing
        os.remove(image_path)

        img_width, img_height = pixmap.width, pixmap.height
        pdf_width, pdf_height = page.rect.width, page.rect.height
        scale_x = pdf_width / img_width
        scale_y = pdf_height / img_height

        # Create a new page in the output PDF and insert the image
        new_page = new_pdf.new_page(width=pdf_width, height=pdf_height)
        new_page.insert_image(page.rect, pixmap=pixmap)

        # Insert extracted text at appropriate locations on the page
        for word_text, vertices in bounding_boxes:
            x0, y0 = vertices[0]
            x1, y1 = vertices[2]
            x0, y0 = x0 * scale_x, y0 * scale_y
            x1, y1 = x1 * scale_x, y1 * scale_y

            box_height = y1 - y0
            font_size = box_height * 0.8
            new_page.insert_text(
                (x0, y0),
                word_text,
                fontsize=font_size,
                fontname="helv",
                color=(0, 0, 0),
                render_mode=3
            )

    # Save the new PDF with OCR text overlay
    new_pdf.save(ocr_pdf_filename)
    new_pdf.close()
    pdf_document.close()
    print(f"Hybrid OCR'd PDF saved to: {ocr_pdf_filename}")

if __name__ == "__main__":
    print("Debug: Starting the script.")
    main(debug=True)
