# Hybrid OCR PDF Generator

This Python script performs OCR (Optical Character Recognition) on PDF files, creates a hybrid version of those PDFs by adding text overlays to the original pages, and saves the new PDF files with enhanced accessibility and searchability.

## Features

- Extracts text from images and PDF files using Google Vision OCR API.
- Overlays recognized text back onto the PDF for better accessibility and search.
- Allows selection of individual PDFs, a range of PDFs, or all PDFs in a folder.
- Generates text bounding boxes to position text in the correct areas of the page.
- Progress indicators to track the processing of multiple pages and PDFs.
- Automatically installs the necessary packages if not already installed.

## Requirements

To run this script, you will need the following:

- Python 3.x
- Google Vision API credentials JSON key
- The following Python packages:
  - `openai`
  - `pymupdf`
  - `google-cloud-vision`
  - `pillow`
  - `tqdm`

The script checks and installs these dependencies automatically.

## Setup

1. **Install Python 3.x**  
   Make sure you have Python 3.x installed on your system. You can download it from [Python.org](https://www.python.org/downloads/).

2. **Google Vision API**  
   Set up Google Cloud Vision API credentials and obtain a JSON key. Save this key in a secure location.

3. **Install Required Packages**  
   The script automatically installs the required packages if they are missing. You can also install them manually using `pip`:

   ```bash
   pip install openai pymupdf google-cloud-vision pillow tqdm
