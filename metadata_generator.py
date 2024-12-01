import os
import subprocess
import json
import time
from tqdm import tqdm
import random
from tenacity import retry, stop_after_attempt, wait_exponential

# Step 0: Function to ensure required packages are installed
def ensure_required_packages():
    required_packages = {
        "openai": "openai==0.28",
        "tqdm": "tqdm",
        "tenacity": "tenacity"
    }

    for package, install_name in required_packages.items():
        try:
            __import__(package)
            print(f"Debug: {package} is already installed.")
        except ImportError:
            print(f"Debug: {package} is not installed. Installing...")
            subprocess.check_call(['pip', 'install', install_name])

# Step 1: Ensure required packages are installed
ensure_required_packages()

# Step 2: Import the required packages
import openai

# Step 3: Ensure proper version of OpenAI is installed (re-affirmation)
print("Debug: Ensuring proper version of OpenAI package is installed")
subprocess.check_call(['pip', 'install', 'openai==0.28'])

# Step 4: Set OpenAI API key
print("Debug: Setting OpenAI API key")
openai.api_key = "YOUR_API_KEY_HERE" 

# Step 5: Set folder paths
print("Debug: Setting folder paths")
ocr_folder = ""  # Folder where OCR text files are saved
metadata_folder = "" # Folder where metadata JSON files are saved

print("Debug: Checking folder existence")
if not os.path.exists(ocr_folder):
    raise SystemExit("OCR folder not found. Please ensure the OCR text files are saved in the correct folder.")
if not os.path.exists(metadata_folder):
    os.makedirs(metadata_folder)

print("Debug: Reading OCR text files from folder: {}".format(ocr_folder))
ocr_files = [f for f in os.listdir(ocr_folder) if f.endswith('.txt')]
while True:
    try:
        num_files_to_process = int(input("Enter the number of files to process (or enter -1 to process all files, or specify a number to process up to that limit): "))
        if num_files_to_process == -1 or num_files_to_process > 0:
            break
        else:
            print("Please enter a valid number greater than 0 or -1 to process all files.")
    except ValueError:
        print("Invalid input. Please enter a numeric value.")

if num_files_to_process > 0:
    ocr_files = ocr_files[:min(len(ocr_files), num_files_to_process)]
if len(ocr_files) == 0:
    raise SystemExit("No OCR text files found in the OCR folder. Please run the OCR extraction first.")

print("Debug: Number of OCR files to process: {}".format(len(ocr_files)))

# Step 6: Helper function to fetch response from OpenAI
print("Debug: Defining helper function to fetch response from OpenAI")
@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=4, max=60))
def fetch_response(prompt, model, max_tokens, temperature, retry_attempt=0):
    print("Debug: Fetching response from OpenAI")
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": "You are an expert metadata generator."}, {"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
            request_timeout=30
        )
        delay = random.uniform(1, 3)
        print(f"Waiting for {delay:.2f} seconds to avoid rate limit...")
        time.sleep(delay)
        return response
    except openai.error.InvalidRequestError as e:
        if 'maximum context length' in str(e) and retry_attempt < 3:
            print("Token limit exceeded, retrying with additional chunking...")
            reduced_prompt = prompt[:len(prompt) // 2]  # Reduce the prompt size by half
            return fetch_response(reduced_prompt, model, max_tokens // 2, temperature, retry_attempt + 1)
        raise
    except openai.error.RateLimitError as e:
        retry_after = int(e.headers.get('Retry-After', 60))
        print(f"Rate limit reached. Waiting for {retry_after} seconds before retrying...")
        time.sleep(retry_after)
        raise
    except openai.error.OpenAIError as e:
        print(f"Error during API request: {e}")
        return None

# Step 7: Function to process a single file
print("Debug: Defining function to process a single file")
def process_file(ocr_file):
    print(f"Debug: Processing file: {ocr_file}")
    try:
        with open(os.path.join(ocr_folder, ocr_file), 'r') as file:
            ocr_text = file.read()
    except FileNotFoundError as e:
        print(f"Error reading file {ocr_file}: {e}")
        return None

    # Split text into manageable chunks if it exceeds token limits
    print("Debug: Splitting text into manageable chunks")
    def chunk_text(text, chunk_size):
        for i in range(0, len(text), chunk_size):
            yield text[i:i + chunk_size]

    chunks = list(chunk_text(ocr_text, 2000))
    batched_chunks = ['\n'.join(chunks[i:i + min(5, len(chunks) - i)]) for i in range(0, len(chunks), 5)]  # Batch chunks into groups of 5

    # Generate metadata fields for the document
    print(f"Debug: Generating metadata for the entire document: {ocr_file}")
    prompt = f"""
    You are an expert metadata curator and librarian specializing in media studies. Your task is to analyze the provided OCR text from a document and generate structured metadata in JSON format. Please ensure the metadata is detailed, contextually accurate, and adheres to the following keys:

- Abstract (Resumen): A concise summary (150-200 words) of the document's primary themes and content.
- Keywords (Palabras Clave): A list of relevant keywords (6-10) summarizing the document's core topics.
- Description: A brief (1-2 sentence) description capturing the essence of the document.
- Title: The main title of the document.
- Creator (Autor): The name(s) of the document's author(s).
- Subject: The general subject or field the document pertains to.
- Basic Keywords: A list of simple keywords summarizing the document.
- Long-tail keywords: Detailed, descriptive keyword phrases related to the document's content.
- SEO Keywords: Keywords optimized for search engines, capturing both broad and niche aspects of the document.

**Output:**
The metadata should be returned as a well-structured JSON object with all keys filled. Use "Unknown" or "Not Available" for any fields that cannot be determined from the input text.

{''.join(chunks)}
"""

    response = fetch_response(prompt, "gpt-4", 500, 0.5)
    if response:
        metadata_content = response['choices'][0]['message']['content'].strip()
        # Convert metadata_content to a dictionary
        try:
            metadata_dict = json.loads(metadata_content)
            metadata_dict["file_name"] = ocr_file  # Add the file name to the metadata dictionary
        except json.JSONDecodeError:
            print("Error: The metadata content is not valid JSON.")
            metadata_dict = {"file_name": ocr_file}
    else:
        metadata_dict = {"file_name": ocr_file}

    return metadata_dict

# Step 8: Process OCR files sequentially
print("Debug: Starting to process OCR files sequentially")
metadata_results = []
for ocr_file in tqdm(ocr_files, desc="Processing OCR Files", bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"):
    result = process_file(ocr_file)
    if result:
        metadata_results.append(result)

# Step 9: Save metadata to a JSON file
metadata_filename = os.path.join(metadata_folder, 'metadata_results.json')
print(f"Saving metadata to file: {metadata_filename}")
with open(metadata_filename, 'w', encoding="utf-8") as json_file:
    json.dump(metadata_results, json_file, indent=4)

print(f"Metadata saved to {metadata_filename}")
