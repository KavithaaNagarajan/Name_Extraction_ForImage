# Name_Extraction_ForImage
![image](https://github.com/user-attachments/assets/eeca0594-7513-45a2-ab07-e1493d0b1b88)


![image](https://github.com/user-attachments/assets/2ea162f9-b080-475a-8b86-b9fb213b0fbe)
# Image Processing and Data Extraction Web Application
This project is a Flask-based web application designed for processing images and extracting structured data from them. The application performs several image processing tasks, including text extraction from images, text cleaning, and extracting key data points such as university names, degrees, and years. It supports multiple file formats and saves the results in CSV files, which can be viewed or downloaded via the web interface.

## Features
### Upload Files: Allows users to upload various image files (e.g., PDFs, PNG, JPG) for processing.
### Image Processing: Processes the uploaded files using multiple steps to extract text and data from the images.
### Data Extraction: Extracts specific pieces of information, such as university names, degrees, and years, from the images.
### Data Cleaning: Cleans and refines the extracted data for further processing.
### University Name Matching: Matches extracted university names with known keywords.
### CSV Export: Provides the processed data in CSV format, with the ability to download the final results.
### Web Interface: View and download the CSV results through the web interface.
# Installation Requirements
### 1.Python 3.x
### 2.Flask: Web framework for handling routes and rendering templates.
### 3.Werkzeug: Utility for secure file handling.
### 4.Pandas: For reading and writing CSV files.
### 5.Python Scripts: Custom Python scripts for image processing and text extraction (ImageProcessor, LayoutAwareTextExtractor, etc.).
### 6.Poppler: For PDF image extraction (needed for converting PDF to images).
### To install the required dependencies, run:

### bash
### Copy
### pip install -r requirements.txt
# Setup
### Install Poppler:

You will need to have Poppler installed for PDF to image conversion. The application uses Poppler for processing PDF files. You can download and install it from Poppler.
Ensure the path to the Poppler installation is set correctly in the script (e.g., poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin").
### Configure the Flask Application:

The app is configured to allow file uploads and store them in the UPLOAD_FOLDER. The path for uploading files should be defined in the application (e.g., UPLOAD_FOLDER = 'path_to_upload_folder').
Ensure the output directories for processed images and final results are also set.
### File Processing Workflow
### 1. Upload Files
The user uploads files (e.g., PDFs, images) via the web interface.
The uploaded files are checked for allowed extensions (pdf, png, jpg, jpeg, gif) and stored in the UPLOAD_FOLDER.
### 2. Image Processing (ImageProcessor)
The uploaded images (including PDFs) are processed using custom Python scripts to extract image data.
Poppler is used to convert PDFs to images.
### 3. Text Extraction (LayoutAwareTextExtractor)
The extracted images are then passed through a text extraction step where the layout of the text is analyzed, and the text is saved to a CSV file.
### 4. Text Cleaning (TextCleaner, uniTextCleaner)
The extracted text undergoes a cleaning process to remove irrelevant or unwanted information, improving the quality of the extracted data.
### 5. Data Extraction (UniversityExtractor, DegreeExtractor, YearExtractor)
Specific data points such as university names, degree types, and years are extracted from the cleaned text. Each step outputs a CSV with the extracted data.
### 6. University Name Matching (UniversityNameMatcher)
The extracted university names are compared with a list of known university keywords to match them with their corresponding universities.
### 7. Final Data Processing (UniversityDataProcessor_all)
All extracted data is processed, cleaned, and saved into a final CSV file for easy access and download.
# Web Interface
### 1. Home Page (/)
The home page (index.html) allows users to upload their files for processing. It also includes a simple flash message system to indicate the success or failure of file uploads.
###2. CSV View Page (/view_csv)
After processing the files, users can view the extracted data in a table format. The processed data is presented in an HTML table using Pandas to convert the CSV to HTML.
### 3. CSV Download (/download_csv)
Users can download the final processed CSV file containing the extracted and cleaned data.
### How to Run the Application
### Start the Flask Server:

### In the project directory, run the Flask server:
### bash
### Copy
python app.py
The server will be accessible at http://localhost:8868/.
### Upload Files:

On the home page, click the button to upload files (PDFs or images).
The files will be processed sequentially.
### View or Download Processed Data:

After processing, the final CSV can be viewed on the View CSV page or downloaded directly via the Download CSV link.
### Example Workflow
A user uploads an image or PDF containing text.
The application processes the file, extracting text from the image using layout-aware techniques.
The text is cleaned, and university names, degrees, and years are extracted.
The extracted data is matched with known university names and saved to a CSV file.
The user can view or download the CSV file containing the extracted and matched data.
### Known Issues
Poppler must be installed and configured correctly for PDF files to be processed.
File size limitations may apply based on server configuration.
### Credits
Poppler for PDF image extraction.
Flask for the web framework.
Pandas for handling CSV data.
Python Scripts for custom image processing, text extraction, and data handling.

