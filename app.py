from flask import Flask, render_template, request, redirect, url_for, flash,send_file
import os
from werkzeug.utils import secure_filename
import pandas as pd
from Pythonscript import ImageProcessor,LayoutAwareTextExtractor,TextCleaner,UniversityExtractor,uniTextCleaner,DegreeExtractor,YearExtractor,UniversityDataProcessor,UniversityNameMatcher,UniversityDataProcessor_all
app = Flask(__name__)

# Configure the upload folder and allowed file extensions
UPLOAD_FOLDER = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\upload_files'
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg', 'gif'}

# Set the secret key for flash messages
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Function to check allowed extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    # Check if the post request has the file part
    if 'files' not in request.files:
        flash('No file part')
        return redirect(request.url)

    files = request.files.getlist('files')
    uploaded_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_files.append(filename)
        else:
            flash(f"File '{file.filename}' is not allowed")
    #step 1 : ImageProcessing
    poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Adjust this path for your system
    input_folder = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\upload_files'
    output_folder = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\output_images'
    
    image_processor = ImageProcessor(poppler_path, input_folder, output_folder)
    image_processor.process_all_files()
    #step 2 : LayoutAwareTextExtractor
    folder_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\output_images'  
    output_csv_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_text.csv' 
    extractor = LayoutAwareTextExtractor()
    df = extractor.process_folder_images(folder_path, output_csv_path)
    #step 3 : TextCleaner
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_text.csv'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext.csv'
    # Initialize the TextCleaner class
    text_cleaner = TextCleaner(input_csv, output_csv)
    text_cleaner.process_data()
    text_cleaner.display_data()
    text_cleaner.save_to_csv()
    #step 4 : UniversityExtractor
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext.csv'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext1.csv'
    university_extractor = UniversityExtractor(input_csv, output_csv)
    university_extractor.process_data()
    university_extractor.display_data()
    university_extractor.save_to_csv()
    #step 5 : uniTextCleaner
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext1.csv'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext2.csv'
    text_cleaner1 = uniTextCleaner(input_csv, output_csv)
    text_cleaner1.process_data()
    text_cleaner1.display_data()
    text_cleaner1.save_to_csv()
    #step 6 : DegreeExtractor
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext2.csv'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext3.csv'
    degree_extractor = DegreeExtractor(input_csv, output_csv)
    degree_extractor.process_data()
    degree_extractor.display_data()
    degree_extractor.save_to_csv()
    #step 7 : YearExtractor
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext3.csv' 
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext4.csv' 
    year_extractor = YearExtractor(input_csv, output_csv)
    year_extractor.process_data()
    year_extractor.display_data()
    year_extractor.save_to_csv()
    #step 8 : UniversityDataProcessor
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext4.csv'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext5.csv' 
    processor1 = UniversityDataProcessor(input_csv, output_csv)
    processor1.process_data()
    processor1.display_data()
    processor1.save_to_csv()
    #step 9 : UniversityNameMatcher
    df_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext5.csv'
    test_csv = r'C:\Users\inc3061\Downloads\university_keywords.xlsx'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\hashing_image.csv'
    matcher = UniversityNameMatcher(df_csv, test_csv, output_csv)
    matcher.process_data()
    matcher.display_data()
    #step 10 : UniversityDataProcessor
    input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\hashing_image.csv'
    output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\Final_hashing_image.csv'
    processor_time = UniversityDataProcessor_all(input_csv, output_csv)
    processor_time.process_data()  # Process the data
    processor_time.save_to_csv()  # Save the processed data to the output CSV
    
    if uploaded_files:
        flash(f'Files uploaded successfully: {", ".join(uploaded_files)}')
    return redirect(url_for('index'))

@app.route('/view_csv')
def view_csv():
    # Path to the final CSV file
    final_csv_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\Final_hashing_image.csv'

    # Check if the file exists
    if os.path.exists(final_csv_path):
        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(final_csv_path)

        # Render the DataFrame to HTML
        return render_template('view_csv.html', tables=[df.to_html(classes='data', header=True)])
    else:
        flash('The CSV file does not exist.')
        return redirect(url_for('index'))

@app.route('/download_csv')
def download_csv():
    # Path to the final CSV file
    final_csv_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\Final_hashing_image.csv'

    # Check if the file exists
    if os.path.exists(final_csv_path):
        return send_file(final_csv_path, as_attachment=True)
    else:
        flash('The CSV file does not exist.')
        return redirect(url_for('index'))

if __name__ == '__main__':
    # Ensure the upload folder exists
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    app.run(host='0.0.0.0', port=8868, threaded=True, debug=False)
