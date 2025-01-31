import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
import os
import shutil
from pdf2image import convert_from_path

class ImageProcessor:
    def __init__(self, poppler_path, input_folder, output_folder):
        """
        Initializes the ImageProcessor with paths for Poppler, input, and output folders.
        """
        self.poppler_path = poppler_path
        self.input_folder = input_folder
        self.output_folder = output_folder
        
        # Ensure output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Define supported image file extensions
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

    def process_pdfs(self):
        """
        Converts PDF files from the input folder to images and saves them in the output folder.
        """
        for filename in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, filename)

            if filename.lower().endswith(".pdf"):
                # Convert the PDF to images
                pages = convert_from_path(file_path, dpi=300, poppler_path=self.poppler_path)

                # Save each page as an image in the output folder
                for i, page in enumerate(pages):
                    output_image_path = os.path.join(self.output_folder, f"{os.path.splitext(filename)[0]}_page_{i+1}.jpg")
                    page.save(output_image_path, 'JPEG')

                print(f"Processed PDF: {filename}")

    def copy_images(self):
        """
        Copies image files from the input folder to the output folder.
        """
        for filename in os.listdir(self.input_folder):
            file_path = os.path.join(self.input_folder, filename)

            if filename.lower().endswith(self.image_extensions):
                # If it's an image file, copy it to the output folder
                output_image_path = os.path.join(self.output_folder, filename)
                shutil.copy(file_path, output_image_path)

                print(f"Copied image: {filename}")

    def process_all_files(self):
        """
        Processes both PDF and image files from the input folder.
        """
        self.process_pdfs()
        self.copy_images()
        print("All PDFs and image files have been processed.")

# Example usage:
poppler_path = r"C:\Program Files\poppler-24.08.0\Library\bin"  # Adjust this path for your system
input_folder = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\upload_files'
output_folder = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\output_images'

image_processor = ImageProcessor(poppler_path, input_folder, output_folder)
image_processor.process_all_files()


import os
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification
from PIL import Image
import pytesseract
import numpy as np
import pandas as pd

class LayoutAwareTextExtractor:
    def __init__(self, model_name="microsoft/layoutlmv3-large", apply_ocr=False):
        # Initialize LayoutLMv3 processor and model from Hugging Face
        self.processor = LayoutLMv3Processor.from_pretrained(model_name, apply_ocr=apply_ocr)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)
    
    def extract_text_with_layout(self, image_path):
        # Load the image using PIL
        pil_image = Image.open(image_path)
        
        # Perform OCR with pytesseract to get bounding boxes and text
        ocr_result = pytesseract.image_to_data(pil_image, output_type=pytesseract.Output.DICT)
        
        # Extract text and bounding boxes
        text = []
        boxes = []
        for i in range(len(ocr_result['text'])):
            if int(ocr_result['conf'][i]) > 0:  # Filter out low-confidence text
                text.append(ocr_result['text'][i])
                boxes.append([ocr_result['left'][i], ocr_result['top'][i], 
                              ocr_result['width'][i], ocr_result['height'][i]])
        
        # Normalize bounding boxes to be within [0, 1000] range
        image_width, image_height = pil_image.size
        boxes = np.array(boxes).reshape(-1, 4)
        boxes = [
            [box[0] / image_width * 1000, box[1] / image_height * 1000,
             (box[0] + box[2]) / image_width * 1000, (box[1] + box[3]) / image_height * 1000]
            for box in boxes
        ]
        
        return text  # Return only the text, for simplicity

    def process_image_for_layoutlm(self, image_path):
        # Extract text and bounding boxes using pytesseract (OCR)
        text, boxes, pil_image = self.extract_text_with_layout(image_path)
        
        # Ensure bounding boxes are in the correct data type for LayoutLMv3
        boxes = torch.tensor(boxes, dtype=torch.long)  # Convert boxes to long type for embedding
        
        # Use LayoutLMv3Processor to process the text and bounding boxes
        encoding = self.processor(pil_image, text, boxes=boxes.tolist(), return_tensors="pt", padding=True, truncation=True)
        
        return encoding

    def extract_layout_aware_text(self, image_path):
        # Process image and get layout-aware input format
        encoding = self.process_image_for_layoutlm(image_path)
        
        # Perform inference (forward pass)
        with torch.no_grad():
            outputs = self.model(**encoding)
        
        # Convert model outputs to text (using argmax to select the highest probability class)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        
        # Map predictions to their labels (we use token classification labels)
        labels = self.processor.tokenizer.convert_ids_to_tokens(predictions[0].tolist())
        
        # Filter out special tokens like [PAD], <s>, </s>, and others
        filtered_labels = [label for label in labels if label not in ["[PAD]", "<s>", "</s>", "[SEP]"]]
        
        # Join the filtered labels to form the final extracted text
        extracted_text = " ".join(filtered_labels)
        
        # Clean up multiple spaces or empty tokens if any
        extracted_text = " ".join(extracted_text.split())
        
        return extracted_text

    def process_folder_images(self, folder_path,output_csv_path):
        # Initialize an empty list to hold the results
        results = []
        
        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            # Check if the file is an image (by extension)
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder_path, filename)
                try:
                    # Extract the text from the image
                    text = self.extract_text_with_layout(image_path)
                    
                    # Append the result to the list (filename, extracted_text)
                    results.append({"filename": filename, "extracted_text": text})
                except Exception as e:
                    # Handle any errors (e.g., corrupted image or OCR issues)
                    print(f"Error processing {filename}: {e}")
                    continue
        
        # Create a DataFrame from the results
        df = pd.DataFrame(results)
        
        # Display the results for debugging
        print("\nExtracted Text for All Images in Folder:")
        print(df)
        df.to_csv(output_csv_path, index=False)
        print(f"Results saved to {output_csv_path}")
        
        return df


# Example usageś
folder_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\output_images'  # Replace with the path to your image folder
output_csv_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_text.csv' 
# Create an instance of the LayoutAwareTextExtractor class
extractor = LayoutAwareTextExtractor()

# Process the images in the folder and store the results in a CSV file
df = extractor.process_folder_images(folder_path, output_csv_path)

# import pandas as pd
# import re

# class TextCleaner:
#     def __init__(self, input_file, output_file):
#         self.input_file = input_file
#         self.output_file = output_file
#         self.df = None

#     # Function to read the CSV file
#     def read_csv(self):
#         self.df = pd.read_csv(self.input_file)

#     # Function to clean the extracted text
#     def clean_extracted_text(self, extracted_text):
#         # Join the list of words into a single string
#         text = " ".join(extracted_text)
        
#         # Remove unwanted spaces, commas, special characters, and digits
#         text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
#         text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
#         text = text.strip()                  # Remove leading and trailing spaces
        
#         return text

#     # Function to apply cleaning process
#     def apply_cleaning(self):
#         if self.df is not None:
#             # Apply cleaning function to the 'extracted_text' column
#             self.df['cleaned_text'] = self.df['extracted_text'].apply(self.clean_extracted_text)
#             # Convert the cleaned text to lowercase
#             self.df['cleaned_text'] = self.df['cleaned_text'].str.lower()

#     # Function to save the cleaned data to a new CSV file
#     def save_to_csv(self):
#         if self.df is not None:
#             self.df.to_csv(self.output_file, index=False)

#     # Function to run the entire cleaning process
#     def process(self):
#         self.read_csv()      # Read the input CSV
#         self.apply_cleaning() # Clean the text data
#         self.save_to_csv()    # Save the cleaned data to output CSV

# # Usage example
# input_file = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_text.csv'   # Replace with your input CSV file path
# output_file = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext.csv'   # Replace with your desired output file path

# cleaner = TextCleaner(input_file, output_file)
# cleaner.process()


import pandas as pd
import re

class TextCleaner:
    def __init__(self, input_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param input_csv: Path to the input CSV file
        :param output_csv: Path to the output CSV file
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Load the input CSV into a DataFrame
    
    def clean_extracted_text(self, extracted_text):
        """
        Cleans the extracted text by removing unwanted spaces, punctuation, 
        digits, and other non-alphanumeric characters.

        :param extracted_text: The raw text to clean (either a string or list of words)
        :return: Cleaned text as a string
        """
        # Join the list of words into a single string if it's a list of words
        if isinstance(extracted_text, list):
            text = " ".join(extracted_text)
        else:
            text = extracted_text
        
        # Remove unwanted spaces, commas, special characters, and digits
        text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation and special characters
        text = re.sub(r'\s+', ' ', text)     # Replace multiple spaces with a single space
        text = text.strip()                  # Remove leading and trailing spaces
        
        return text
    
    def process_data(self):
        """
        Processes the DataFrame by applying the cleaning function on the 'extracted_text'
        column and storing the result in the 'cleaned_text' column.
        """
        # Apply cleaning function to the 'extracted_text' column
        self.df['cleaned_text'] = self.df['extracted_text'].apply(self.clean_extracted_text)
    
    def save_to_csv(self):
        """
        Saves the updated DataFrame with the cleaned text to the output CSV file.
        """
        self.df.to_csv(self.output_csv, index=False)
    
    def display_data(self):
        """
        Displays the relevant columns (filename, cleaned_text, and extracted_text) for verification.
        """
        print(self.df[['filename', 'cleaned_text', 'extracted_text']])

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_text.csv'   # Provide the path to your input CSV file
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext.csv' # Provide the path for the output CSV file

# Initialize the TextCleaner class
text_cleaner = TextCleaner(input_csv, output_csv)

# Process the data to clean the text
text_cleaner.process_data()

# Display the cleaned data
text_cleaner.display_data()

# Save the cleaned data to the output CSV
text_cleaner.save_to_csv()



import pandas as pd
import re

class UniversityExtractor:
    def __init__(self, input_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param input_csv: Path to the input CSV file
        :param output_csv: Path to the output CSV file
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Load the input CSV into a DataFrame
    
    def extract_university_name(self, text):
        """
        Extracts university name based on keywords and first few words.

        :param text: The cleaned text from which the university name is extracted
        :return: Extracted university name or first few words if no match is found
        """
        keywords = ["university", "Examination Council", "Council", "State Board", "Education", "Examinations Council", "Examinations"]
        
        # Create a pattern to search for the first occurrence of any keyword and capture words before and after
        pattern = r'(\S+\s+\S+\s+\S+\s+)(\b(?:' + '|'.join(keywords) + r')\b)(\s+\S+\s+\S+\s+\S+)'
        
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            # Extract the words before and after the first occurrence of the keyword
            before_keyword = match.group(1).strip()
            keyword = match.group(2).strip()
            after_keyword = match.group(3).strip()
            
            # Combine them to return the university name
            university_name = f"{before_keyword} {keyword} {after_keyword}"
        else:
            # If no match found based on keywords, use the first few words as a fallback
            university_name = ' '.join(text.split()[:3])  # Take first 3 words
        
        return university_name
    
    def process_data(self):
        """
        Processes the DataFrame by applying the extract_university_name function to the cleaned_text column.
        Also stores the first few words in the header_uniname column.
        """
        # Apply the extract_university_name function to the cleaned_text column
        self.df['university_name'] = self.df['cleaned_text'].apply(self.extract_university_name)
        
        # Store the first few words separately in the header_uniname column (first 3 words)
        self.df['header_uniname'] = self.df['cleaned_text'].apply(lambda x: ' '.join(x.split()[:3]))  # First 3 words
    
    def save_to_csv(self):
        """
        Saves the updated DataFrame to the output CSV file.
        """
        self.df.to_csv(self.output_csv, index=False)
    
    def display_data(self):
        """
        Displays the DataFrame with the extracted university names and header_uniname for review.
        """
        print(self.df[['filename', 'cleaned_text', 'university_name', 'header_uniname']])

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext.csv'  # Path to the input CSV file
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext1.csv'  # Path for the output CSV file

# Initialize the UniversityExtractor class
university_extractor = UniversityExtractor(input_csv, output_csv)

# Process the data to extract university names
university_extractor.process_data()

# Display the updated data for review
university_extractor.display_data()

# Save the results to the output CSV
university_extractor.save_to_csv()




import pandas as pd
import re

class uniTextCleaner:
    def __init__(self, input_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param input_csv: Path to the input CSV file
        :param output_csv: Path to the output CSV file
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Load the input CSV into a DataFrame
    
    def clean_text(self, text):
        """
        Cleans the university name and header_uniname columns by converting to lowercase, 
        removing unwanted characters, and eliminating extra spaces.

        :param text: The text to be cleaned
        :return: Cleaned text
        """
        # Convert to lowercase for uniformity
        text = text.lower()
        
        # Remove unwanted characters, numbers, and special symbols
        text = re.sub(r'[^a-z\s]', '', text)  # Keep only letters and spaces
        
        # Remove extra spaces
        text = re.sub(r'\s+', ' ', text).strip()  # Replace multiple spaces with a single space
        
        return text

    def process_data(self):
        """
        Processes the DataFrame by applying the clean_text function to the university_name and header_uniname columns.
        """
        # Apply the clean_text function to the university_name and header_uniname columns
        self.df['university_name'] = self.df['university_name'].apply(self.clean_text)
        self.df['header_uniname'] = self.df['header_uniname'].apply(self.clean_text)
    
    def save_to_csv(self):
        """
        Saves the cleaned DataFrame to the output CSV file.
        """
        self.df.to_csv(self.output_csv, index=False)
    
    def display_data(self):
        """
        Displays the cleaned DataFrame for review.
        """
        print(self.df[['filename', 'university_name', 'header_uniname']])

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext1.csv'  # Path to the input CSV file
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext2.csv'  # Path for the output CSV file

# Initialize the TextCleaner class
text_cleaner1 = uniTextCleaner(input_csv, output_csv)

# Process the data to clean the text
text_cleaner1.process_data()

# Display the cleaned data for review
text_cleaner1.display_data()

# Save the results to the output CSV
text_cleaner1.save_to_csv()


import pandas as pd
import re

class DegreeExtractor:
    def __init__(self, input_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param input_csv: Path to the input CSV file
        :param output_csv: Path to the output CSV file
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Load the input CSV into a DataFrame

    def find_degree(self, text):
        """
        Extracts degrees based on specific keywords and their context.

        :param text: The text from which the degree is extracted
        :return: Extracted degree or None if no match is found
        """
        keywords = [
            ('degree of', 3, 'next', False),  # Take next 3 words, exclude "degree of"
            ('Bachelor of', 3, 'next', True),  # Take next 3 words, include "Bachelor of"
            ('Master of', 3, 'next', True),
            ('Bachelor', 3, 'next', True),     # Take next 3 words (from "Bachelor")
            ('Surgery', 3, 'previous', False), # Take previous 3 words (before "Surgery")
            ('M S', 3, 'next', True),          # Take next 3 words (from "M S")
            ('MBBS', 0, 'exact', True),        # Just return "MBBS"
            ('MB BS', 0, 'exact', True),
            ('Major', 3, 'next', False),
            ('Programme', 3, 'next', False),
            ('Course', 3, 'next', False)
        ]
        
        # Iterate over each keyword pattern
        for keyword, count, direction, include_keyword in keywords:
            # Perform case-insensitive search for the keyword
            match = re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE)
            
            if match:
                # Based on direction and count, extract the required words
                words = text.split()
                idx = match.start()
                # Find the position of the keyword in words
                keyword_position = len(re.findall(r'\b\w+\b', text[:idx]))  # This gives the word index
                
                if direction == 'next':
                    # Get the next 'count' words
                    if include_keyword:
                        # Include the keyword and next words
                        result = ' '.join(words[keyword_position: keyword_position + 1 + count])
                    else:
                        # Exclude the keyword and take next words
                        result = ' '.join(words[keyword_position + len(keyword.split()): keyword_position + len(keyword.split()) + count])
                elif direction == 'previous':
                    # Get the previous 'count' words
                    result = ' '.join(words[keyword_position - count: keyword_position])
                elif direction == 'exact':
                    # Exact match for "MBBS"
                    result = words[keyword_position]
                
                return result
        
        return None  # If no keyword matches

    def process_data(self):
        """
        Processes the DataFrame by applying the find_degree function to the cleaned_text column.
        """
        # Apply the find_degree function to the cleaned_text column
        self.df['degree'] = self.df['cleaned_text'].apply(self.find_degree)
    
    def save_to_csv(self):
        """
        Saves the updated DataFrame to the output CSV file.
        """
        self.df.to_csv(self.output_csv, index=False)
    
    def display_data(self):
        """
        Displays the DataFrame with the extracted degree for review.
        """
        print(self.df[['filename', 'cleaned_text', 'degree']])

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext2.csv'  # Path to the input CSV file
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext3.csv'  # Path for the output CSV file

# Initialize the DegreeExtractor class
degree_extractor = DegreeExtractor(input_csv, output_csv)

# Process the data to extract degrees
degree_extractor.process_data()

# Display the updated data for review
degree_extractor.display_data()

# Save the results to the output CSV
degree_extractor.save_to_csv()



import pandas as pd
import re

class YearExtractor:
    def __init__(self, input_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param input_csv: Path to the input CSV file
        :param output_csv: Path to the output CSV file
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Load the input CSV into a DataFrame

    def extract_year(self, text):
        """
        Extracts the first year (4-digit number) from the text.

        :param text: The text from which the year is extracted
        :return: The first year found or None if no year is found
        """
        match = re.search(r'\b(\d{4})\b', text)
        if match:
            return match.group(1)  # Return the first 4-digit number found
        return None  # Return None if no year is found

    def extract_years(self, text):
        """
        Extracts all years (4-digit numbers) from the text.

        :param text: The text from which the years are extracted
        :return: A list of years found or None if no years are found
        """
        years = re.findall(r'\b(\d{4})\b', text)
        return years if years else None  # Return the list of years or None if no years found

    def process_data(self):
        """
        Processes the DataFrame by applying the extract_year and extract_years functions
        to the cleaned_text column.
        """
        self.df['year'] = self.df['cleaned_text'].apply(self.extract_year)  # First year found
        self.df['list_year'] = self.df['cleaned_text'].apply(self.extract_years)  # All years found
    
    def save_to_csv(self):
        """
        Saves the updated DataFrame to the output CSV file.
        """
        self.df.to_csv(self.output_csv, index=False)
    
    def display_data(self):
        """
        Displays the DataFrame with the extracted years for review.
        """
        print(self.df[['filename', 'cleaned_text', 'year', 'list_year']])

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext3.csv'  # Path to the input CSV file
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext4.csv'  # Path for the output CSV file

# Initialize the YearExtractor class
year_extractor = YearExtractor(input_csv, output_csv)

# Process the data to extract years
year_extractor.process_data()

# Display the updated data for review
year_extractor.display_data()

# Save the results to the output CSV
year_extractor.save_to_csv()



import pandas as pd
import spacy

class UniversityDataProcessor:
    def __init__(self, input_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param input_csv: Path to the input CSV file
        :param output_csv: Path to the output CSV file
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Load the input CSV into a DataFrame
        self.nlp = spacy.load("en_core_web_sm")  # Load spaCy pre-trained model

    def extract_university_name(self, text):
        """
        Extracts the university name using spaCy NER (Named Entity Recognition).

        :param text: The text from which the university name is extracted
        :return: The extracted university name or None if not found
        """
        if not isinstance(text, str):
            text = str(text)  # Convert non-string inputs (e.g., NaN, float) to a string
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ == "ORG":  # "ORG" is the label for organizations (including universities)
                return ent.text
        return None

    def process_data(self):
        """
        Processes the DataFrame by performing several transformations in sequence.
        """
        # Step 1: Select relevant columns from the DataFrame
        self.df = self.df[['filename', 'degree', 'year', 'list_year', 'university_name', 'header_uniname']]

        # Step 2: Apply extract_university_name function to the relevant columns
        self.df['extracted_universityname'] = self.df['university_name'].apply(self.extract_university_name)
        self.df['extracted_headername'] = self.df['header_uniname'].apply(self.extract_university_name)
        self.df['extracted_degree'] = self.df['degree'].apply(self.extract_university_name)

        # Step 3: Convert all string columns to lowercase
        self.df = self.df.applymap(lambda x: x.lower() if isinstance(x, str) else x)

        # Step 4: Convert 'year' to integer if it’s a float (e.g., 2020.0 becomes 2020)
        self.df['year'] = self.df['year'].apply(lambda x: int(x) if isinstance(x, float) and not pd.isna(x) else x)

        # Step 5: Fill NaN values in 'year' with an empty string
        self.df['year'] = self.df['year'].fillna('')

        # Step 6: Ensure 'year' is a single value (e.g., if it is a list, take the first value)
        self.df['year'] = self.df['year'].apply(lambda x: x[0] if isinstance(x, list) else x)

        # Step 7: If 'extracted_universityname' is NaN or empty, replace it with 'extracted_headername'
        self.df['extracted_universityname'] = self.df['extracted_universityname'].fillna(self.df['extracted_headername'])

        # Step 8: Drop the 'extracted_headername' column (no longer needed)
        self.df.drop(columns=['extracted_headername'], inplace=True)

    def save_to_csv(self):
        """
        Saves the processed DataFrame to the output CSV file.
        """
        self.df.to_csv(self.output_csv, index=False)

    def display_data(self):
        """
        Displays the DataFrame with the extracted and processed data for review.
        """
        print(self.df[['filename', 'degree', 'year', 'list_year', 'extracted_universityname', 'extracted_degree']])

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext4.csv'  # Path to the input CSV file
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext5.csv'    # Path for the output CSV file

# Initialize the UniversityDataProcessor class
processor1 = UniversityDataProcessor(input_csv, output_csv)

# Process the data
processor1.process_data()

# Display the updated data for review
processor1.display_data()

# Save the results to the output CSV
processor1.save_to_csv()


import pandas as pd
from fuzzywuzzy import fuzz

class UniversityNameMatcher:
    def __init__(self, df_csv, test_csv, output_csv):
        """
        Initializes the class with input and output CSV file paths.

        :param df_csv: Path to the main input CSV file (df)
        :param test_csv: Path to the comparison CSV file (csv_test)
        :param output_csv: Path to the output CSV file after processing
        """
        self.df = pd.read_csv(df_csv)  # Load the main input CSV into a DataFrame
        self.csv_test = pd.read_excel(test_csv)  # Load the test DataFrame (csv_test)
        self.output_csv = output_csv  # Path for the output CSV file

        # Convert university names to lowercase for case-insensitive comparison
        self.df['university_name'] = self.df['university_name'].str.lower()
        self.csv_test['uni_name'] = self.csv_test['uni_name'].str.lower()

        # Set a threshold for fuzzy matching (higher values = stricter match)
        self.threshold = 80

    def get_best_match(self, university_name):
        """
        Finds the best match for a given university name from the test DataFrame.
        
        :param university_name: The name of the university to find the best match for
        :return: The best match university name or an empty string if no match above threshold
        """
        if not isinstance(university_name, str):  # Skip non-string values (e.g., NaN or float)
            return ""  # Return empty string if not a valid string
        
        best_match = None
        best_score = 0
        for uni_name in self.csv_test['uni_name']:
            score = fuzz.partial_ratio(university_name, uni_name)  # Calculate similarity score
            if score > best_score:
                best_score = score
                best_match = uni_name
        
        # If the best score is above the threshold, return the match
        if best_score >= self.threshold:
            return best_match
        else:
            return ""  # Return empty string if no good match found

    def process_data(self):
        """
        Processes the DataFrame by applying fuzzy matching to the relevant columns.
        """
        # Step 1: Apply the fuzzy matching function to both university_name and header_uniname columns
        self.df['university_name_test'] = self.df['university_name'].apply(self.get_best_match)
        self.df['header_uniname_test'] = self.df['header_uniname'].apply(self.get_best_match)

        # Step 2: Replace empty strings in 'university_name_test' with corresponding values from 'header_uniname_test'
        self.df['university_name_test'] = self.df.apply(
            lambda row: row['header_uniname_test'] if row['university_name_test'] == "" else row['university_name_test'],
            axis=1
        )

        # Step 3: Replace occurrences of 'arsi university' with 'header_uniname_test' value
        self.df['university_name_test'] = self.df.apply(
            lambda row: row['header_uniname_test'] if 'arsi university' in row['university_name_test'].lower() else row['university_name_test'],
            axis=1
        )
        # # create a dataframe for my validation 
        # seld.df.to_csv(self.university_name_campre.csv,index=False)

        # Final DataFrame for validation (columns of interest)
        self.df = self.df[['filename', 'university_name_test', 'degree', 'year', 'list_year']]

        # Step 4: Create a new column 'university_name_hash' by hashing 'university_name_test'
        self.df['university_name_hash'] = self.df['university_name_test'].apply(lambda x: hash(x))
        print("Final_dataframe:",self.df)
        self.df = self.df[['filename', 'university_name_test', 'degree', 'year', 'list_year','university_name_hash']]

        # Step 5: Prepare the final DataFrame for validation and save to CSV
        self.df.to_csv(self.output_csv, index=False)

    
    def display_data(self):
        """
        Displays the final DataFrame after processing for validation.
        """
        print(self.df)


# Example usage:

# Define file paths
df_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\extracted_cleantext5.csv'  # Path to the main input CSV
test_csv = r'C:\Users\inc3061\Downloads\university_keywords.xlsx'  # Path to the comparison CSV (csv_test)
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\hashing_image.csv'  # Path to store the result

# Create an instance of the UniversityNameMatcher class
matcher = UniversityNameMatcher(df_csv, test_csv, output_csv)

# Process the data (apply fuzzy matching, transform columns)
matcher.process_data()

# Optionally, display the data for review
matcher.display_data()


import pandas as pd
import re

class UniversityDataProcessor_all:
    def __init__(self, input_csv, output_csv):
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = pd.read_csv(input_csv)  # Read the input CSV file

    def clean_degree(self, degree_text):
        # Check if the value is not NaN and is a string
        if isinstance(degree_text, str):
            # Convert to lower case to handle case insensitivity
            degree_text = degree_text.lower()

            # Standardize common typos
            degree_text = degree_text.replace("bachglor", "bachelor")
            degree_text = degree_text.replace("mb", "mbbs")

            # Regular expressions to match degree patterns
            degree_patterns = [
                r"(bachelor of [a-z\s]+)",  # Matches Bachelor degrees
                r"(master of [a-z\s]+)",     # Matches Master degrees
                r"(doctor of [a-z\s]+)",     # Matches Doctor degrees
                r"mbbs",                     # Matches MBBS
                r"(bachelor of science)",    # Matches Bachelor of Science
                r"(master of science)",      # Matches Master of Science
                r"(bachelor of arts)",       # Matches Bachelor of Arts
                r"(bachelor of technology)", # Matches Bachelor of Technology
            ]
            
            # Try matching the degree patterns and return the first match
            for pattern in degree_patterns:
                match = re.search(pattern, degree_text)
                if match:
                    return match.group(0).title()  # Return matched degree with proper title case
            
            # If no match found, return a generic category (e.g., "Unknown degree")
            return "Unknown degree"
        else:
            # If the value is NaN or not a string, return "Unknown degree"
            return "Unknown degree"
        
    def process_data(self):
        # Apply the function to clean the 'degree' column
        self.df['cleaned_degree'] = self.df['degree'].apply(self.clean_degree)

        # View the cleaned DataFrame (can be skipped in the class, just for reference)
        print(self.df[['degree', 'cleaned_degree']])
        print(self.df.columns)
        # Rename columns as required
        self.df.rename(columns={'university_name_test':'university_name',
                               'cleaned_degree':'Degrees',
                               'year':'completion_year',
                               'university_name_hash':'Hashing_value',
                               'list_year':'years_list'}, inplace=True)
        print(self.df.columns)

        # Replace 'unknown degree' in the 'Degrees' column with the value from the 'Degree' column
        self.df['Degrees'] = self.df.apply(lambda row: row['degree'] if row['Degrees'] == 'Unknown degree' else row['Degrees'], axis=1)
        

        # Select the necessary columns
        self.df = self.df[['filename','university_name','Degrees','completion_year','Hashing_value','years_list']]

        # Fill NaN values with empty strings
        self.df = self.df.fillna('')
        
        # Convert all columns to string type
        self.df = self.df.astype(str)

        # Convert 'completion_year' to numeric, coercing errors to NaN, and then replace NaN with 0
        self.df['completion_year'] = pd.to_numeric(self.df['completion_year'], errors='coerce')
        self.df['completion_year'] = self.df['completion_year'].fillna(0).astype(int)

        # Ensure all columns are still treated as strings
        self.df = self.df.astype(str)

    def save_to_csv(self):
        # Write the final DataFrame to a CSV file
        self.df.to_csv(self.output_csv, index=False)
        print(f"Processed data saved to {self.output_csv}")

# Example usage:
input_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\hashing_image.csv'  # Input CSV file path
output_csv = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Output\Final_hashing_image.csv'  # Output CSV file path

# Create an instance of the class and process the data
processor_time = UniversityDataProcessor_all(input_csv, output_csv)
processor_time.process_data()  # Process the data
processor_time.save_to_csv()  # Save the processed data to the output CSV


import cv2
import os
import time

class LogoExtractor:
    def __init__(self, input_folder, output_folder, target_width=500, target_height=500):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.target_width = target_width
        self.target_height = target_height
        
        # Ensure the output folder exists
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
        
        # Define regions to check for the logo
        self.regions = {
            'top': (5, 0, 0, 0),               # Top 20%
            'bottom': (0, 0, 0, 0),  # Bottom 20%
            'left': (0, 0, 0, 0),             # Left 20%
            'right': (0, 0, 0, 0),  # Right 20%
            'center': (0, 0, 0, 0), # Center area
        }

    # Function to process the image in a specific region
    def process_region(self, image, x, y, w, h, padding_top=50, padding_bottom=50):
        # Extract region of interest (ROI) based on the region coordinates
        roi = image[y:y+h, x:x+w]
        # Apply grayscale and threshold to the ROI
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
        _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY_INV)

        # Find contours in the thresholded ROI
        roi_contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Loop through contours in the region
        for contour in roi_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 100 < w < 400 and 100 < h < 400:  # Adjust size condition for logo
                # Add padding to the bottom of the extracted ROI
                y2 = min(y + h + padding_bottom, roi.shape[0])  # Ensure the padding doesn't go out of bounds
                return roi[y:y2, x:x+w]
        return None

    # Resize the extracted logo
    def resize_image(self, image):
        return cv2.resize(image, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)

    # Function to process all images in the input folder
    def process_images(self):
        # Start the timer
        start_time = time.time()

        # Process each image in the input folder
        for filename in os.listdir(self.input_folder):
            # Check if the file is an image (you can adjust this condition based on your image types)
            if filename.endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(self.input_folder, filename)
                print(f"Processing image: {image_path}")
                
                # Read the image
                image = cv2.imread(image_path)
                
                # Check if the image was successfully loaded
                if image is None:
                    print(f"Error: Image '{filename}' could not be loaded. Skipping.")
                    continue
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Apply GaussianBlur to reduce noise
                gray = cv2.GaussianBlur(gray, (5, 5), 0)

                # Apply thresholding to isolate the highlighted logo area
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

                # Get the dimensions of the image
                height, width = image.shape[:2]
                
                # Update regions based on the image size
                self.regions = {
                    'top': (5, 0, width, height // 5),               # Top 1/5
                    'bottom': (0, height - height // 5, width, height),  # Bottom 1/5
                    'left': (0, 0, width // 5, height),             # Left 1/5
                    'right': (width - width // 5, 0, width, height),  # Right 1/5
                    'center': (width // 4, height // 4, width // 2, height // 2)  # Center area
                }

                # Iterate through the regions and check for the logo
                detected_in_any_region = False
                for region_name, (x, y, w, h) in self.regions.items():
                    print(f"Checking {region_name} region...")
                    roi_logo = self.process_region(image, x, y, w, h, padding_bottom=70)  # Adjust padding as needed
                    if roi_logo is not None:
                        detected_in_any_region = True

                        # Resize the detected logo to desired dimensions (e.g., 500x500)
                        resized_logo = self.resize_image(roi_logo)

                        # Define output path including the filename and region
                        output_path = os.path.join(self.output_folder, f"{os.path.splitext(filename)[0]}_logo_{region_name}.jpg")
                        
                        # Save the resized logo to the specified output path
                        cv2.imwrite(output_path, resized_logo)
                        print(f"Logo detected in the {region_name} region and saved as '{output_path}'")
                
                # If logo was not detected in any region
                if not detected_in_any_region:
                    print(f"No logo detected in the specified regions for '{filename}'.")

        # End the timer and calculate the execution time
        end_time = time.time()
        execution_time = end_time - start_time

        # Print the execution time
        print(f"Execution time: {execution_time:.4f} seconds")

# Example usage:
input_folder = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\output_images'
output_folder = r'C:\Users\inc3061\Documents\Process_data'

# Create an instance of the LogoExtractor class
logo_extractor = LogoExtractor(input_folder, output_folder)

# Call the process_images method to start the processing
logo_extractor.process_images()



import cv2
import os
import csv

class ImageComparer:
    def __init__(self, folder1, folder2, output_csv_path):
        self.folder1 = folder1
        self.folder2 = folder2
        self.output_csv_path = output_csv_path

        # Get all image files in both folders
        self.image1_files = [f for f in os.listdir(self.folder1) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        self.image2_files = [f for f in os.listdir(self.folder2) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
    # Function to calculate and compare histograms
    def compare_images(self, image1_path, image2_path):
        # Load images
        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            print(f"Error loading images: {image1_path} or {image2_path}")
            return None

        # Calculate histograms for each image
        hist_img1 = cv2.calcHist([image1], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img1[255, 255, 255] = 0  # Ignore all white pixels
        cv2.normalize(hist_img1, hist_img1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        hist_img2 = cv2.calcHist([image2], [0, 1, 2], None, [256, 256, 256], [0, 256, 0, 256, 0, 256])
        hist_img2[255, 255, 255] = 0  # Ignore all white pixels
        cv2.normalize(hist_img2, hist_img2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

        # Compare the histograms and return the similarity score
        metric_val = cv2.compareHist(hist_img1, hist_img2, cv2.HISTCMP_CORREL)
        return round(metric_val, 2)

    # Function to process all image pairs and save results to a CSV file
    def compare_all_images(self):
        # Open a CSV file to write the results
        with open(self.output_csv_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            # Write the header row
            writer.writerow(["Image1 Filename", "Image2 Filename", "Similarity Score"])

            # Compare every image in folder1 with every image in folder2
            for image1_name in self.image1_files:
                image1_path = os.path.join(self.folder1, image1_name)

                for image2_name in self.image2_files:
                    image2_path = os.path.join(self.folder2, image2_name)

                    # Get the similarity score for this pair of images
                    similarity_score = self.compare_images(image1_path, image2_path)

                    if similarity_score is not None:
                        # Write the result to the CSV file
                        writer.writerow([image1_name, image2_name, similarity_score])

        print(f"Results have been saved to '{self.output_csv_path}'.")

# Example usage:
folder1 = r'C:\Users\inc3061\Documents\Process_data'  # Folder with images to compare
folder2 = r'C:\Users\inc3061\Documents\Templete_Logo'    # Folder with images to compare against
output_csv_path = r'C:\Users\inc3061\OneDrive - Texila American University\Python\Image Processing\imgenv\Similarity_score\image_similarity_levelall.csv'

# Create an instance of the ImageComparer class
image_comparer = ImageComparer(folder1, folder2, output_csv_path)

# Call the compare_all_images method to start the comparison and save the results
image_comparer.compare_all_images()























