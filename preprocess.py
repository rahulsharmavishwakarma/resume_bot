import os
import json
from PyPDF2 import PdfReader

# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to parse resume into structured data
def parse_resume(text):
    resume_data = {}

    # Extract different sections
    if "Professional Summary" in text:
        summary_start = text.find("Professional Summary")
        skills_start = text.find("Skills", summary_start)
        resume_data["professional_summary"] = text[summary_start:skills_start].strip()

    if "Skills" in text:
        skills_start = text.find("Skills")
        experience_start = text.find("Work History", skills_start)
        if experience_start == -1:
            experience_start = text.find("Experience", skills_start)  # Check for 'Experience' as a fallback
        resume_data["skills"] = text[skills_start:experience_start].strip()

    # Handle both 'Work History' and 'Experience' labels
    if "Work History" in text or "Experience" in text:
        # Find whichever comes first in the text
        work_history_start = text.find("Work History")
        experience_start = text.find("Experience")
        
        if work_history_start == -1:  # Only 'Experience' exists
            work_history_start = experience_start
        elif experience_start != -1:  # Both exist, take the first occurrence
            work_history_start = min(work_history_start, experience_start)
        
        education_start = text.find("Education", work_history_start)
        resume_data["work_history"] = text[work_history_start:education_start].strip()

    if "Education" in text:
        education_start = text.find("Education")
        resume_data["education"] = text[education_start:].strip()

    return resume_data

# Directory path where profession folders are located
base_dir = "/teamspace/studios/this_studio/data"  # Change this to your directory path

# List to store parsed resumes
parsed_resumes = []

# Walk through the base directory and process resumes in each profession folder
for profession in os.listdir(base_dir):
    profession_path = os.path.join(base_dir, profession)
    
    # Check if it's a directory (profession folder)
    if os.path.isdir(profession_path):
        for resume_file in os.listdir(profession_path):
            resume_path = os.path.join(profession_path, resume_file)
            
            # Process only PDF files
            if resume_file.endswith(".pdf"):
                try:
                    text = extract_text_from_pdf(resume_path)
                    parsed_resume = parse_resume(text)
                    parsed_resume["profession"] = profession  # Add profession to resume data
                    parsed_resume["file_name"] = resume_file   # Add the file name for reference
                    parsed_resumes.append(parsed_resume)
                except Exception as e:
                    print(f"Error processing {resume_file}: {e}")

# Save extracted resume data to a single JSON file
output_file = "data.json"
with open(output_file, "w") as json_file:
    json.dump(parsed_resumes, json_file, indent=4)

print(f"All resume data saved to {output_file}")
