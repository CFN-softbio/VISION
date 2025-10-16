import requests
import os
from pathlib import Path

def process_pdf(pdf_path, output_dir="processed_files"):
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get the filename without extension
    pdf_name = Path(pdf_path).stem
    
    # GROBID API endpoint
    url = "http://localhost:8070/api/processFulltextDocument"
    
    try:
        # Send PDF file to GROBID
        with open(pdf_path, 'rb') as pdf:
            files = {'input': pdf}
            response = requests.post(url, files=files)
            
        # Check if request was successful
        response.raise_for_status()
        
        # Create output file path
        output_path = os.path.join(output_dir, f"{pdf_name}.xml")
        
        # Save the TEI XML to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(response.text)
            
        print(f"Successfully processed and saved: {output_path}")
        return response.text
    
    except requests.exceptions.RequestException as e:
        print(f"Error processing PDF: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        return None

# Process the PDF
pdf_path = '/home2/common/Chatbot_Papers/AI/Alireza/SciAgents.pdf'
result = process_pdf(pdf_path, output_dir="grobid_output")

# Print the result if needed
if result:
    print("Processing completed successfully")
else:
    print("Processing failed")
