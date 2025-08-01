import logging
import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
import fitz  # PyMuPDF
import pdfplumber

logger = logging.getLogger(__name__)

class PDFTextExtractor:
    """Extracts text from PDF documents with metadata"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the PDF text extractor"""
        self.config = config
        self.pdf_config = config.get("pdf_extraction", {})
        self.input_dir = Path(self.pdf_config.get("input_dir", "data/raw/HandbookforEVChargingInfrastructureImplementation081221.pdf"))
        self.output_dir = Path(self.pdf_config.get("output_dir", "data/extracted/"))
        self.settings = self.pdf_config.get("settings", {})
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"PDF extractor initialized with input: {self.input_dir}, output: {self.output_dir}")
    
    def extract_all_pdfs(self) -> bool:
        """Extract text from all PDFs in the input directory"""
        pdf_files = list(self.input_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.input_dir}")
            return False
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        extracted_data = []
        for pdf_file in pdf_files:
            try:
                extracted = self.extract_pdf_text(pdf_file)
                if extracted:
                    extracted_data.append(extracted)
                    logger.info(f"Successfully extracted text from {pdf_file.name}")
            except Exception as e:
                logger.error(f"Failed to extract text from {pdf_file.name}: {e}")
        
        # Save extracted data
        if extracted_data:
            self.save_extracted_data(extracted_data)
            logger.info(f"Extracted text from {len(extracted_data)} PDF files")
            return True
        
        return False
    
    def extract_pdf_text(self, pdf_path: Path) -> Optional[Dict[str, Any]]:
        """Extract text from a single PDF file"""
        try:
            # Extract basic metadata
            metadata = self.extract_metadata(pdf_path)
            
            # Extract text with layout preservation
            text_content = self.extract_text_with_layout(pdf_path)
            
            if not text_content:
                logger.warning(f"No text content extracted from {pdf_path.name}")
                return None
            
            # Create extraction result
            result = {
                "source_file": pdf_path.name,
                "source_path": str(pdf_path),
                "metadata": metadata,
                "text_content": text_content,
                "extraction_settings": self.settings,
                "extraction_timestamp": self.get_timestamp()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error extracting text from {pdf_path}: {e}")
            return None
    
    def extract_metadata(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract metadata from PDF file"""
        try:
            with fitz.open(pdf_path) as doc:
                metadata = {
                    "title": doc.metadata.get("title", ""),
                    "author": doc.metadata.get("author", ""),
                    "subject": doc.metadata.get("subject", ""),
                    "creator": doc.metadata.get("creator", ""),
                    "producer": doc.metadata.get("producer", ""),
                    "creation_date": doc.metadata.get("creationDate", ""),
                    "modification_date": doc.metadata.get("modDate", ""),
                    "page_count": len(doc),
                    "file_size": pdf_path.stat().st_size
                }
            return metadata
        except Exception as e:
            logger.error(f"Error extracting metadata from {pdf_path}: {e}")
            return {}
    
    def extract_text_with_layout(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract text with layout preservation"""
        try:
            text_content = {
                "full_text": "",
                "pages": [],
                "tables": [],
                "images": []
            }
            
            # Extract using PyMuPDF for main text
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    
                    # Extract text blocks with position information
                    blocks = page.get_text("dict")
                    
                    page_content = {
                        "page_number": page_num + 1,
                        "text_blocks": [],
                        "page_text": ""
                    }
                    
                    for block in blocks.get("blocks", []):
                        if "lines" in block:
                            for line in block["lines"]:
                                line_text = ""
                                for span in line.get("spans", []):
                                    line_text += span.get("text", "")
                                
                                if line_text.strip():
                                    page_content["text_blocks"].append({
                                        "text": line_text.strip(),
                                        "bbox": line.get("bbox", []),
                                        "font": span.get("font", ""),
                                        "size": span.get("size", 0)
                                    })
                                    page_content["page_text"] += line_text + "\n"
                    
                    text_content["pages"].append(page_content)
                    text_content["full_text"] += page_content["page_text"] + "\n"
            
            # Extract tables using pdfplumber if enabled
            if self.settings.get("extract_tables", True):
                text_content["tables"] = self.extract_tables(pdf_path)
            
            # Extract image information if enabled
            if self.settings.get("extract_images", False):
                text_content["images"] = self.extract_image_info(pdf_path)
            
            return text_content
            
        except Exception as e:
            logger.error(f"Error extracting text with layout from {pdf_path}: {e}")
            return {"full_text": "", "pages": [], "tables": [], "images": []}
    
    def extract_tables(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract tables from PDF using pdfplumber"""
        tables = []
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    page_tables = page.extract_tables()
                    
                    for table_num, table in enumerate(page_tables):
                        if table and any(any(cell for cell in row) for row in table):
                            tables.append({
                                "page_number": page_num + 1,
                                "table_number": table_num + 1,
                                "data": table,
                                "bbox": page.find_tables()[table_num].bbox if page.find_tables() else None
                            })
        except Exception as e:
            logger.error(f"Error extracting tables from {pdf_path}: {e}")
        
        return tables
    
    def extract_image_info(self, pdf_path: Path) -> List[Dict[str, Any]]:
        """Extract image information from PDF"""
        images = []
        
        try:
            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]
                    image_list = page.get_images()
                    
                    for img_index, img in enumerate(image_list):
                        images.append({
                            "page_number": page_num + 1,
                            "image_number": img_index + 1,
                            "bbox": img[0:4],
                            "width": img[2] - img[0],
                            "height": img[3] - img[1]
                        })
        except Exception as e:
            logger.error(f"Error extracting image info from {pdf_path}: {e}")
        
        return images
    
    def save_extracted_data(self, extracted_data: List[Dict[str, Any]]):
        """Save extracted data to output directory"""
        try:
            # Save individual files
            for data in extracted_data:
                source_name = Path(data["source_file"]).stem
                output_file = self.output_dir / f"{source_name}_extracted.json"
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save combined data
            combined_file = self.output_dir / "extracted_data.json"
            with open(combined_file, 'w', encoding='utf-8') as f:
                json.dump(extracted_data, f, indent=2, ensure_ascii=False)
            
            # Save full text
            full_text_file = self.output_dir / "full_text.txt"
            with open(full_text_file, 'w', encoding='utf-8') as f:
                for data in extracted_data:
                    f.write(f"=== {data['source_file']} ===\n")
                    f.write(data['text_content']['full_text'])
                    f.write("\n\n")
            
            logger.info(f"Saved extracted data to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving extracted data: {e}")
    
    def get_timestamp(self) -> str:
        """Get current timestamp string"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def validate_extraction(self, extracted_data: List[Dict[str, Any]]) -> bool:
        """Validate extracted data quality"""
        if not extracted_data:
            return False
        
        for data in extracted_data:
            text_content = data.get("text_content", {})
            full_text = text_content.get("full_text", "")
            
            # Check minimum text length
            if len(full_text.strip()) < self.settings.get("min_text_length", 50):
                logger.warning(f"Extracted text too short for {data['source_file']}")
                return False
        
        return True

def main():
    """Test PDF extraction functionality"""
    import yaml
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize extractor
    extractor = PDFTextExtractor(config)
    
    # Run extraction
    success = extractor.extract_all_pdfs()
    
    if success:
        print("PDF extraction completed successfully")
    else:
        print("PDF extraction failed or no files found")

if __name__ == "__main__":
    main() 