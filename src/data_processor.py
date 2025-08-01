#!/usr/bin/env python3
"""
Data Processor Module

Cleans, processes, and prepares EV charging infrastructure data for training.
"""

import logging
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class EVChargingDataProcessor:
    """Processes and cleans EV charging infrastructure data"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the data processor"""
        self.config = config
        self.processing_config = config.get("data_processing", {})
        self.cleaning_config = self.processing_config.get("cleaning", {})
        self.input_dir = Path("data/extracted/")
        self.output_dir = Path("data/processed/")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("Data processor initialized")
    
    def process_all_data(self) -> bool:
        """Process all available data sources"""
        try:
            # Load extracted data
            extracted_data = self.load_extracted_data()
            if not extracted_data:
                logger.warning("No extracted data found")
                return False
            
            # Process each data source
            processed_data = []
            for data in extracted_data:
                processed = self.process_data_source(data)
                if processed:
                    processed_data.extend(processed)
            
            if not processed_data:
                logger.warning("No data was processed")
                return False
            
            # Clean and filter data
            cleaned_data = self.clean_data(processed_data)
            
            # Save processed data
            self.save_processed_data(cleaned_data)
            
            logger.info(f"Processed {len(cleaned_data)} data items")
            return True
            
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return False
    
    def load_extracted_data(self) -> List[Dict[str, Any]]:
        """Load extracted data from various sources"""
        data_sources = []
        
        # Load PDF extracted data
        pdf_data_file = self.input_dir / "extracted_data.json"
        if pdf_data_file.exists():
            try:
                with open(pdf_data_file, 'r', encoding='utf-8') as f:
                    pdf_data = json.load(f)
                data_sources.extend(pdf_data)
                logger.info(f"Loaded {len(pdf_data)} PDF data sources")
            except Exception as e:
                logger.error(f"Failed to load PDF data: {e}")
        
        # Load scraped data
        scraped_data_file = Path("data/scraped/scraped_data.json")
        if scraped_data_file.exists():
            try:
                with open(scraped_data_file, 'r', encoding='utf-8') as f:
                    scraped_data = json.load(f)
                data_sources.extend(scraped_data)
                logger.info(f"Loaded {len(scraped_data)} scraped data sources")
            except Exception as e:
                logger.error(f"Failed to load scraped data: {e}")
        
        return data_sources
    
    def process_data_source(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process a single data source"""
        processed_items = []
        
        try:
            source_type = data.get("source_type", "unknown")
            
            if "text_content" in data:  # PDF data
                processed_items.extend(self.process_pdf_data(data))
            elif "content" in data:  # Scraped data
                processed_items.extend(self.process_scraped_data(data))
            else:
                logger.warning(f"Unknown data format for source: {data.get('source_name', 'unknown')}")
        
        except Exception as e:
            logger.error(f"Error processing data source: {e}")
        
        return processed_items
    
    def process_pdf_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process PDF extracted data"""
        processed_items = []
        
        try:
            text_content = data.get("text_content", {})
            full_text = text_content.get("full_text", "")
            
            if not full_text.strip():
                return processed_items
            
            # Split text into chunks
            chunks = self.chunk_text(full_text)
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) >= self.processing_config.get("min_chunk_length", 100):
                    processed_items.append({
                        "text": chunk.strip(),
                        "source": data.get("source_file", "unknown"),
                        "source_type": "pdf",
                        "chunk_id": i,
                        "metadata": data.get("metadata", {})
                    })
        
        except Exception as e:
            logger.error(f"Error processing PDF data: {e}")
        
        return processed_items
    
    def process_scraped_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process scraped web data"""
        processed_items = []
        
        try:
            content = data.get("content", {})
            main_content = content.get("content", "")
            
            if not main_content.strip():
                return processed_items
            
            # Clean and process content
            cleaned_content = self.clean_text(main_content)
            
            if len(cleaned_content.strip()) >= self.processing_config.get("min_chunk_length", 100):
                processed_items.append({
                    "text": cleaned_content.strip(),
                    "source": data.get("source_name", "unknown"),
                    "source_type": "web",
                    "url": data.get("source_url", ""),
                    "metadata": data.get("metadata", {})
                })
        
        except Exception as e:
            logger.error(f"Error processing scraped data: {e}")
        
        return processed_items
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunk_size = self.processing_config.get("chunk_size", 512)
        chunk_overlap = self.processing_config.get("chunk_overlap", 50)
        
        # Split by sentences first
        sentences = re.split(r'[.!?]+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed chunk size
            if len(current_chunk + sentence) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                
                # Start new chunk with overlap
                if chunk_overlap > 0:
                    words = current_chunk.split()
                    overlap_words = words[-chunk_overlap:] if len(words) > chunk_overlap else words
                    current_chunk = " ".join(overlap_words) + " " + sentence
                else:
                    current_chunk = sentence
            else:
                current_chunk += " " + sentence
        
        # Add the last chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def clean_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and filter processed data"""
        cleaned_data = []
        
        # Remove duplicates
        if self.cleaning_config.get("remove_duplicates", True):
            seen_texts = set()
            for item in data:
                text = item.get("text", "").strip()
                if text and text not in seen_texts:
                    seen_texts.add(text)
                    cleaned_data.append(item)
        else:
            cleaned_data = data
        
        # Remove empty chunks
        if self.cleaning_config.get("remove_empty_chunks", True):
            cleaned_data = [item for item in cleaned_data if item.get("text", "").strip()]
        
        # Normalize whitespace
        if self.cleaning_config.get("normalize_whitespace", True):
            for item in cleaned_data:
                item["text"] = re.sub(r'\s+', ' ', item["text"]).strip()
        
        # Remove special characters
        if self.cleaning_config.get("remove_special_chars", False):
            for item in cleaned_data:
                item["text"] = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', item["text"])
        
        # Filter by minimum length
        min_length = self.processing_config.get("min_chunk_length", 100)
        cleaned_data = [item for item in cleaned_data if len(item.get("text", "")) >= min_length]
        
        logger.info(f"Cleaned data: {len(data)} -> {len(cleaned_data)} items")
        return cleaned_data
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove HTML tags if any
        text = re.sub(r'<[^>]+>', '', text)
        
        # Normalize quotes and dashes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        text = text.replace('–', '-').replace('—', '-')
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def save_processed_data(self, data: List[Dict[str, Any]]):
        """Save processed data to output directory"""
        try:
            # Save as JSON
            json_file = self.output_dir / "processed_data.json"
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Save as text file
            text_file = self.output_dir / "processed_text.txt"
            with open(text_file, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(f"=== {item.get('source', 'unknown')} ===\n")
                    f.write(item.get("text", ""))
                    f.write("\n\n")
            
            # Create statistics
            stats = self.create_data_statistics(data)
            stats_file = self.output_dir / "data_statistics.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Saved processed data to {self.output_dir}")
            
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
    
    def create_data_statistics(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create statistics about the processed data"""
        stats = {
            "total_items": len(data),
            "total_characters": sum(len(item.get("text", "")) for item in data),
            "total_words": sum(len(item.get("text", "").split()) for item in data),
            "source_distribution": {},
            "text_length_distribution": {
                "short": 0,  # < 200 chars
                "medium": 0,  # 200-500 chars
                "long": 0     # > 500 chars
            }
        }
        
        # Source distribution
        for item in data:
            source = item.get("source", "unknown")
            stats["source_distribution"][source] = stats["source_distribution"].get(source, 0) + 1
        
        # Length distribution
        for item in data:
            text_length = len(item.get("text", ""))
            if text_length < 200:
                stats["text_length_distribution"]["short"] += 1
            elif text_length < 500:
                stats["text_length_distribution"]["medium"] += 1
            else:
                stats["text_length_distribution"]["long"] += 1
        
        # Average lengths
        if data:
            stats["avg_text_length"] = stats["total_characters"] / len(data)
            stats["avg_word_count"] = stats["total_words"] / len(data)
        
        return stats
    
    def validate_processed_data(self, data: List[Dict[str, Any]]) -> bool:
        """Validate processed data quality"""
        if not data:
            return False
        
        for item in data:
            text = item.get("text", "")
            
            # Check minimum length
            if len(text.strip()) < self.processing_config.get("min_chunk_length", 100):
                logger.warning(f"Processed text too short: {len(text)} characters")
                return False
            
            # Check for required fields
            if not item.get("source"):
                logger.warning("Processed item missing source")
                return False
        
        return True

def main():
    """Test data processing functionality"""
    import yaml
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize processor
    processor = EVChargingDataProcessor(config)
    
    # Run processing
    success = processor.process_all_data()
    
    if success:
        print("Data processing completed successfully")
    else:
        print("Data processing failed or no data found")

if __name__ == "__main__":
    main() 