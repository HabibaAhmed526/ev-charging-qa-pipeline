#!/usr/bin/env python3
"""
Test Suite for EV Charging QA Pipeline

Comprehensive tests for all pipeline components.
"""

import pytest
import os
import tempfile
import json
from unittest.mock import Mock, patch
import yaml

# Test data processing
def test_data_processor():
    """Test data processing functionality"""
    from src.data_processor import EVChargingDataProcessor
    
    # Mock config
    config = {
        "data_processing": {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "min_chunk_length": 100,
            "cleaning": {
                "remove_duplicates": True,
                "remove_empty_chunks": True,
                "normalize_whitespace": True
            }
        }
    }
    
    processor = EVChargingDataProcessor(config)
    
    # Test data cleaning
    test_data = [
        {"text": "EV charging station safety requirements.", "source": "test"},
        {"text": "EV charging station safety requirements.", "source": "test"},  # Duplicate
        {"text": "", "source": "test"},  # Empty
    ]
    
    cleaned_data = processor.clean_data(test_data)
    assert len(cleaned_data) == 1  # Should remove duplicates and empty entries
    assert cleaned_data[0]["text"] == "EV charging station safety requirements."

# Test PDF extraction
def test_pdf_extraction():
    """Test PDF text extraction"""
    from src.extract_pdf_text import PDFTextExtractor
    
    # Mock config
    config = {
        "pdf_extraction": {
            "input_dir": "data/raw/",
            "output_dir": "data/extracted/",
            "settings": {
                "extract_tables": True,
                "preserve_layout": True,
                "min_text_length": 50
            }
        }
    }
    
    extractor = PDFTextExtractor(config)
    
    # Test with mock PDF data
    mock_pdf_content = "EV charging infrastructure requirements"
    
    with patch('PyMuPDF.open') as mock_open:
        mock_doc = Mock()
        mock_page = Mock()
        mock_page.get_text.return_value = mock_pdf_content
        mock_doc.__iter__.return_value = [mock_page]
        mock_open.return_value = mock_doc
        
        result = extractor.extract_pdf_text("mock.pdf")
        assert "EV charging infrastructure" in result

# Test API endpoints
def test_api_endpoints():
    """Test API endpoint functionality"""
    from fastapi.testclient import TestClient
    from src.serve_model import app
    
    client = TestClient(app)
    
    # Test health endpoint
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data

# Test model evaluation
def test_model_evaluation():
    """Test model evaluation metrics"""
    from src.evaluate_model import ModelEvaluator
    
    # Mock config
    config = {
        "evaluation": {
            "benchmark_dataset": "data/benchmark_dataset.jsonl",
            "metrics": ["bleu", "rouge", "exact_match"]
        },
        "environment": {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "device": "cpu"
        }
    }
    
    evaluator = ModelEvaluator(config)
    
    # Test BLEU calculation
    predictions = ["EV charging stations require proper grounding."]
    references = [["EV charging stations require proper grounding."]]
    
    bleu_score = evaluator.calculate_bleu(predictions, references)
    assert 0 <= bleu_score <= 1

# Test configuration loading
def test_config_loading():
    """Test configuration file loading"""
    config_path = "config.yaml"
    assert os.path.exists(config_path)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    assert "domain" in config
    assert "environment" in config
    assert "api" in config

# Test pipeline orchestration
def test_pipeline_orchestrator():
    """Test pipeline orchestrator initialization"""
    from src.pipeline_orchestrator import EVChargingPipelineOrchestrator
    
    orchestrator = EVChargingPipelineOrchestrator()
    assert hasattr(orchestrator, 'stages')
    assert len(orchestrator.stages) > 0

# Test web scraper
def test_web_scraper():
    """Test web scraping functionality"""
    from src.web_scraper import EVChargingWebScraper
    
    # Mock config
    config = {
        "web_scraping": {
            "sources": [
                {
                    "name": "Test Source",
                    "url": "https://example.com",
                    "selectors": {
                        "title": "h1",
                        "content": "p"
                    }
                }
            ],
            "settings": {
                "timeout": 30,
                "max_retries": 3
            }
        }
    }
    
    scraper = EVChargingWebScraper(config)
    assert hasattr(scraper, 'session')
    assert scraper.output_dir == "data/scraped"

# Test model training
def test_model_training():
    """Test model training functionality"""
    from src.finetune_and_test_llm import ModelTrainer
    
    # Mock config
    config = {
        "training": {
            "lora_r": 16,
            "lora_alpha": 32,
            "learning_rate": 2e-4,
            "batch_size": 4
        },
        "environment": {
            "base_model": "mistralai/Mistral-7B-Instruct-v0.2",
            "device": "cpu"
        },
        "experiment_tracking": {
            "mlflow": {
                "tracking_uri": "http://localhost:5000",
                "experiment_name": "test"
            }
        }
    }
    
    trainer = ModelTrainer(config)
    assert hasattr(trainer, 'training_config')
    assert hasattr(trainer, 'device')

# Test data validation
def test_data_validation():
    """Test data validation functions"""
    from src.data_processor import EVChargingDataProcessor
    
    config = {"data_processing": {"min_chunk_length": 100}}
    processor = EVChargingDataProcessor(config)
    
    # Test valid data
    valid_data = [{"text": "Valid text with sufficient length", "source": "test"}]
    assert processor.validate_processed_data(valid_data) == True
    
    # Test invalid data
    invalid_data = [{"text": "Short", "source": "test"}]  # Too short
    assert processor.validate_processed_data(invalid_data) == False

# Test text chunking
def test_text_chunking():
    """Test text chunking functionality"""
    from src.data_processor import EVChargingDataProcessor
    
    config = {
        "data_processing": {
            "chunk_size": 100,
            "chunk_overlap": 20,
            "min_chunk_length": 50
        }
    }
    
    processor = EVChargingDataProcessor(config)
    
    # Test text chunking
    long_text = "This is a long text that should be chunked into smaller pieces. " * 10
    chunks = processor.chunk_text(long_text)
    
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100

# Test configuration validation
def test_config_validation():
    """Test configuration validation"""
    config_path = "config.yaml"
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Check required sections
    required_sections = ["domain", "environment", "api", "training"]
    for section in required_sections:
        assert section in config, f"Missing required config section: {section}"
    
    # Check domain configuration
    assert "topic" in config["domain"]
    assert "sources" in config["domain"]
    
    # Check API configuration
    assert "host" in config["api"]
    assert "port" in config["api"]
    assert "auth" in config["api"]

# Test file structure
def test_file_structure():
    """Test that required files and directories exist"""
    required_files = [
        "config.yaml",
        "requirements.txt",
        "README.md"
    ]
    
    required_dirs = [
        "src",
        "tests",
        "data"
    ]
    
    for file_path in required_files:
        assert os.path.exists(file_path), f"Missing required file: {file_path}"
    
    for dir_path in required_dirs:
        assert os.path.isdir(dir_path), f"Missing required directory: {dir_path}"

# Test imports
def test_imports():
    """Test that all modules can be imported"""
    try:
        from src.pipeline_orchestrator import EVChargingPipelineOrchestrator
        from src.extract_pdf_text import PDFTextExtractor
        from src.web_scraper import EVChargingWebScraper
        from src.data_processor import EVChargingDataProcessor
        from src.finetune_and_test_llm import ModelTrainer
        from src.evaluate_model import ModelEvaluator
        from src.serve_model import app
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__]) 