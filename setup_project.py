#!/usr/bin/env python3
"""
Setup script for EV Charging QA Pipeline
Creates necessary directories and validates project structure
"""

import os
import sys
from pathlib import Path

def create_directories():
    """Create all necessary directories for the project"""
    
    # Define required directories
    directories = [
        "logs",
        "data/raw",
        "data/extracted", 
        "data/processed",
        "data/qa",
        "data/mistral-lora-finetuned",
        "tests"
    ]
    
    print("📁 Creating project directories...")
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"  ✅ Created: {directory}")
    
    print("✅ All directories created successfully!")

def check_requirements():
    """Check if requirements.txt exists"""
    if not os.path.exists("requirements.txt"):
        print("❌ requirements.txt not found!")
        return False
    
    print("✅ requirements.txt found")
    return True

def check_config():
    """Check if config.yaml exists"""
    if not os.path.exists("config.yaml"):
        print("❌ config.yaml not found!")
        return False
    
    print("✅ config.yaml found")
    return True

def check_source_files():
    """Check if essential source files exist"""
    essential_files = [
        "src/pipeline_orchestrator.py",
        "src/data_processor.py", 
        "src/experiment_tracker.py",
        "src/performance_test.py",
        "src/extract_pdf_text.py",
        "src/chunk_text.py",
        "src/generate_qa_mistral.py",
        "src/finetune_and_test_llm.py",
        "src/evaluate_model.py",
        "src/serve_model.py"
    ]
    
    missing_files = []
    for file in essential_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing essential files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    
    print("✅ All essential source files found")
    return True

def create_sample_files():
    """Create sample files if they don't exist"""
    
    # Create .gitignore if it doesn't exist
    if not os.path.exists(".gitignore"):
        gitignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.venv/
venv/
ENV/

# IDE
.vscode/
.idea/
*.swp
*.swo

# Logs
logs/*.log

# Data
data/mistral-lora-finetuned/
data/processed/
*.jsonl

# Environment variables
.env

# Model files
*.safetensors
*.bin
*.pt

# OS
.DS_Store
Thumbs.db
"""
        with open(".gitignore", "w") as f:
            f.write(gitignore_content)
        print("✅ Created .gitignore")

def main():
    """Main setup function"""
    print("🚀 Setting up EV Charging QA Pipeline...")
    print("=" * 50)
    
    # Create directories
    create_directories()
    print()
    
    # Check requirements
    if not check_requirements():
        print("❌ Please ensure requirements.txt exists")
        sys.exit(1)
    
    # Check config
    if not check_config():
        print("❌ Please ensure config.yaml exists")
        sys.exit(1)
    
    # Check source files
    if not check_source_files():
        print("❌ Please ensure all source files exist")
        sys.exit(1)
    
    # Create sample files
    create_sample_files()
    
    print()
    print("🎉 Project setup completed successfully!")
    print()
    print("📋 Next steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Set up environment: cp .env.example .env")
    print("3. Run the pipeline: python src/pipeline_orchestrator.py")
    print()
    print("📁 Project structure:")
    print("├── src/                    # Source code")
    print("├── data/                   # Data directory")
    print("│   ├── raw/               # Raw PDF files")
    print("│   ├── extracted/         # Extracted text")
    print("│   ├── processed/         # Processed datasets")
    print("│   └── qa/               # QA datasets")
    print("├── logs/                  # Log files")
    print("├── tests/                 # Test files")
    print("├── config.yaml           # Configuration")
    print("└── requirements.txt      # Dependencies")

if __name__ == "__main__":
    main()