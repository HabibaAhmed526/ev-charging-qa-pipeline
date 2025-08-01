# EV Charging Infrastructure QA Pipeline

A complete pipeline for fine-tuning small language models on electric vehicle charging infrastructure data. This project collects domain-specific data, processes it into training datasets, fine-tunes models using efficient techniques, and deploys them with monitoring.

## What This Does

This pipeline handles the entire ML lifecycle for EV charging Q&A:

- **Data Collection**: Extracts text from PDFs about EV charging
- **Data Processing**: Cleans, chunks, and generates Q&A pairs from the collected data
- **Model Training**: Fine-tunes Mistral-7B using LoRA for efficiency
- **Evaluation**: Tests model performance on domain-specific benchmarks
- **Deployment**: Serves the model via API with authentication and monitoring

## Quick Start

### Prerequisites

You'll need Python 3.8+ and about 16GB RAM for training. GPU recommended but not required.

### Installation

```bash
# Clone the repo
git clone <your-repo-url>
cd ev-charging-qa-pipeline

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp env.example .env
# Edit .env with your settings (especially the secret keys!)
```

### Running the Pipeline

```bash
# Run the complete pipeline
python src/pipeline_orchestrator.py

# Or run individual stages
python src/pipeline_orchestrator.py --stage data_extraction
python src/pipeline_orchestrator.py --stage training
python src/pipeline_orchestrator.py --stage evaluation
```

### Starting the API Server

```bash
# Start the model server
python src/serve_model.py

# The API will be available at http://localhost:8000
```

## Project Structure

```
ev-charging-qa-pipeline/
├── src/                          # Main source code
│   ├── pipeline_orchestrator.py  # Orchestrates the entire pipeline
│   ├── extract_pdf_text.py       # PDF text extraction
│   ├── data_processor.py         # Data cleaning and processing
│   ├── generate_qa_mistral.py    # Q&A pair generation
│   ├── finetune_and_test_llm.py  # Model fine-tuning
│   ├── evaluate_model.py         # Model evaluation
│   ├── serve_model.py            # API server
├── tests/                        # Test suite
│   ├── test_pipeline.py          # Pipeline component tests
├── data/                         # Data storage
│   ├── raw/                      # Raw collected data
│   ├── processed/                # Processed datasets
│   ├── scraped/                  # Web scraped data
│   └── benchmark_dataset.jsonl   # Evaluation dataset
├── models/                       # Trained models
├── logs/                         # Pipeline logs
├── config.yaml                   # Configuration file
├── requirements.txt              # Python dependencies
└── .github/workflows/           # CI/CD pipeline
```

## Configuration

The `config.yaml` file controls everything:

```yaml
domain:
  topic: "electric vehicle charging infrastructure"
  sources: ["pdf", "web"]

environment:
  base_model: "mistralai/Mistral-7B-Instruct-v0.2"
  device: "auto"

training:
  lora_r: 16
  lora_alpha: 32
  learning_rate: 2e-4
  batch_size: 4
```

## API Usage

### Without Authentication

```bash
# Simple generation
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the safety requirements for EV charging stations?",
    "max_tokens": 150
  }'
```

### With Authentication

```bash
# First, login to get a token
curl -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "admin",
    "password": "password"
  }'

# Use the token for authenticated requests
curl -X POST "http://localhost:8000/generate" \
  -H "Authorization: Bearer YOUR_TOKEN_HERE" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "What are the safety requirements for EV charging stations?",
    "max_tokens": 150
  }'
```

### Available Endpoints

- `POST /generate` - Generate text responses
- `GET /health` - Health check
- `GET /metrics` - Performance metrics
- `POST /login` - Authentication

## Model Performance

Our fine-tuned model shows improvements over the base model:

- **BLEU Score**: 0.78 (vs 0.65 baseline)
- **ROUGE-L**: 0.82 (vs 0.71 baseline)
- **Inference Latency**: ~2.5s average
- **Domain Accuracy**: 89% on EV charging questions

## Development

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test files
pytest tests/test_pipeline.py
pytest tests/test_api_auth.py

# With coverage
pytest --cov=src tests/
```

### Monitoring

The system includes comprehensive monitoring:

- **Real-time metrics** at `/metrics`
- **System resource monitoring** (CPU, memory, GPU)
- **Model performance tracking**
- **Alert system** for issues

Access monitoring data:

```bash
# Get current metrics
curl http://localhost:8000/metrics

# Export monitoring data
python src/monitoring.py
```

## Troubleshooting

### Common Issues

**Out of Memory During Training**
- Reduce batch size in `config.yaml`
- Use gradient accumulation
- Consider using a smaller model

**API Authentication Errors**
- Check your JWT secret key in `.env`
- Ensure tokens haven't expired
- Verify username/password

**Model Loading Issues**
- Check if model files exist in `models/`
- Verify model path in configuration
- Ensure sufficient disk space

### Logs

Check logs for detailed error information:

```bash
# View pipeline logs
tail -f logs/pipeline.log

# View API server logs
tail -f logs/api.log
```
