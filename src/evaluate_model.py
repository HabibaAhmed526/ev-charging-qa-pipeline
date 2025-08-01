#!/usr/bin/env python3
"""
Model Evaluation Module

Evaluates the fine-tuned model performance using various metrics.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import numpy as np

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluates model performance using various metrics"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the model evaluator"""
        self.config = config
        self.evaluation_config = config.get("evaluation", {})
        self.environment_config = config.get("environment", {})
        
        self.base_model = self.environment_config.get("base_model", "mistralai/Mistral-7B-Instruct-v0.2")
        self.device = self.environment_config.get("device", "auto")
        
        # Setup device
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Initialize metrics
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        
        logger.info(f"Model evaluator initialized with device: {self.device}")
    
    def evaluate_model(self) -> bool:
        """Evaluate the fine-tuned model"""
        try:
            # Load benchmark dataset
            benchmark_data = self.load_benchmark_dataset()
            if not benchmark_data:
                logger.error("No benchmark dataset found")
                return False
            
            # Load models
            fine_tuned_model, fine_tuned_tokenizer = self.load_fine_tuned_model()
            baseline_model, baseline_tokenizer = self.load_baseline_model()
            
            if not fine_tuned_model or not baseline_model:
                logger.error("Failed to load models for evaluation")
                return False
            
            # Run evaluation
            fine_tuned_results = self.evaluate_model_on_dataset(fine_tuned_model, fine_tuned_tokenizer, benchmark_data)
            baseline_results = self.evaluate_model_on_dataset(baseline_model, baseline_tokenizer, benchmark_data)
            
            # Compare results
            comparison = self.compare_results(fine_tuned_results, baseline_results)
            
            # Save evaluation results
            self.save_evaluation_results(fine_tuned_results, baseline_results, comparison)
            
            logger.info("Model evaluation completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            return False
    
    def load_benchmark_dataset(self) -> List[Dict[str, Any]]:
        """Load benchmark dataset"""
        benchmark_file = Path(self.evaluation_config.get("benchmark_dataset", "data/benchmark_dataset.jsonl"))
        
        if not benchmark_file.exists():
            logger.warning("Benchmark dataset not found, creating sample dataset")
            return self.create_sample_dataset()
        
        benchmark_data = []
        try:
            with open(benchmark_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        benchmark_data.append(json.loads(line))
            
            logger.info(f"Loaded {len(benchmark_data)} benchmark samples")
            return benchmark_data
            
        except Exception as e:
            logger.error(f"Failed to load benchmark dataset: {e}")
            return []
    
    def create_sample_dataset(self) -> List[Dict[str, Any]]:
        """Create a sample benchmark dataset for EV charging"""
        sample_questions = [
            {
                "question": "What are the safety requirements for EV charging stations?",
                "reference_answer": "EV charging stations must meet several safety requirements including proper grounding, circuit protection, weather resistance, and compliance with electrical codes. They should have emergency stop buttons, clear signage, and regular safety inspections.",
                "category": "safety"
            },
            {
                "question": "How do you install an EV charging station?",
                "reference_answer": "EV charging station installation involves site assessment, electrical system evaluation, obtaining permits, installing the charging equipment, connecting to the electrical grid, and final inspection and testing.",
                "category": "installation"
            },
            {
                "question": "What are the different types of EV charging connectors?",
                "reference_answer": "Common EV charging connectors include Type 1 (J1772), Type 2 (Mennekes), CHAdeMO, CCS (Combined Charging System), and Tesla Supercharger. Each has different power levels and compatibility.",
                "category": "technical"
            },
            {
                "question": "What are the costs involved in EV charging infrastructure?",
                "reference_answer": "EV charging infrastructure costs include equipment purchase, installation, electrical upgrades, permitting, maintenance, and electricity costs. Total costs vary based on charging level and site requirements.",
                "category": "cost"
            },
            {
                "question": "What regulations govern EV charging station installation?",
                "reference_answer": "EV charging station installation is governed by local electrical codes, building codes, zoning regulations, and environmental requirements. Permits are typically required for electrical work and site modifications.",
                "category": "regulations"
            }
        ]
        
        # Save sample dataset
        benchmark_file = Path("data/benchmark_dataset.jsonl")
        benchmark_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(benchmark_file, 'w', encoding='utf-8') as f:
            for item in sample_questions:
                f.write(json.dumps(item) + '\n')
        
        logger.info(f"Created sample benchmark dataset with {len(sample_questions)} questions")
        return sample_questions
    
    def load_fine_tuned_model(self):
        """Load the fine-tuned model"""
        try:
            model_path = Path("models/finetuned")
            if not model_path.exists():
                logger.warning("No fine-tuned model found")
                return None, None
            
            logger.info("Loading fine-tuned model")
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load fine-tuned model: {e}")
            return None, None
    
    def load_baseline_model(self):
        """Load the baseline model"""
        try:
            logger.info("Loading baseline model")
            
            tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True
            )
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"Failed to load baseline model: {e}")
            return None, None
    
    def evaluate_model_on_dataset(self, model, tokenizer, dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a model on the benchmark dataset"""
        results = {
            "predictions": [],
            "metrics": {},
            "latency": []
        }
        
        for item in dataset:
            question = item["question"]
            reference_answer = item["reference_answer"]
            
            # Generate response
            start_time = time.time()
            prediction = self.generate_response(model, tokenizer, question)
            latency = time.time() - start_time
            
            results["predictions"].append({
                "question": question,
                "reference": reference_answer,
                "prediction": prediction,
                "latency": latency
            })
            results["latency"].append(latency)
        
        # Calculate metrics
        results["metrics"] = self.calculate_metrics(results["predictions"])
        
        return results
    
    def generate_response(self, model, tokenizer, question: str) -> str:
        """Generate response for a question"""
        try:
            # Format prompt
            formatted_prompt = f"<s>[INST] {question} [/INST]"
            
            # Tokenize input
            inputs = tokenizer(formatted_prompt, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=150,
                    temperature=0.7,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.replace(formatted_prompt, "").strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""
    
    def calculate_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate evaluation metrics"""
        metrics = {
            "bleu": 0.0,
            "rouge1": 0.0,
            "rouge2": 0.0,
            "rougeL": 0.0,
            "exact_match": 0.0,
            "avg_latency": 0.0
        }
        
        if not predictions:
            return metrics
        
        bleu_scores = []
        rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
        exact_matches = 0
        latencies = []
        
        for pred in predictions:
            reference = pred["reference"]
            prediction = pred["prediction"]
            latency = pred["latency"]
            
            # BLEU score
            try:
                bleu_score = sentence_bleu([reference.split()], prediction.split(), smoothing_function=self.smoothing)
                bleu_scores.append(bleu_score)
            except:
                bleu_scores.append(0.0)
            
            # ROUGE scores
            try:
                rouge_result = self.scorer.score(reference, prediction)
                for metric in ["rouge1", "rouge2", "rougeL"]:
                    rouge_scores[metric].append(rouge_result[metric].fmeasure)
            except:
                for metric in ["rouge1", "rouge2", "rougeL"]:
                    rouge_scores[metric].append(0.0)
            
            # Exact match
            if prediction.strip().lower() == reference.strip().lower():
                exact_matches += 1
            
            latencies.append(latency)
        
        # Calculate averages
        metrics["bleu"] = np.mean(bleu_scores)
        metrics["rouge1"] = np.mean(rouge_scores["rouge1"])
        metrics["rouge2"] = np.mean(rouge_scores["rouge2"])
        metrics["rougeL"] = np.mean(rouge_scores["rougeL"])
        metrics["exact_match"] = exact_matches / len(predictions)
        metrics["avg_latency"] = np.mean(latencies)
        
        return metrics
    
    def compare_results(self, fine_tuned_results: Dict[str, Any], baseline_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compare fine-tuned vs baseline results"""
        comparison = {
            "improvements": {},
            "fine_tuned_metrics": fine_tuned_results["metrics"],
            "baseline_metrics": baseline_results["metrics"]
        }
        
        fine_tuned_metrics = fine_tuned_results["metrics"]
        baseline_metrics = baseline_results["metrics"]
        
        for metric in ["bleu", "rouge1", "rouge2", "rougeL", "exact_match"]:
            if metric in fine_tuned_metrics and metric in baseline_metrics:
                improvement = fine_tuned_metrics[metric] - baseline_metrics[metric]
                improvement_pct = (improvement / baseline_metrics[metric]) * 100 if baseline_metrics[metric] > 0 else 0
                
                comparison["improvements"][metric] = {
                    "absolute": improvement,
                    "percentage": improvement_pct
                }
        
        return comparison
    
    def save_evaluation_results(self, fine_tuned_results: Dict[str, Any], baseline_results: Dict[str, Any], comparison: Dict[str, Any]):
        """Save evaluation results"""
        try:
            output_dir = Path("evaluation_results")
            output_dir.mkdir(exist_ok=True)
            
            # Save detailed results
            results = {
                "fine_tuned_results": fine_tuned_results,
                "baseline_results": baseline_results,
                "comparison": comparison,
                "evaluation_timestamp": time.time()
            }
            
            with open(output_dir / "evaluation_results.json", 'w') as f:
                json.dump(results, f, indent=2)
            
            # Save summary
            summary = {
                "fine_tuned_metrics": fine_tuned_results["metrics"],
                "baseline_metrics": baseline_results["metrics"],
                "improvements": comparison["improvements"],
                "avg_latency_fine_tuned": fine_tuned_results["metrics"]["avg_latency"],
                "avg_latency_baseline": baseline_results["metrics"]["avg_latency"]
            }
            
            with open(output_dir / "evaluation_summary.json", 'w') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(f"Evaluation results saved to {output_dir}")
            
        except Exception as e:
            logger.error(f"Failed to save evaluation results: {e}")

def main():
    """Test model evaluation functionality"""
    import yaml
    
    # Load config
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config)
    
    # Run evaluation
    success = evaluator.evaluate_model()
    
    if success:
        print("Model evaluation completed successfully")
    else:
        print("Model evaluation failed")

if __name__ == "__main__":
    main() 