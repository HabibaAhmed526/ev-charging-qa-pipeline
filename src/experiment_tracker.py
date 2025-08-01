import mlflow
import wandb
import yaml
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional
import torch
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """Experiment tracking with MLflow and Weights & Biases integration"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.mlflow_enabled = self.config["experiment_tracking"]["mlflow"]["enabled"]
        self.wandb_enabled = self.config["experiment_tracking"]["wandb"]["enabled"]
        
        # Initialize MLflow
        if self.mlflow_enabled:
            self._setup_mlflow()
        
        # Initialize Weights & Biases
        if self.wandb_enabled:
            self._setup_wandb()
    
    def _setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            tracking_uri = os.getenv("MLFLOW_TRACKING_URI", 
                                   self.config["experiment_tracking"]["mlflow"]["tracking_uri"])
            mlflow.set_tracking_uri(tracking_uri)
            mlflow.set_experiment(self.config["experiment_tracking"]["mlflow"]["experiment_name"])
            logger.info(f"MLflow initialized with tracking URI: {tracking_uri}")
        except Exception as e:
            logger.error(f"Failed to setup MLflow: {e}")
            self.mlflow_enabled = False
    
    def _setup_wandb(self):
        """Setup Weights & Biases tracking"""
        try:
            wandb.init(
                project=self.config["experiment_tracking"]["wandb"]["project"],
                entity=os.getenv("WANDB_ENTITY", self.config["experiment_tracking"]["wandb"]["entity"]),
                config=self.config
            )
            logger.info("Weights & Biases initialized successfully")
        except Exception as e:
            logger.error(f"Failed to setup Weights & Biases: {e}")
            self.wandb_enabled = False
    
    def start_run(self, run_name: str, tags: Optional[Dict[str, str]] = None):
        """Start a new experiment run"""
        self.run_name = run_name
        self.start_time = datetime.now()
        
        if self.mlflow_enabled:
            mlflow.start_run(run_name=run_name, tags=tags or {})
        
        if self.wandb_enabled:
            wandb.run.name = run_name
            if tags:
                wandb.run.tags = list(tags.values())
        
        logger.info(f"Started experiment run: {run_name}")
    
    def log_params(self, params: Dict[str, Any]):
        """Log hyperparameters"""
        if self.mlflow_enabled:
            mlflow.log_params(params)
        
        if self.wandb_enabled:
            wandb.config.update(params)
        
        logger.info(f"Logged parameters: {list(params.keys())}")
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics"""
        if self.mlflow_enabled:
            mlflow.log_metrics(metrics, step=step)
        
        if self.wandb_enabled:
            wandb.log(metrics, step=step)
        
        logger.info(f"Logged metrics: {list(metrics.keys())}")
    
    def log_model(self, model_path: str, model_name: str = "model"):
        """Log model artifacts"""
        if self.mlflow_enabled:
            mlflow.log_artifact(model_path, artifact_path=model_name)
        
        if self.wandb_enabled:
            wandb.save(model_path)
        
        logger.info(f"Logged model: {model_path}")
    
    def log_dataset(self, dataset_path: str, dataset_name: str = "dataset"):
        """Log dataset artifacts"""
        if self.mlflow_enabled:
            mlflow.log_artifact(dataset_path, artifact_path=dataset_name)
        
        if self.wandb_enabled:
            wandb.save(dataset_path)
        
        logger.info(f"Logged dataset: {dataset_path}")
    
    def log_training_metrics(self, epoch: int, train_loss: float, val_loss: Optional[float] = None):
        """Log training metrics for each epoch"""
        metrics = {"epoch": epoch, "train_loss": train_loss}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        
        self.log_metrics(metrics, step=epoch)
    
    def log_evaluation_metrics(self, metrics: Dict[str, float], model_type: str = "fine_tuned"):
        """Log evaluation metrics"""
        prefixed_metrics = {f"{model_type}_{k}": v for k, v in metrics.items()}
        self.log_metrics(prefixed_metrics)
    
    def log_performance_metrics(self, latency_ms: float, throughput: float, memory_usage_gb: float):
        """Log performance metrics"""
        metrics = {
            "inference_latency_ms": latency_ms,
            "throughput_requests_per_min": throughput,
            "memory_usage_gb": memory_usage_gb
        }
        self.log_metrics(metrics)
    
    def log_system_info(self):
        """Log system information"""
        import psutil
        import GPUtil
        
        system_info = {
            "cpu_count": psutil.cpu_count(),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0]
                system_info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_total_gb": gpu.memoryTotal / 1024
                })
            except:
                system_info["gpu_info"] = "unavailable"
        
        self.log_params(system_info)
    
    def end_run(self):
        """End the current experiment run"""
        duration = datetime.now() - self.start_time
        
        if self.mlflow_enabled:
            mlflow.end_run()
        
        if self.wandb_enabled:
            wandb.finish()
        
        logger.info(f"Ended experiment run: {self.run_name} (Duration: {duration})")
    
    def create_benchmark_dataset(self, output_path: str = "data/benchmark_dataset.jsonl"):
        """Create domain-specific benchmark dataset for EV charging"""
        benchmark_questions = [
            {
                "question": "What are the key safety requirements for EV charging station installation?",
                "context": "EV charging station safety requirements include proper grounding, circuit protection, weather resistance, and emergency shutdown capabilities.",
                "expected_answer": "Key safety requirements include proper grounding to prevent electrical hazards, circuit protection devices to prevent overloads, weather-resistant enclosures for outdoor installations, and emergency shutdown capabilities for safety emergencies."
            },
            {
                "question": "How do you calculate the electrical load for a commercial EV charging station?",
                "context": "Electrical load calculation involves determining the power requirements, considering charging speed, number of stations, and peak usage patterns.",
                "expected_answer": "Calculate electrical load by determining the power rating of each charging station (e.g., 50kW), multiplying by the number of stations, applying a diversity factor (typically 0.8-0.9), and adding 20% for future expansion."
            },
            {
                "question": "What are the environmental considerations for EV charging infrastructure?",
                "context": "Environmental considerations include reducing carbon footprint, managing electronic waste, and sustainable energy sources.",
                "expected_answer": "Environmental considerations include using renewable energy sources like solar panels, implementing energy storage systems, proper disposal of old equipment, minimizing construction impact, and ensuring the charging infrastructure supports grid stability."
            },
            {
                "question": "What permits are required for installing EV charging stations?",
                "context": "Permit requirements vary by jurisdiction but typically include electrical permits, building permits, and environmental assessments.",
                "expected_answer": "Required permits typically include electrical permits for wiring and equipment, building permits for structural modifications, environmental permits for large installations, and zoning permits for commercial locations."
            },
            {
                "question": "How do you optimize the cost of EV charging station deployment?",
                "context": "Cost optimization involves balancing upfront costs with operational efficiency and long-term benefits.",
                "expected_answer": "Cost optimization strategies include selecting appropriate charging speeds for usage patterns, leveraging government incentives and tax credits, implementing smart charging to reduce peak demand charges, and choosing scalable infrastructure for future growth."
            },
            {
                "question": "What maintenance procedures are required for EV charging stations?",
                "context": "Regular maintenance ensures reliable operation and extends equipment lifespan.",
                "expected_answer": "Maintenance procedures include monthly visual inspections, quarterly electrical testing, annual software updates, cleaning of connectors and displays, and preventive maintenance of cooling systems and power electronics."
            },
            {
                "question": "How do you ensure cybersecurity for EV charging networks?",
                "context": "Cybersecurity is critical for protecting user data and preventing unauthorized access to charging infrastructure.",
                "expected_answer": "Cybersecurity measures include implementing secure communication protocols, regular security updates, user authentication systems, encryption of payment data, network monitoring for suspicious activity, and compliance with industry security standards."
            },
            {
                "question": "What are the different types of EV charging connectors and their specifications?",
                "context": "Different connector types support various charging speeds and vehicle compatibility.",
                "expected_answer": "Common connector types include Type 1 (J1772) for Level 2 charging up to 19.2kW, Type 2 (Mennekes) for Level 2 and DC fast charging up to 350kW, CHAdeMO for DC fast charging up to 400kW, and CCS (Combined Charging System) for both AC and DC charging."
            },
            {
                "question": "How do you handle peak demand management for EV charging stations?",
                "context": "Peak demand management helps reduce costs and grid stress during high usage periods.",
                "expected_answer": "Peak demand management strategies include implementing smart charging algorithms that stagger charging times, offering time-of-use pricing incentives, integrating with grid demand response programs, and using energy storage systems to buffer peak loads."
            },
            {
                "question": "What are the accessibility requirements for EV charging stations?",
                "context": "Accessibility ensures that charging stations are usable by people with disabilities.",
                "expected_answer": "Accessibility requirements include providing accessible parking spaces with adequate space for wheelchair users, mounting charging equipment at accessible heights (15-48 inches), ensuring clear pathways to charging equipment, providing audio and visual feedback, and following ADA guidelines for signage and controls."
            }
        ]
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save benchmark dataset
        with open(output_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(benchmark_questions):
                benchmark_item = {
                    "id": f"benchmark_{i+1:03d}",
                    "question": item["question"],
                    "context": item["context"],
                    "expected_answer": item["expected_answer"],
                    "category": "ev_charging_infrastructure",
                    "difficulty": "medium"
                }
                f.write(json.dumps(benchmark_item, ensure_ascii=False) + "\n")
        
        logger.info(f"Created benchmark dataset with {len(benchmark_questions)} questions: {output_path}")
        return output_path
    
    def evaluate_benchmark(self, model, tokenizer, benchmark_path: str = "data/benchmark_dataset.jsonl"):
        """Evaluate model performance on benchmark dataset"""
        import json
        from transformers import pipeline
        
        # Load benchmark dataset
        with open(benchmark_path, "r", encoding="utf-8") as f:
            benchmark_data = [json.loads(line) for line in f]
        
        # Setup generation pipeline
        generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
        
        results = []
        total_latency = 0
        
        for item in benchmark_data:
            prompt = f"Question: {item['question']}\nContext: {item['context']}\nAnswer:"
            
            # Measure latency
            start_time = datetime.now()
            response = generator(prompt, max_new_tokens=150, do_sample=False)
            latency = (datetime.now() - start_time).total_seconds()
            total_latency += latency
            
            # Extract generated answer
            generated_text = response[0]["generated_text"]
            answer = generated_text[len(prompt):].strip()
            
            # Calculate simple similarity (can be enhanced with more sophisticated metrics)
            expected_words = set(item["expected_answer"].lower().split())
            generated_words = set(answer.lower().split())
            overlap = len(expected_words.intersection(generated_words))
            similarity = overlap / len(expected_words) if expected_words else 0
            
            results.append({
                "id": item["id"],
                "question": item["question"],
                "expected": item["expected_answer"],
                "generated": answer,
                "similarity": similarity,
                "latency": latency
            })
        
        # Calculate aggregate metrics
        avg_similarity = sum(r["similarity"] for r in results) / len(results)
        avg_latency = total_latency / len(results)
        
        metrics = {
            "benchmark_avg_similarity": avg_similarity,
            "benchmark_avg_latency_ms": avg_latency * 1000,
            "benchmark_total_questions": len(results)
        }
        
        self.log_metrics(metrics)
        
        # Save detailed results
        results_path = "data/benchmark_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump({
                "summary": metrics,
                "detailed_results": results
            }, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark evaluation completed. Avg similarity: {avg_similarity:.3f}, Avg latency: {avg_latency*1000:.1f}ms")
        return metrics, results

# Convenience function for quick setup
def get_tracker(config_path: str = "config.yaml") -> ExperimentTracker:
    """Get experiment tracker instance"""
    return ExperimentTracker(config_path)