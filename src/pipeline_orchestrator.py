import os
import sys
import logging
import yaml
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
import subprocess
import time

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from experiment_tracker import ExperimentTracker
from data_processor import EVChargingDataProcessor

logger = logging.getLogger(__name__)

class EVChargingPipelineOrchestrator:
    """Main pipeline orchestrator for EV charging QA system"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        # Setup logging
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config["logging"]["file"]),
                logging.StreamHandler()
            ]
        )
        
        self.experiment_tracker = ExperimentTracker()
        self.data_processor = EVChargingDataProcessor()
        
        # Pipeline stages
        self.stages = [
            "data_extraction",
            "text_chunking", 
            "qa_generation",
            "data_cleaning",
            "data_formatting",
            "data_processing",
            "fine_tuning",
            "model_merging",
            "evaluation",
            "benchmark_evaluation",
            "performance_testing"
        ]
        
        self.results = {}
    
    def run_stage(self, stage_name: str) -> bool:
        """Run a single pipeline stage"""
        logger.info(f"🔄 Running stage: {stage_name}")
        
        try:
            if stage_name == "data_extraction":
                return self._run_data_extraction()
            elif stage_name == "text_chunking":
                return self._run_text_chunking()
            elif stage_name == "qa_generation":
                return self._run_qa_generation()
            elif stage_name == "data_cleaning":
                return self._run_data_cleaning()
            elif stage_name == "data_formatting":
                return self._run_data_formatting()
            elif stage_name == "data_processing":
                return self._run_data_processing()
            elif stage_name == "fine_tuning":
                return self._run_fine_tuning()
            elif stage_name == "model_merging":
                return self._run_model_merging()
            elif stage_name == "evaluation":
                return self._run_evaluation()
            elif stage_name == "benchmark_evaluation":
                return self._run_benchmark_evaluation()
            elif stage_name == "performance_testing":
                return self._run_performance_testing()
            else:
                logger.error(f"Unknown stage: {stage_name}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Stage {stage_name} failed: {e}")
            return False
    
    def _run_data_extraction(self) -> bool:
        """Run PDF data extraction"""
        try:
            from extract_pdf_text import main as extract_main
            extract_main()
            self.results["data_extraction"] = "completed"
            return True
        except Exception as e:
            logger.error(f"Data extraction failed: {e}")
            return False
    
    def _run_text_chunking(self) -> bool:
        """Run text chunking"""
        try:
            result = subprocess.run([sys.executable, "src/chunk_text.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("Text chunking completed")
            self.results["text_chunking"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Text chunking failed: {e.stderr}")
            return False
    
    def _run_qa_generation(self) -> bool:
        """Run QA generation"""
        try:
            result = subprocess.run([sys.executable, "src/generate_qa_mistral.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("QA generation completed")
            self.results["qa_generation"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"QA generation failed: {e.stderr}")
            return False
    
    def _run_data_cleaning(self) -> bool:
        """Run data cleaning"""
        try:
            result = subprocess.run([sys.executable, "qa/clean_qa_dataset_mistral.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("Data cleaning completed")
            self.results["data_cleaning"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Data cleaning failed: {e.stderr}")
            return False
    
    def _run_data_formatting(self) -> bool:
        """Run data formatting"""
        try:
            result = subprocess.run([sys.executable, "qa/format_for_llama3.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("Data formatting completed")
            self.results["data_formatting"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Data formatting failed: {e.stderr}")
            return False
    
    def _run_data_processing(self) -> bool:
        """Run comprehensive data processing"""
        try:
            input_path = "qa/qa_dataset_mistral_cleaned.jsonl"
            if os.path.exists(input_path):
                results = self.data_processor.process_qa_dataset(input_path)
                self.results["data_processing"] = results
                logger.info("Data processing completed")
                return True
            else:
                logger.warning(f"Input file not found: {input_path}")
                return True  # Skip if file doesn't exist
        except Exception as e:
            logger.error(f"Data processing failed: {e}")
            return False
    
    def _run_fine_tuning(self) -> bool:
        """Run model fine-tuning"""
        try:
            # Start experiment tracking
            self.experiment_tracker.start_run("ev_charging_pipeline", 
                                           tags={"pipeline": "full", "domain": "ev_charging"})
            
            result = subprocess.run([sys.executable, "src/finetune_and_test_llm.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("Fine-tuning completed")
            self.results["fine_tuning"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Fine-tuning failed: {e.stderr}")
            return False
    
    def _run_model_merging(self) -> bool:
        """Run model merging"""
        try:
            result = subprocess.run([sys.executable, "src/merge_lora.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("Model merging completed")
            self.results["model_merging"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model merging failed: {e.stderr}")
            return False
    
    def _run_evaluation(self) -> bool:
        """Run model evaluation"""
        try:
            result = subprocess.run([sys.executable, "src/evaluate_model.py"], 
                                 capture_output=True, text=True, check=True)
            logger.info("Model evaluation completed")
            self.results["evaluation"] = "completed"
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Model evaluation failed: {e.stderr}")
            return False
    
    def _run_benchmark_evaluation(self) -> bool:
        """Run benchmark evaluation"""
        try:
            # Create benchmark dataset
            benchmark_path = self.experiment_tracker.create_benchmark_dataset()
            
            # Load model for evaluation
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            model_path = self.config["data"]["paths"]["model_output"]
            base_model = self.config["environment"]["base_model"]
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype="float16"
            )
            model = PeftModel.from_pretrained(model, model_path)
            
            # Run benchmark evaluation
            benchmark_metrics, benchmark_results = self.experiment_tracker.evaluate_benchmark(
                model, tokenizer, benchmark_path
            )
            
            self.results["benchmark_evaluation"] = {
                "metrics": benchmark_metrics,
                "results_path": "data/benchmark_results.json"
            }
            
            logger.info("Benchmark evaluation completed")
            return True
        except Exception as e:
            logger.error(f"Benchmark evaluation failed: {e}")
            return False
    
    def _run_performance_testing(self) -> bool:
        """Run performance testing"""
        try:
            from performance_test import PerformanceTester
            
            # Load model
            from transformers import AutoTokenizer, AutoModelForCausalLM
            from peft import PeftModel
            
            model_path = self.config["data"]["paths"]["model_output"]
            base_model = self.config["environment"]["base_model"]
            
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                load_in_4bit=True,
                device_map="auto",
                torch_dtype="float16"
            )
            model = PeftModel.from_pretrained(model, model_path)
            
            # Run performance tests
            tester = PerformanceTester()
            performance_results = tester.run_comprehensive_test(model, tokenizer)
            
            self.results["performance_testing"] = performance_results
            
            logger.info("Performance testing completed")
            return True
        except Exception as e:
            logger.error(f"Performance testing failed: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline end-to-end"""
        logger.info("🚀 Starting EV Charging QA Pipeline")
        start_time = time.time()
        
        # Check if required files exist
        required_files = [
            "data/raw/HandbookforEVChargingInfrastructureImplementation081221.pdf"
        ]
        
        for file_path in required_files:
            if not os.path.exists(file_path):
                logger.error(f"❌ Required file not found: {file_path}")
                return False
        
        # Run all stages
        successful_stages = []
        failed_stages = []
        
        for stage in self.stages:
            if self.run_stage(stage):
                successful_stages.append(stage)
                logger.info(f"✅ Stage {stage} completed successfully")
            else:
                failed_stages.append(stage)
                logger.error(f"❌ Stage {stage} failed")
        
        # End experiment tracking
        if "fine_tuning" in successful_stages:
            self.experiment_tracker.end_run()
        
        # Generate pipeline summary
        duration = time.time() - start_time
        summary = {
            "pipeline_name": "EV Charging QA Pipeline",
            "start_time": datetime.fromtimestamp(start_time).isoformat(),
            "duration_seconds": duration,
            "successful_stages": successful_stages,
            "failed_stages": failed_stages,
            "total_stages": len(self.stages),
            "success_rate": len(successful_stages) / len(self.stages),
            "results": self.results
        }
        
        # Save pipeline summary
        summary_path = "data/pipeline_summary.json"
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        # Print results
        print("\n" + "="*50)
        print("🎯 PIPELINE EXECUTION SUMMARY")
        print("="*50)
        print(f"✅ Successful stages: {len(successful_stages)}/{len(self.stages)}")
        print(f"⏱️  Total duration: {duration:.1f} seconds")
        print(f"📊 Success rate: {summary['success_rate']:.1%}")
        
        if successful_stages:
            print("\n✅ Successful stages:")
            for stage in successful_stages:
                print(f"  - {stage}")
        
        if failed_stages:
            print("\n❌ Failed stages:")
            for stage in failed_stages:
                print(f"  - {stage}")
        
        print(f"\n📁 Results saved to: {summary_path}")
        
        return len(failed_stages) == 0

def main():
    """Main function to run the pipeline"""
    orchestrator = EVChargingPipelineOrchestrator()
    success = orchestrator.run_full_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Pipeline completed with errors!")
        sys.exit(1)

if __name__ == "__main__":
    main()