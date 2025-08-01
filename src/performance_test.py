import time
import torch
import psutil
import GPUtil
import json
import logging
import yaml
import asyncio
import aiohttp
from typing import Dict, List, Any, Optional
from datetime import datetime
import statistics
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

class PerformanceTester:
    """Comprehensive performance testing for the EV charging QA model"""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.performance_config = self.config["evaluation"]["performance"]
        self.results = {}
    
    def measure_single_inference(self, model, tokenizer, prompt: str, max_tokens: int = 150) -> Dict[str, float]:
        """Measure single inference performance"""
        # Warm up
        _ = model.generate(
            tokenizer(prompt[:100], return_tensors="pt").to(model.device),
            max_new_tokens=10,
            do_sample=False
        )
        
        # Measure inference
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        # Get initial memory usage
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated() / (1024**3)  # GB
        
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        
        end_time = time.time()
        
        # Calculate metrics
        latency = end_time - start_time
        tokens_generated = len(outputs[0]) - len(inputs["input_ids"][0])
        tokens_per_second = tokens_generated / latency if latency > 0 else 0
        
        # Memory usage
        if torch.cuda.is_available():
            final_memory = torch.cuda.memory_allocated() / (1024**3)
            memory_used = final_memory - initial_memory
        else:
            memory_used = 0
        
        return {
            "latency_seconds": latency,
            "latency_ms": latency * 1000,
            "tokens_generated": tokens_generated,
            "tokens_per_second": tokens_per_second,
            "memory_used_gb": memory_used,
            "input_tokens": len(inputs["input_ids"][0])
        }
    
    def run_latency_test(self, model, tokenizer, num_requests: int = 100) -> Dict[str, Any]:
        """Run comprehensive latency testing"""
        logger.info(f"Running latency test with {num_requests} requests")
        
        test_prompts = [
            "What are the safety requirements for EV charging stations?",
            "How do you calculate electrical load for commercial charging stations?",
            "What permits are needed for EV charging installation?",
            "How do you optimize costs for EV charging deployment?",
            "What maintenance procedures are required for charging stations?",
            "How do you ensure cybersecurity for EV charging networks?",
            "What are the environmental considerations for charging infrastructure?",
            "How do you handle peak demand management for charging stations?",
            "What are the accessibility requirements for EV charging?",
            "How do you select appropriate charging connectors?"
        ]
        
        results = []
        
        for i in range(num_requests):
            prompt = test_prompts[i % len(test_prompts)]
            result = self.measure_single_inference(model, tokenizer, prompt)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Completed {i + 1}/{num_requests} requests")
        
        # Calculate statistics
        latencies = [r["latency_ms"] for r in results]
        tokens_per_sec = [r["tokens_per_second"] for r in results]
        memory_usage = [r["memory_used_gb"] for r in results]
        
        stats = {
            "total_requests": len(results),
            "avg_latency_ms": statistics.mean(latencies),
            "median_latency_ms": statistics.median(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "std_latency_ms": statistics.stdev(latencies),
            "avg_tokens_per_second": statistics.mean(tokens_per_sec),
            "avg_memory_usage_gb": statistics.mean(memory_usage),
            "throughput_requests_per_minute": 60 / statistics.mean(latencies) * 1000 if latencies else 0
        }
        
        self.results["latency_test"] = {
            "statistics": stats,
            "raw_results": results
        }
        
        logger.info(f"Latency test completed. Avg latency: {stats['avg_latency_ms']:.1f}ms")
        return stats
    
    def run_throughput_test(self, model, tokenizer, duration_seconds: int = 60) -> Dict[str, Any]:
        """Run throughput testing for specified duration"""
        logger.info(f"Running throughput test for {duration_seconds} seconds")
        
        test_prompts = [
            "What are the key safety requirements?",
            "How do you calculate electrical load?",
            "What permits are required?",
            "How do you optimize costs?",
            "What maintenance is needed?"
        ]
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        requests_completed = 0
        total_tokens = 0
        latencies = []
        
        while time.time() < end_time:
            prompt = test_prompts[requests_completed % len(test_prompts)]
            
            request_start = time.time()
            result = self.measure_single_inference(model, tokenizer, prompt, max_tokens=100)
            request_end = time.time()
            
            requests_completed += 1
            total_tokens += result["tokens_generated"]
            latencies.append((request_end - request_start) * 1000)
        
        actual_duration = time.time() - start_time
        
        stats = {
            "duration_seconds": actual_duration,
            "requests_completed": requests_completed,
            "total_tokens_generated": total_tokens,
            "requests_per_second": requests_completed / actual_duration,
            "requests_per_minute": (requests_completed / actual_duration) * 60,
            "tokens_per_second": total_tokens / actual_duration,
            "avg_latency_ms": statistics.mean(latencies),
            "p95_latency_ms": np.percentile(latencies, 95),
            "p99_latency_ms": np.percentile(latencies, 99)
        }
        
        self.results["throughput_test"] = {
            "statistics": stats,
            "raw_results": latencies
        }
        
        logger.info(f"Throughput test completed. {stats['requests_per_minute']:.1f} requests/minute")
        return stats
    
    def run_concurrent_test(self, api_url: str, num_concurrent: int = 10, num_requests: int = 100) -> Dict[str, Any]:
        """Run concurrent API testing"""
        logger.info(f"Running concurrent test with {num_concurrent} concurrent clients, {num_requests} total requests")
        
        async def make_request(session, request_id):
            payload = {
                "prompt": f"What are the safety requirements for EV charging station {request_id}?",
                "max_tokens": 100
            }
            
            start_time = time.time()
            try:
                async with session.post(f"{api_url}/generate", json=payload) as response:
                    end_time = time.time()
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "request_id": request_id,
                            "latency_ms": (end_time - start_time) * 1000,
                            "success": True,
                            "tokens_generated": result.get("tokens_generated", 0)
                        }
                    else:
                        return {
                            "request_id": request_id,
                            "latency_ms": (end_time - start_time) * 1000,
                            "success": False,
                            "error": f"HTTP {response.status}"
                        }
            except Exception as e:
                return {
                    "request_id": request_id,
                    "latency_ms": (time.time() - start_time) * 1000,
                    "success": False,
                    "error": str(e)
                }
        
        async def run_concurrent_test():
            async with aiohttp.ClientSession() as session:
                tasks = []
                for i in range(num_requests):
                    task = make_request(session, i)
                    tasks.append(task)
                    
                    # Limit concurrent requests
                    if len(tasks) >= num_concurrent:
                        results = await asyncio.gather(*tasks)
                        tasks = []
                
                # Handle remaining tasks
                if tasks:
                    results = await asyncio.gather(*tasks)
                else:
                    results = []
                
                return results
        
        # Run the concurrent test
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            results = loop.run_until_complete(run_concurrent_test())
        finally:
            loop.close()
        
        # Calculate statistics
        successful_requests = [r for r in results if r["success"]]
        failed_requests = [r for r in results if not r["success"]]
        
        if successful_requests:
            latencies = [r["latency_ms"] for r in successful_requests]
            tokens_generated = [r["tokens_generated"] for r in successful_requests]
            
            stats = {
                "total_requests": len(results),
                "successful_requests": len(successful_requests),
                "failed_requests": len(failed_requests),
                "success_rate": len(successful_requests) / len(results),
                "avg_latency_ms": statistics.mean(latencies),
                "median_latency_ms": statistics.median(latencies),
                "p95_latency_ms": np.percentile(latencies, 95),
                "p99_latency_ms": np.percentile(latencies, 99),
                "min_latency_ms": min(latencies),
                "max_latency_ms": max(latencies),
                "avg_tokens_generated": statistics.mean(tokens_generated),
                "throughput_requests_per_minute": (len(successful_requests) / (max(latencies) / 1000)) * 60
            }
        else:
            stats = {"error": "No successful requests"}
        
        self.results["concurrent_test"] = {
            "statistics": stats,
            "raw_results": results
        }
        
        logger.info(f"Concurrent test completed. Success rate: {stats.get('success_rate', 0):.2%}")
        return stats
    
    def measure_system_resources(self) -> Dict[str, Any]:
        """Measure current system resource usage"""
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        system_info = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "memory_total_gb": memory.total / (1024**3),
            "memory_used_gb": memory.used / (1024**3)
        }
        
        # GPU information
        if torch.cuda.is_available():
            try:
                gpu = GPUtil.getGPUs()[0]
                gpu_memory = torch.cuda.memory_stats()
                
                system_info.update({
                    "gpu_name": gpu.name,
                    "gpu_memory_used_gb": gpu.memoryUsed / 1024,
                    "gpu_memory_total_gb": gpu.memoryTotal / 1024,
                    "gpu_utilization_percent": gpu.load * 100,
                    "torch_gpu_memory_allocated_gb": gpu_memory.get("allocated_bytes.all.current", 0) / (1024**3),
                    "torch_gpu_memory_reserved_gb": gpu_memory.get("reserved_bytes.all.current", 0) / (1024**3)
                })
            except Exception as e:
                system_info["gpu_error"] = str(e)
        
        self.results["system_resources"] = system_info
        return system_info
    
    def run_comprehensive_test(self, model, tokenizer, api_url: Optional[str] = None) -> Dict[str, Any]:
        """Run comprehensive performance testing"""
        logger.info("Starting comprehensive performance testing")
        
        # System resources before testing
        initial_resources = self.measure_system_resources()
        
        # Latency test
        latency_stats = self.run_latency_test(model, tokenizer, self.performance_config["num_requests"])
        
        # Throughput test
        throughput_stats = self.run_throughput_test(model, tokenizer, duration_seconds=30)
        
        # Concurrent test (if API URL provided)
        concurrent_stats = None
        if api_url:
            concurrent_stats = self.run_concurrent_test(api_url, self.performance_config["concurrent_requests"])
        
        # System resources after testing
        final_resources = self.measure_system_resources()
        
        # Compile comprehensive results
        comprehensive_results = {
            "timestamp": datetime.now().isoformat(),
            "initial_resources": initial_resources,
            "final_resources": final_resources,
            "latency_test": latency_stats,
            "throughput_test": throughput_stats,
            "concurrent_test": concurrent_stats,
            "summary": {
                "avg_latency_ms": latency_stats["avg_latency_ms"],
                "throughput_requests_per_minute": throughput_stats["requests_per_minute"],
                "memory_usage_gb": final_resources.get("torch_gpu_memory_allocated_gb", 0),
                "gpu_utilization_percent": final_resources.get("gpu_utilization_percent", 0)
            }
        }
        
        self.results["comprehensive"] = comprehensive_results
        
        # Save results
        output_path = "data/performance_results.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(comprehensive_results, f, indent=2)
        
        logger.info(f"Comprehensive testing completed. Results saved to {output_path}")
        return comprehensive_results
    
    def generate_performance_report(self, output_path: str = "data/performance_report.md") -> str:
        """Generate a markdown performance report"""
        if not self.results:
            return "No performance test results available"
        
        report = f"""# EV Charging QA Model Performance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary

"""
        
        if "comprehensive" in self.results:
            summary = self.results["comprehensive"]["summary"]
            report += f"""
- **Average Latency**: {summary['avg_latency_ms']:.1f}ms
- **Throughput**: {summary['throughput_requests_per_minute']:.1f} requests/minute
- **Memory Usage**: {summary['memory_usage_gb']:.2f}GB
- **GPU Utilization**: {summary['gpu_utilization_percent']:.1f}%

"""
        
        # Detailed results
        for test_name, test_results in self.results.items():
            if test_name == "comprehensive":
                continue
            
            report += f"## {test_name.replace('_', ' ').title()}\n\n"
            
            if "statistics" in test_results:
                stats = test_results["statistics"]
                for key, value in stats.items():
                    if isinstance(value, float):
                        report += f"- **{key.replace('_', ' ').title()}**: {value:.2f}\n"
                    else:
                        report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
            
            report += "\n"
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
        
        logger.info(f"Performance report generated: {output_path}")
        return report

# Convenience function
def run_performance_test(model, tokenizer, api_url: Optional[str] = None, config_path: str = "config.yaml"):
    """Run comprehensive performance testing"""
    tester = PerformanceTester(config_path)
    return tester.run_comprehensive_test(model, tokenizer, api_url)