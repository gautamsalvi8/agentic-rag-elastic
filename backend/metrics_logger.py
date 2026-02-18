"""
metrics_logger.py
---------------------------------
Production-grade metrics logging for RAG system
- Logs every query with full performance breakdown
- Aggregate statistics (avg latency, cache hit rate, etc.)
- Persistent storage to JSON
- Dashboard-ready metrics
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import statistics

class MetricsLogger:
    def __init__(self, log_file="metrics_log.json"):
        """
        Initialize metrics logger
        
        Args:
            log_file: Path to JSON file for persistent storage
        """
        self.log_file = log_file
        self.metrics = []
        
        # Load existing metrics if file exists
        self._load_existing_metrics()
    
    def _load_existing_metrics(self):
        """Load metrics from file if exists"""
        if Path(self.log_file).exists():
            try:
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    self.metrics = data.get("metrics", [])
                print(f"📊 Loaded {len(self.metrics)} existing metrics from {self.log_file}")
            except Exception as e:
                print(f"⚠️ Could not load metrics file: {e}")
                self.metrics = []
    
    def log(self, query: str, response_data: Dict[str, Any]):
        """
        Log query metrics
        
        Args:
            query: User query string
            response_data: Dict with keys: total_latency, search_time, rerank_time, 
                          embedding_time, num_results, results, cached
        """
        # Extract top score safely
        top_score = 0.0
        results = response_data.get("results", [])
        if results and len(results) > 0:
            first_result = results[0]
            if isinstance(first_result, dict):
                top_score = first_result.get("score", 0.0)
        
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_length": len(query.split()),
            "total_latency": response_data.get("total_latency", 0),
            "search_time": response_data.get("search_time", 0),
            "rerank_time": response_data.get("rerank_time", 0),
            "embedding_time": response_data.get("embedding_time", 0),
            "generation_time": response_data.get("generation_time", 0),  # If you track this
            "num_results": response_data.get("num_results", 0),
            "top_score": top_score,
            "cached": response_data.get("cached", False)
        }
        
        self.metrics.append(entry)
        
        # Save to file after each log (ensures persistence)
        self._save_to_file()
    
    def _save_to_file(self):
        """Save metrics to JSON file"""
        try:
            data = {
                "last_updated": datetime.now().isoformat(),
                "total_queries": len(self.metrics),
                "metrics": self.metrics
            }
            with open(self.log_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Error saving metrics: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get aggregate statistics
        
        Returns:
            Dict with aggregate metrics
        """
        if not self.metrics:
            return {
                "total_queries": 0,
                "avg_latency": 0,
                "median_latency": 0,
                "p95_latency": 0,
                "avg_search_time": 0,
                "avg_rerank_time": 0,
                "avg_embedding_time": 0,
                "cache_hit_rate": 0,
                "avg_top_score": 0,
                "avg_query_length": 0
            }
        
        # Extract values
        latencies = [m["total_latency"] for m in self.metrics]
        search_times = [m["search_time"] for m in self.metrics]
        rerank_times = [m["rerank_time"] for m in self.metrics]
        embedding_times = [m["embedding_time"] for m in self.metrics]
        top_scores = [m["top_score"] for m in self.metrics]
        query_lengths = [m["query_length"] for m in self.metrics]
        cached_count = sum(1 for m in self.metrics if m["cached"])
        
        # Calculate statistics
        stats = {
            "total_queries": len(self.metrics),
            "avg_latency": statistics.mean(latencies),
            "median_latency": statistics.median(latencies),
            "p95_latency": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies),
            "min_latency": min(latencies),
            "max_latency": max(latencies),
            "avg_search_time": statistics.mean(search_times),
            "avg_rerank_time": statistics.mean(rerank_times),
            "avg_embedding_time": statistics.mean(embedding_times),
            "cache_hit_rate": (cached_count / len(self.metrics)) * 100,
            "avg_top_score": statistics.mean(top_scores),
            "avg_query_length": statistics.mean(query_lengths),
            "total_cached": cached_count,
            "total_fresh": len(self.metrics) - cached_count
        }
        
        return stats
    
    def get_recent_metrics(self, n: int = 10) -> List[Dict]:
        """
        Get N most recent queries
        
        Args:
            n: Number of recent queries to return
            
        Returns:
            List of recent metric entries
        """
        return self.metrics[-n:] if len(self.metrics) >= n else self.metrics
    
    def get_slow_queries(self, threshold: float = 3.0) -> List[Dict]:
        """
        Get queries that took longer than threshold
        
        Args:
            threshold: Latency threshold in seconds
            
        Returns:
            List of slow queries
        """
        return [m for m in self.metrics if m["total_latency"] > threshold]
    
    def get_cache_performance(self) -> Dict[str, Any]:
        """Get detailed cache statistics"""
        if not self.metrics:
            return {"cache_hit_rate": 0, "hits": 0, "misses": 0}
        
        hits = sum(1 for m in self.metrics if m["cached"])
        misses = len(self.metrics) - hits
        
        return {
            "cache_hit_rate": (hits / len(self.metrics)) * 100,
            "total_hits": hits,
            "total_misses": misses,
            "avg_cached_latency": statistics.mean([m["total_latency"] for m in self.metrics if m["cached"]]) if hits > 0 else 0,
            "avg_fresh_latency": statistics.mean([m["total_latency"] for m in self.metrics if not m["cached"]]) if misses > 0 else 0
        }
    
    def export_report(self, filepath: str = None) -> str:
        """
        Export comprehensive metrics report
        
        Args:
            filepath: Optional custom filepath
            
        Returns:
            Path to exported file
        """
        if filepath is None:
            filepath = f"metrics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "summary": self.get_stats(),
            "cache_performance": self.get_cache_performance(),
            "slow_queries": self.get_slow_queries(),
            "recent_queries": self.get_recent_metrics(20),
            "all_metrics": self.metrics
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✅ Exported metrics report to {filepath}")
        return filepath
    
    def clear_metrics(self):
        """Clear all metrics (use with caution!)"""
        self.metrics = []
        self._save_to_file()
        print("🗑️ Cleared all metrics")
    
    def print_summary(self):
        """Print formatted summary to console"""
        stats = self.get_stats()
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE METRICS SUMMARY")
        print("="*60)
        print(f"Total Queries:        {stats['total_queries']}")
        print(f"Average Latency:      {stats['avg_latency']:.3f}s")
        print(f"Median Latency:       {stats['median_latency']:.3f}s")
        print(f"P95 Latency:          {stats['p95_latency']:.3f}s")
        print(f"Min/Max Latency:      {stats['min_latency']:.3f}s / {stats['max_latency']:.3f}s")
        print(f"\nBreakdown:")
        print(f"  Search Time:        {stats['avg_search_time']:.3f}s")
        print(f"  Rerank Time:        {stats['avg_rerank_time']:.3f}s")
        print(f"  Embedding Time:     {stats['avg_embedding_time']:.3f}s")
        print(f"\nCache Performance:")
        print(f"  Hit Rate:           {stats['cache_hit_rate']:.1f}%")
        print(f"  Cached:             {stats['total_cached']}")
        print(f"  Fresh:              {stats['total_fresh']}")
        print(f"\nQuality Metrics:")
        print(f"  Avg Top Score:      {stats['avg_top_score']:.3f}")
        print(f"  Avg Query Length:   {stats['avg_query_length']:.1f} words")
        print("="*60 + "\n")


# ==========================================
# INTEGRATION EXAMPLE
# ==========================================

if __name__ == "__main__":
    print("=== Testing MetricsLogger ===\n")
    
    # Initialize logger
    logger = MetricsLogger("test_metrics.json")
    
    # Simulate some queries
    test_queries = [
        ("What are OLEDs?", {"total_latency": 1.5, "search_time": 0.3, "rerank_time": 0.8, "embedding_time": 0.1, "num_results": 5, "results": [{"score": 8.5}], "cached": False}),
        ("How do OLEDs work?", {"total_latency": 1.2, "search_time": 0.2, "rerank_time": 0.7, "embedding_time": 0.1, "num_results": 5, "results": [{"score": 7.8}], "cached": False}),
        ("What are OLEDs?", {"total_latency": 0.3, "search_time": 0.1, "rerank_time": 0.1, "embedding_time": 0.05, "num_results": 5, "results": [{"score": 8.5}], "cached": True}),  # Cached!
        ("OLED applications?", {"total_latency": 1.8, "search_time": 0.4, "rerank_time": 0.9, "embedding_time": 0.15, "num_results": 5, "results": [{"score": 7.2}], "cached": False}),
    ]
    
    for query, response_data in test_queries:
        logger.log(query, response_data)
        print(f"✅ Logged: {query}")
    
    # Print summary
    logger.print_summary()
    
    # Get cache performance
    cache_perf = logger.get_cache_performance()
    print(f"Cache Hit Rate: {cache_perf['cache_hit_rate']:.1f}%")
    print(f"Avg Cached Latency: {cache_perf['avg_cached_latency']:.3f}s")
    print(f"Avg Fresh Latency: {cache_perf['avg_fresh_latency']:.3f}s")
    print(f"Speed Improvement: {(1 - cache_perf['avg_cached_latency']/cache_perf['avg_fresh_latency'])*100:.1f}%\n")
    
    # Export report
    logger.export_report("test_report.json")
    
    print("\n✅ Test complete! Check test_metrics.json and test_report.json")