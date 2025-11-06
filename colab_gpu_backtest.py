#!/usr/bin/env python3
"""
Google Colab GPU GGUF Backtesting Integration
===========================================

Optimized script to run GGUF ensemble backtesting on Google Colab GPU,
maximizing your 200 paid GPU hours with efficient resource management.

Key Features:
- Smart GPU memory management
- Batch processing to minimize overhead
- Progress tracking and intermediate saves
- Resource monitoring and optimization
- Auto-recovery from interruptions
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import subprocess
import psutil
import gc

# Colab-specific imports
try:
    from google.colab import drive, files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Import our local backtesting pipeline
from gguf_ensemble_backtest import GGUFEnsembleBacktester, EnsembleBacktestResult

class ColabGPUManager:
    """Manages GPU resources and optimization for Colab environment"""
    
    def __init__(self):
        self.gpu_available = self._check_gpu_availability()
        self.session_start_time = time.time()
        self.max_session_hours = 12  # Colab GPU limit
        
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and what type"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                gpu_info = result.stdout.strip().split('\n')[0]
                print(f"🚀 GPU Detected: {gpu_info}")
                return True
        except FileNotFoundError:
            pass
        
        print("⚠️ No GPU detected - running on CPU (will be much slower)")
        return False
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU and system memory usage"""
        gpu_memory = {"used": 0, "total": 0, "free": 0}
        
        if self.gpu_available:
            try:
                result = subprocess.run([
                    'nvidia-smi', '--query-gpu=memory.used,memory.total', 
                    '--format=csv,noheader,nounits'
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    used, total = map(int, result.stdout.strip().split(', '))
                    gpu_memory = {
                        "used": used,
                        "total": total, 
                        "free": total - used
                    }
            except Exception as e:
                print(f"Warning: Could not get GPU memory info: {e}")
        
        # System memory
        system_memory = psutil.virtual_memory()
        
        return {
            "gpu_memory_mb": gpu_memory,
            "system_memory_mb": {
                "used": system_memory.used / 1024 / 1024,
                "total": system_memory.total / 1024 / 1024,
                "percent": system_memory.percent
            },
            "session_hours": (time.time() - self.session_start_time) / 3600
        }
    
    def optimize_for_gpu(self):
        """Optimize system settings for GPU usage"""
        if not self.gpu_available:
            return
            
        print("🔧 Optimizing for GPU usage...")
        
        # Clear any existing GPU memory
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                print("   ✅ Cleared PyTorch GPU cache")
        except ImportError:
            pass
            
        # Force garbage collection
        gc.collect()
        print("   ✅ Cleared system memory")
    
    def should_continue_session(self) -> bool:
        """Check if we should continue or save and exit"""
        elapsed_hours = (time.time() - self.session_start_time) / 3600
        
        if elapsed_hours > (self.max_session_hours - 0.5):  # Leave 30 min buffer
            print(f"⏰ Session approaching {self.max_session_hours}h limit, should save progress")
            return False
            
        memory_info = self.get_memory_usage()
        
        # Check if we're running out of GPU memory
        if self.gpu_available:
            gpu_usage = memory_info["gpu_memory_mb"]
            if gpu_usage["free"] < 500:  # Less than 500MB free
                print("⚠️ Low GPU memory, should optimize or save progress")
                
        return True

class ColabGGUFBacktester(GGUFEnsembleBacktester):
    """Extended backtester optimized for Google Colab environment"""
    
    def __init__(self, drive_path: str = "/content/drive"):
        # Set paths for Colab environment
        if IN_COLAB:
            self.drive_path = Path(drive_path)
            models_dir = self.drive_path / "MyDrive/football_betting_models/gguf"
            data_dir = self.drive_path / "MyDrive/football_betting_data"
            results_dir = self.drive_path / "MyDrive/football_betting_results"
        else:
            models_dir = Path("models/gguf")  
            data_dir = Path("data/football/historical")
            results_dir = Path("reports/backtesting")
            
        super().__init__(models_dir=models_dir, data_dir=data_dir)
        self.results_dir = results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.gpu_manager = ColabGPUManager()
        self.checkpoint_frequency = 100  # Save progress every N games
        
    async def run_colab_optimized_backtest(self, seasons: List[str] = None, 
                                          resume_from: str = None) -> EnsembleBacktestResult:
        """Run backtesting optimized for Colab GPU environment"""
        
        print("🚀 COLAB GPU OPTIMIZED GGUF BACKTESTING")
        print("=" * 60)
        
        # Setup Colab environment
        await self._setup_colab_environment()
        
        # Optimize for GPU usage
        self.gpu_manager.optimize_for_gpu()
        
        # Load or resume from checkpoint
        if resume_from:
            print(f"📂 Resuming from checkpoint: {resume_from}")
            checkpoint_data = await self._load_checkpoint(resume_from)
            if checkpoint_data:
                return await self._resume_backtest_from_checkpoint(checkpoint_data)
        
        # Start fresh backtest with checkpointing
        return await self._run_checkpointed_backtest(seasons)
    
    async def _setup_colab_environment(self):
        """Setup Google Drive and dependencies in Colab"""
        if not IN_COLAB:
            return
            
        print("🔧 Setting up Colab environment...")
        
        # Mount Google Drive
        try:
            drive.mount('/content/drive', force_remount=True)
            print("   ✅ Google Drive mounted")
        except Exception as e:
            print(f"   ⚠️ Drive mount failed: {e}")
        
        # Install required packages
        required_packages = [
            "llama-cpp-python",
            "pandas",
            "numpy", 
            "sqlite3"
        ]
        
        for package in required_packages:
            try:
                subprocess.run([
                    "pip", "install", package, "-q"
                ], check=True)
                print(f"   ✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"   ⚠️ Failed to install {package}")
    
    async def _run_checkpointed_backtest(self, seasons: List[str] = None) -> EnsembleBacktestResult:
        """Run backtest with regular checkpointing to avoid data loss"""
        
        if not seasons:
            seasons = ["2021", "2022", "2023"]
            
        print(f"📊 Starting backtesting for seasons: {seasons}")
        
        # Load historical data
        historical_data = await self._load_historical_data(seasons)
        print(f"📈 Loaded {len(historical_data)} historical games")
        
        # Load models with GPU optimization
        await self._load_models_gpu_optimized()
        
        # Initialize results structure
        results = {
            "individual_results": {},
            "ensemble_predictions": [],
            "processed_games": 0,
            "total_games": len(historical_data),
            "seasons": seasons,
            "start_time": time.time()
        }
        
        # Process each model with checkpointing
        for model_key, config in self.model_configs.items():
            if model_key not in self.loaded_models:
                continue
                
            print(f"\n🤖 Processing {config.name}...")
            
            model_result = await self._backtest_model_with_checkpoints(
                model_key, historical_data, results
            )
            results["individual_results"][model_key] = model_result
            
            # Check if we should continue
            if not self.gpu_manager.should_continue_session():
                print("💾 Saving progress and exiting due to resource limits")
                await self._save_checkpoint(results)
                return self._create_partial_results(results)
        
        # Run ensemble analysis
        print("\n🎭 Running ensemble analysis...")
        ensemble_result = await self._backtest_ensemble_checkpointed(historical_data, results)
        
        # Create final results
        final_result = EnsembleBacktestResult(
            individual_results=results["individual_results"],
            ensemble_result=ensemble_result,
            consensus_analysis=await self._analyze_consensus(historical_data),
            model_agreement_rates=await self._calculate_agreement_rates(historical_data)
        )
        
        # Save final results
        await self._save_results(final_result)
        self._display_results(final_result)
        
        return final_result
    
    async def _load_models_gpu_optimized(self):
        """Load models with GPU optimization and memory management"""
        print("\n🔧 Loading GGUF Models (GPU Optimized)...")
        
        # Sort models by priority (load most important first)
        model_priority = ["mistral", "neural_chat", "codellama", "openchat", "dolphin"]
        
        for model_key in model_priority:
            if model_key not in self.model_configs:
                continue
                
            config = self.model_configs[model_key]
            
            if not config.file_path.exists():
                print(f"   ⚠️ Model file not found: {config.file_path}")
                continue
            
            print(f"   Loading {config.name}...")
            
            # Check available memory before loading
            memory_info = self.gpu_manager.get_memory_usage()
            print(f"     GPU Memory: {memory_info['gpu_memory_mb']['free']:.0f}MB free")
            
            try:
                # Optimize for GPU if available
                if self.gpu_manager.gpu_available:
                    model = Llama(
                        model_path=str(config.file_path),
                        n_ctx=config.n_ctx,
                        n_threads=min(8, psutil.cpu_count()),  # Use more threads on GPU
                        n_gpu_layers=32,  # Offload to GPU
                        verbose=False
                    )
                else:
                    # CPU fallback
                    model = Llama(
                        model_path=str(config.file_path),
                        n_ctx=config.n_ctx,
                        n_threads=config.n_threads,
                        verbose=False
                    )
                
                self.loaded_models[model_key] = model
                print(f"   ✅ {config.name} loaded successfully")
                
                # Monitor memory after loading
                memory_info = self.gpu_manager.get_memory_usage()
                print(f"     GPU Memory after load: {memory_info['gpu_memory_mb']['free']:.0f}MB free")
                
            except Exception as e:
                print(f"   ❌ Failed to load {config.name}: {e}")
                
                # If GPU memory is full, try CPU-only
                if "CUDA out of memory" in str(e) and self.gpu_manager.gpu_available:
                    print(f"     Retrying {config.name} on CPU...")
                    try:
                        model = Llama(
                            model_path=str(config.file_path),
                            n_ctx=config.n_ctx,
                            n_threads=config.n_threads,
                            n_gpu_layers=0,  # Force CPU
                            verbose=False
                        )
                        self.loaded_models[model_key] = model
                        print(f"   ✅ {config.name} loaded on CPU")
                    except Exception as e2:
                        print(f"   ❌ CPU fallback also failed: {e2}")
        
        print(f"🎉 Loaded {len(self.loaded_models)}/5 models successfully")
        
        # Display final memory usage
        memory_info = self.gpu_manager.get_memory_usage()
        print(f"📊 Final Memory Usage:")
        print(f"   GPU: {memory_info['gpu_memory_mb']['used']:.0f}MB used / {memory_info['gpu_memory_mb']['total']:.0f}MB total")
        print(f"   System: {memory_info['system_memory_mb']['percent']:.1f}% used")
    
    async def _backtest_model_with_checkpoints(self, model_key: str, historical_data: List[Dict[str, Any]], 
                                             results: Dict[str, Any]) -> Any:
        """Backtest individual model with checkpointing"""
        
        if model_key not in self.loaded_models:
            return None
            
        model = self.loaded_models[model_key]
        config = self.model_configs[model_key]
        
        predictions = []
        profits = []
        
        start_time = time.time()
        
        for i, game in enumerate(historical_data):
            # Progress reporting
            if i % 50 == 0:
                elapsed = time.time() - start_time
                games_per_sec = i / elapsed if elapsed > 0 else 0
                eta_seconds = (len(historical_data) - i) / games_per_sec if games_per_sec > 0 else 0
                
                memory_info = self.gpu_manager.get_memory_usage()
                
                print(f"   Progress: {i}/{len(historical_data)} ({i/len(historical_data)*100:.1f}%)")
                print(f"   Speed: {games_per_sec:.1f} games/sec | ETA: {eta_seconds/60:.1f} min")
                print(f"   GPU Memory: {memory_info['gpu_memory_mb']['free']:.0f}MB free")
                
                # Check if we should checkpoint
                if i > 0 and i % self.checkpoint_frequency == 0:
                    results["processed_games"] = i
                    results["partial_predictions"] = {model_key: predictions}
                    await self._save_checkpoint(results, f"checkpoint_{model_key}_{i}")
            
            # Check session limits
            if not self.gpu_manager.should_continue_session():
                print(f"🛑 Stopping {config.name} early due to resource limits")
                break
            
            # Generate prediction
            prediction = await self._get_model_prediction(model, config, game)
            
            if prediction and prediction.get("make_bet"):
                profit = self._calculate_bet_profit(prediction, game)
                profits.append(profit)
                predictions.append({
                    **prediction,
                    "actual_profit": profit,
                    "game_id": game["game_id"]
                })
        
        # Calculate final metrics for this model
        total_profit = sum(profits)
        total_bets = len(profits)
        winning_bets = len([p for p in profits if p > 0])
        
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        roi = (total_profit / (total_bets * 100)) * 100 if total_bets > 0 else 0
        
        sharpe_ratio = self._calculate_sharpe_ratio(profits)
        max_drawdown = self._calculate_max_drawdown(profits)
        confidence_accuracy = self._analyze_confidence_accuracy(predictions)
        
        from gguf_ensemble_backtest import BacktestResult
        return BacktestResult(
            model_name=config.name,
            total_games=len(historical_data),
            predictions_made=total_bets,
            correct_predictions=winning_bets,
            total_profit=total_profit,
            roi_percent=roi,
            win_rate=win_rate,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            confidence_accuracy=confidence_accuracy
        )
    
    async def _save_checkpoint(self, results: Dict[str, Any], checkpoint_name: str = None):
        """Save checkpoint to prevent data loss"""
        if not checkpoint_name:
            checkpoint_name = f"checkpoint_{int(time.time())}"
            
        checkpoint_file = self.results_dir / f"{checkpoint_name}.json"
        
        # Add metadata
        checkpoint_data = {
            **results,
            "checkpoint_time": time.time(),
            "session_info": self.gpu_manager.get_memory_usage()
        }
        
        with open(checkpoint_file, 'w') as f:
            json.dump(checkpoint_data, f, indent=2, default=str)
            
        print(f"💾 Checkpoint saved: {checkpoint_file}")
        
        # Also upload to Colab if available
        if IN_COLAB:
            try:
                files.download(str(checkpoint_file))
                print("📤 Checkpoint downloaded to local machine")
            except:
                pass  # Download might fail, but file is still saved
    
    async def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint data"""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            print(f"✅ Loaded checkpoint from {checkpoint_path}")
            return checkpoint_data
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            return None
    
    def _create_partial_results(self, results: Dict[str, Any]) -> EnsembleBacktestResult:
        """Create results from partial data when session is interrupted"""
        
        # Convert individual results
        individual_results = {}
        for model_key, result in results.get("individual_results", {}).items():
            if result:
                individual_results[model_key] = result
        
        # Create dummy ensemble result if not completed
        from gguf_ensemble_backtest import BacktestResult
        ensemble_result = BacktestResult(
            model_name="Partial-Ensemble",
            total_games=results.get("total_games", 0),
            predictions_made=0,
            correct_predictions=0,
            total_profit=0.0,
            roi_percent=0.0,
            win_rate=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            confidence_accuracy={}
        )
        
        return EnsembleBacktestResult(
            individual_results=individual_results,
            ensemble_result=ensemble_result,
            consensus_analysis={"status": "partial_results"},
            model_agreement_rates={}
        )

async def run_colab_backtest(seasons: List[str] = None, resume_from: str = None):
    """Main function to run GGUF backtesting on Google Colab"""
    
    print("🚀 Starting Colab GPU GGUF Backtesting")
    print("=" * 60)
    
    # Initialize Colab-optimized backtester
    backtester = ColabGGUFBacktester()
    
    # Run the backtest
    results = await backtester.run_colab_optimized_backtest(
        seasons=seasons, 
        resume_from=resume_from
    )
    
    # Display resource usage summary
    memory_info = backtester.gpu_manager.get_memory_usage()
    print(f"\n📊 Final Resource Usage:")
    print(f"   Session Time: {memory_info['session_hours']:.2f} hours")
    print(f"   GPU Memory Peak: {memory_info['gpu_memory_mb']['used']:.0f}MB")
    print(f"   System Memory: {memory_info['system_memory_mb']['percent']:.1f}%")
    
    print("\n🎉 Colab GPU backtesting complete!")
    print("💡 Results are saved to Google Drive for persistence")
    
    return results

# Example usage for Colab notebook
async def quick_colab_test():
    """Quick test function for Colab"""
    # Test with small dataset first
    results = await run_colab_backtest(seasons=["2023"])
    return results

if __name__ == "__main__":
    # Run the backtesting
    asyncio.run(run_colab_backtest())