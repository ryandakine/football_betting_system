#!/usr/bin/env python3
"""
Practical GGUF Ensemble Manager
==============================

Efficiently manages multiple smaller GGUF models with intelligent loading/unloading.
Designed for football betting analysis with memory-conscious operation.
"""

import json
import logging
import time
import gc
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    Llama = None

@dataclass
class ModelConfig:
    """Configuration for a single GGUF model."""
    name: str
    filename: str
    specialty: str
    performance_weight: float
    size_gb: float
    max_context: int = 4096
    temperature: float = 0.3
    top_p: float = 0.9
    n_gpu_layers: int = -1

@dataclass 
class ModelStats:
    """Performance statistics for a model."""
    games_analyzed: int = 0
    correct_predictions: int = 0
    total_confidence: float = 0.0
    avg_response_time: float = 0.0
    last_used: float = 0.0
    
    @property
    def accuracy(self) -> float:
        return self.correct_predictions / max(1, self.games_analyzed)
    
    @property
    def avg_confidence(self) -> float:
        return self.total_confidence / max(1, self.games_analyzed)

class PracticalGGUFEnsemble:
    """Memory-efficient ensemble of smaller GGUF models."""
    
    def __init__(self, config_path: str = "models/ensemble_config.json"):
        self.config_path = Path(config_path)
        self.models_dir = Path("models/gguf/")
        self.loaded_models: Dict[str, Llama] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self.model_stats: Dict[str, ModelStats] = {}
        self.max_concurrent = 3
        self.load_lock = threading.Lock()
        
        self._load_config()
        self._initialize_stats()
        
        if not LLAMA_CPP_AVAILABLE:
            raise RuntimeError("llama-cpp-python not available")
    
    def _load_config(self):
        """Load ensemble configuration."""
        if not self.config_path.exists():
            logger.error(f"Config not found: {self.config_path}")
            raise FileNotFoundError("Run setup_practical_gguf_ensemble.py first")
        
        with open(self.config_path) as f:
            config = json.load(f)
        
        ensemble_config = config.get("gguf_ensemble", {})
        self.max_concurrent = ensemble_config.get("max_concurrent_models", 3)
        
        for model_data in ensemble_config.get("models", []):
            model_config = ModelConfig(
                name=model_data["name"],
                filename=model_data["filename"], 
                specialty=model_data["specialty"],
                performance_weight=model_data["performance_weight"],
                size_gb=model_data["size_gb"],
                max_context=model_data.get("max_context", 4096),
                temperature=model_data.get("temperature", 0.3),
                top_p=model_data.get("top_p", 0.9)
            )
            self.model_configs[model_config.name] = model_config
        
        logger.info(f"📦 Loaded config for {len(self.model_configs)} models")
    
    def _initialize_stats(self):
        """Initialize performance tracking."""
        for model_name in self.model_configs.keys():
            self.model_stats[model_name] = ModelStats()
    
    def _get_model_priority(self, model_name: str) -> float:
        """Calculate priority score for model loading."""
        config = self.model_configs[model_name]
        stats = self.model_stats[model_name]
        
        # Base score from performance weight
        score = config.performance_weight
        
        # Boost for accuracy (if we have data)
        if stats.games_analyzed > 5:
            accuracy_boost = stats.accuracy * 2.0
            confidence_boost = stats.avg_confidence * 1.0
            score += accuracy_boost + confidence_boost
        
        # Recent usage boost
        time_since_use = time.time() - stats.last_used
        if time_since_use < 3600:  # Used in last hour
            score += 1.0
        
        return score
    
    def _select_models_to_load(self, required_model: Optional[str] = None) -> List[str]:
        """Select which models to load based on priority."""
        all_models = list(self.model_configs.keys())
        
        if required_model and required_model in all_models:
            selected = [required_model]
            remaining_slots = self.max_concurrent - 1
        else:
            selected = []
            remaining_slots = self.max_concurrent
        
        # Sort by priority
        remaining_models = [m for m in all_models if m not in selected]
        remaining_models.sort(key=self._get_model_priority, reverse=True)
        
        selected.extend(remaining_models[:remaining_slots])
        return selected
    
    def _load_model(self, model_name: str) -> bool:
        """Load a specific model into memory."""
        if model_name in self.loaded_models:
            return True
        
        config = self.model_configs[model_name]
        model_path = self.models_dir / config.filename
        
        if not model_path.exists():
            logger.error(f"Model file not found: {model_path}")
            return False
        
        try:
            logger.info(f"🔄 Loading {model_name}...")
            start_time = time.time()
            
            model = Llama(
                model_path=str(model_path),
                n_ctx=config.max_context,
                n_gpu_layers=config.n_gpu_layers,
                verbose=False,
                n_threads=4,  # Limit threads for better concurrent performance
            )
            
            load_time = time.time() - start_time
            self.loaded_models[model_name] = model
            
            logger.info(f"✅ {model_name} loaded in {load_time:.1f}s")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_name}: {e}")
            return False
    
    def _unload_model(self, model_name: str):
        """Unload a model from memory."""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()  # Force garbage collection
            logger.info(f"🗑️ Unloaded {model_name}")
    
    def _manage_memory(self, required_models: List[str]):
        """Manage which models are loaded to stay within memory limits."""
        with self.load_lock:
            currently_loaded = set(self.loaded_models.keys())
            should_be_loaded = set(required_models)
            
            # Unload models not needed
            to_unload = currently_loaded - should_be_loaded
            for model_name in to_unload:
                self._unload_model(model_name)
            
            # Load required models
            for model_name in required_models:
                if model_name not in self.loaded_models:
                    self._load_model(model_name)
    
    def get_prediction(self, game_data: Dict[str, Any], model_name: Optional[str] = None) -> Dict[str, Any]:
        """Get prediction from a specific model or best available."""
        if model_name and model_name not in self.model_configs:
            logger.error(f"Unknown model: {model_name}")
            return None
        
        # Select models to use
        if model_name:
            target_models = [model_name]
        else:
            target_models = self._select_models_to_load()[:1]  # Just use the best one
        
        if not target_models:
            logger.error("No models available")
            return None
        
        # Ensure model is loaded
        self._manage_memory(target_models)
        
        model_name = target_models[0]
        if model_name not in self.loaded_models:
            logger.error(f"Failed to load {model_name}")
            return None
        
        return self._analyze_with_model(model_name, game_data)
    
    def get_ensemble_prediction(self, game_data: Dict[str, Any], num_models: int = 3) -> Dict[str, Any]:
        """Get ensemble prediction from multiple models."""
        selected_models = self._select_models_to_load()[:num_models]
        
        if not selected_models:
            logger.error("No models available for ensemble")
            return None
        
        # Load selected models
        self._manage_memory(selected_models)
        
        # Get predictions from all loaded models
        predictions = []
        for model_name in selected_models:
            if model_name in self.loaded_models:
                try:
                    pred = self._analyze_with_model(model_name, game_data)
                    if pred:
                        pred['model_name'] = model_name
                        pred['model_specialty'] = self.model_configs[model_name].specialty
                        predictions.append(pred)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed: {e}")
        
        if not predictions:
            logger.error("No successful predictions")
            return None
        
        # Combine predictions
        return self._combine_predictions(predictions, game_data)
    
    def _analyze_with_model(self, model_name: str, game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get prediction from a single model."""
        model = self.loaded_models[model_name]
        config = self.model_configs[model_name]
        
        # Create prompt
        prompt = self._create_prompt(game_data, config)
        
        try:
            start_time = time.time()
            
            response = model(
                prompt,
                max_tokens=400,
                temperature=config.temperature,
                top_p=config.top_p,
                stop=["</s>", "\nUser:", "\nHuman:"],
            )
            
            response_time = time.time() - start_time
            raw_text = response["choices"][0]["text"].strip()
            
            # Parse JSON response
            prediction = self._parse_response(raw_text)
            if prediction:
                # Update stats
                stats = self.model_stats[model_name]
                stats.last_used = time.time()
                stats.avg_response_time = (stats.avg_response_time * stats.games_analyzed + response_time) / (stats.games_analyzed + 1)
                
                prediction['response_time'] = response_time
                prediction['model_specialty'] = config.specialty
                return prediction
            
        except Exception as e:
            logger.error(f"Inference failed for {model_name}: {e}")
        
        return None
    
    def _create_prompt(self, game_data: Dict[str, Any], config: ModelConfig) -> str:
        """Create prompt optimized for the specific model."""
        home = game_data.get('home_team', 'Home')
        away = game_data.get('away_team', 'Away')
        
        # Specialized prompts based on model specialty
        if 'uncensored' in config.specialty.lower() or 'contrarian' in config.specialty.lower():
            system_msg = "You are an uncensored NFL analyst who reveals hidden patterns and biases."
        elif 'analytical' in config.specialty.lower() or 'structured' in config.specialty.lower():
            system_msg = "You are a quantitative NFL analyst focused on data-driven insights."
        elif 'sports' in config.specialty.lower() or 'betting' in config.specialty.lower():
            system_msg = "You are a professional sports betting analyst with expertise in NFL markets."
        else:
            system_msg = "You are an expert NFL analyst providing detailed game analysis."
        
        context_data = {
            'home_team': home,
            'away_team': away,
            'season': game_data.get('season', '2024'),
            'week': game_data.get('week', 'TBD'),
            'spread': game_data.get('spread'),
            'total': game_data.get('total'),
            'referee': game_data.get('referee'),
        }
        
        prompt = f"""{system_msg}

Analyze this NFL matchup and respond in valid JSON format:

Game: {away} @ {home}
Context: {json.dumps(context_data, indent=2)}

Respond with this exact JSON structure:
{{
    "probability": 0.XX,
    "confidence": 0.XX,
    "key_factors": ["factor 1", "factor 2", "factor 3"],
    "analysis": "detailed reasoning in 2-3 sentences",
    "risk_level": "low/medium/high",
    "recommendation": "betting recommendation"
}}

JSON:"""
        
        return prompt
    
    def _parse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse JSON response from model."""
        try:
            # Find JSON in response
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = text[json_start:json_end]
                result = json.loads(json_text)
                
                # Validate required fields
                if all(key in result for key in ['probability', 'confidence', 'analysis']):
                    # Ensure proper types and ranges
                    result['probability'] = max(0.0, min(1.0, float(result['probability'])))
                    result['confidence'] = max(0.0, min(1.0, float(result['confidence'])))
                    result['key_factors'] = result.get('key_factors', [])
                    result['risk_level'] = result.get('risk_level', 'medium')
                    result['recommendation'] = result.get('recommendation', 'no strong opinion')
                    
                    return result
        
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse response: {e}")
        
        return None
    
    def _combine_predictions(self, predictions: List[Dict[str, Any]], game_data: Dict[str, Any]) -> Dict[str, Any]:
        """Combine multiple predictions into ensemble result."""
        if not predictions:
            return None
        
        # Weight predictions by model performance
        total_weight = 0.0
        weighted_prob = 0.0
        weighted_conf = 0.0
        all_factors = []
        
        for pred in predictions:
            model_name = pred['model_name']
            config = self.model_configs[model_name]
            stats = self.model_stats[model_name]
            
            # Calculate weight (base performance + accuracy if available)
            weight = config.performance_weight
            if stats.games_analyzed > 3:
                weight *= (1.0 + stats.accuracy)
            
            weighted_prob += pred['probability'] * weight
            weighted_conf += pred['confidence'] * weight
            total_weight += weight
            
            if pred.get('key_factors'):
                all_factors.extend(pred['key_factors'])
        
        # Normalize
        final_prob = weighted_prob / total_weight if total_weight > 0 else 0.5
        final_conf = weighted_conf / total_weight if total_weight > 0 else 0.5
        
        # Deduplicate factors
        unique_factors = []
        seen = set()
        for factor in all_factors:
            if factor and factor not in seen:
                unique_factors.append(factor)
                seen.add(factor)
        
        # Use analysis from highest-weight model
        best_pred = max(predictions, key=lambda p: self.model_configs[p['model_name']].performance_weight)
        
        return {
            'probability': final_prob,
            'confidence': final_conf,
            'key_factors': unique_factors[:6],
            'analysis': best_pred.get('analysis', ''),
            'risk_level': best_pred.get('risk_level', 'medium'),
            'recommendation': best_pred.get('recommendation', ''),
            'models_used': [p['model_name'] for p in predictions],
            'model_specialties': [p['model_specialty'] for p in predictions],
            'ensemble_size': len(predictions)
        }
    
    def update_model_performance(self, model_name: str, was_correct: bool, confidence_used: float):
        """Update model performance statistics."""
        if model_name in self.model_stats:
            stats = self.model_stats[model_name]
            stats.games_analyzed += 1
            if was_correct:
                stats.correct_predictions += 1
            stats.total_confidence += confidence_used
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current status of all models."""
        status = {
            'loaded_models': list(self.loaded_models.keys()),
            'available_models': len(self.model_configs),
            'max_concurrent': self.max_concurrent,
            'model_stats': {}
        }
        
        for name, stats in self.model_stats.items():
            status['model_stats'][name] = {
                'games_analyzed': stats.games_analyzed,
                'accuracy': round(stats.accuracy, 3) if stats.games_analyzed > 0 else 0,
                'avg_confidence': round(stats.avg_confidence, 3) if stats.games_analyzed > 0 else 0,
                'avg_response_time': round(stats.avg_response_time, 2),
                'specialty': self.model_configs[name].specialty,
                'loaded': name in self.loaded_models
            }
        
        return status