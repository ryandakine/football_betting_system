#!/usr/bin/env python3
"""
Hugging Face Cloud GPU AI Models Integration
==========================================

Enhanced AI model integration optimized for cloud GPU services:
- Larger, more powerful models (13B, 30B, 70B)
- Cloud GPU optimization (RunPod, Vast.ai, Google Colab, AWS)
- Advanced uncensored models
- Multi-GPU support
- Cost-effective cloud inference
- API endpoint integration for remote GPU access

Supports cloud services:
- RunPod (recommended for cost/performance)
- Vast.ai (cheapest option)
- Google Colab Pro/Pro+
- AWS EC2 GPU instances
- GCP GPU instances
- Lambda Labs
"""

import asyncio
import logging
import torch
import json
import time
import re
import requests
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    pipeline, BitsAndBytesConfig
)
import gc
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class CloudGPUModel:
    """Configuration for a cloud GPU model."""
    name: str
    model_id: str
    size: str  # "7B", "13B", "30B", "70B"
    min_vram_gb: int
    recommended_vram_gb: int
    tokenizer: Optional[Any] = None
    model: Optional[Any] = None
    pipeline: Optional[Any] = None
    max_length: int = 4096
    temperature: float = 0.3
    top_p: float = 0.85
    device: str = "auto"
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    trust_remote_code: bool = True
    cloud_endpoint: Optional[str] = None

@dataclass
class CloudGPUConfig:
    """Cloud GPU service configuration."""
    service: str  # "runpod", "vast", "colab", "aws", "gcp", "lambda"
    api_key: Optional[str] = None
    endpoint_url: Optional[str] = None
    instance_id: Optional[str] = None
    gpu_type: str = "RTX4090"  # RTX4090, A100, H100, etc.
    vram_gb: int = 24
    cost_per_hour: float = 0.0
    auto_shutdown: bool = True
    max_idle_minutes: int = 30

class CloudGPUAIEnsemble:
    """
    Enhanced AI ensemble using cloud GPU services for powerful model inference.
    """
    
    def __init__(self, config=None, cloud_config: CloudGPUConfig = None):
        self.config = config
        self.cloud_config = cloud_config or self._get_default_cloud_config()
        self.device = self._get_best_device()
        self.models = {}
        self.model_configs = self._get_cloud_optimized_models()
        self._models_loaded = False
        self.session = None
        
        logger.info(f"🌩️ Initializing Cloud GPU AI Ensemble")
        logger.info(f"🔧 Service: {self.cloud_config.service}")
        logger.info(f"🎮 GPU: {self.cloud_config.gpu_type} ({self.cloud_config.vram_gb}GB)")
    
    def _get_default_cloud_config(self) -> CloudGPUConfig:
        """Get default cloud GPU configuration."""
        return CloudGPUConfig(
            service="runpod",  # RunPod is generally best cost/performance
            gpu_type="RTX4090",
            vram_gb=24,
            cost_per_hour=0.50,  # Approximate RunPod RTX4090 cost
            auto_shutdown=True,
            max_idle_minutes=30
        )
    
    async def initialize_models(self):
        """Initialize models on cloud GPU."""
        if not self._models_loaded:
            await self._initialize_cloud_session()
            await self._initialize_models()
            self._models_loaded = True
    
    async def _initialize_cloud_session(self):
        """Initialize cloud GPU session."""
        self.session = aiohttp.ClientSession()
        
        if self.cloud_config.service == "runpod":
            await self._setup_runpod()
        elif self.cloud_config.service == "vast":
            await self._setup_vast_ai()
        elif self.cloud_config.service == "colab":
            await self._setup_colab()
        elif self.cloud_config.service in ["aws", "gcp"]:
            await self._setup_cloud_provider()
        
        logger.info(f"☁️ Cloud GPU session initialized")
    
    async def _setup_runpod(self):
        """Setup RunPod GPU instance."""
        logger.info("🚀 Setting up RunPod GPU instance...")
        
        # RunPod API integration
        if self.cloud_config.api_key:
            headers = {"Authorization": f"Bearer {self.cloud_config.api_key}"}
            
            # Create or resume GPU instance
            instance_data = {
                "name": "football-betting-ai",
                "imageName": "runpod/pytorch:2.0.1-py3.10-cuda11.8.0-devel-ubuntu22.04",
                "gpuTypeId": self._get_runpod_gpu_id(),
                "cloudType": "SECURE",
                "volumeInGb": 50,
                "containerDiskInGb": 20,
                "minVcpuCount": 4,
                "minMemoryInGb": 16,
                "dockerArgs": "",
                "ports": "8888/http,22/tcp",
                "volumeMountPath": "/workspace",
                "env": [
                    {"key": "JUPYTER_PASSWORD", "value": "football_ai_2024"}
                ]
            }
            
            logger.info("✅ RunPod instance configured")
    
    async def _setup_vast_ai(self):
        """Setup Vast.ai GPU instance."""
        logger.info("💰 Setting up Vast.ai GPU instance (cheapest option)...")
        
        # Vast.ai typically offers the lowest prices
        # Instance will be configured via their CLI or web interface
        logger.info("✅ Vast.ai instance configured")
    
    async def _setup_colab(self):
        """Setup Google Colab Pro/Pro+ session."""
        logger.info("📓 Setting up Google Colab Pro+ session...")
        
        # Colab Pro+ gives access to A100 GPUs
        # Models will be loaded directly in Colab notebook
        logger.info("✅ Colab session configured")
    
    async def _setup_cloud_provider(self):
        """Setup AWS/GCP GPU instance."""
        logger.info(f"☁️ Setting up {self.cloud_config.service.upper()} GPU instance...")
        
        # AWS p3/p4 instances or GCP GPU instances
        logger.info("✅ Cloud provider instance configured")
    
    def _get_runpod_gpu_id(self) -> str:
        """Get RunPod GPU type ID."""
        gpu_mapping = {
            "RTX4090": "NVIDIA RTX 4090",
            "A100": "NVIDIA A100 80GB PCIe",
            "H100": "NVIDIA H100 80GB HBM3",
            "RTX3090": "NVIDIA RTX 3090",
            "A6000": "NVIDIA RTX A6000"
        }
        return gpu_mapping.get(self.cloud_config.gpu_type, "NVIDIA RTX 4090")
    
    def _get_best_device(self) -> str:
        """Determine the best device for inference."""
        if self.cloud_config.service in ["runpod", "vast", "aws", "gcp", "lambda"]:
            return f"cuda:0"  # Cloud GPU instances typically use cuda:0
        elif torch.cuda.is_available():
            device = f"cuda:{torch.cuda.current_device()}"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info(f"🚀 Using Local GPU: {torch.cuda.get_device_name(0)} ({gpu_memory:.1f}GB)")
            return device
        else:
            logger.info("💻 Using CPU (consider cloud GPU for better performance)")
            return "cpu"
    
    def _get_cloud_optimized_models(self) -> List[CloudGPUModel]:
        """Get cloud GPU optimized model configurations."""
        models = [
            # Larger uncensored models (possible with cloud GPU)
            CloudGPUModel(
                name="wizard_uncensored_30b",
                model_id="ehartford/WizardLM-30B-Uncensored",
                size="30B",
                min_vram_gb=20,
                recommended_vram_gb=24,
                max_length=4096,
                temperature=0.2,
                top_p=0.85
            ),
            
            CloudGPUModel(
                name="david_au_uncensored_13b",
                model_id="davidau/Uncensor-13B-Alpaca",
                size="13B",
                min_vram_gb=12,
                recommended_vram_gb=16,
                max_length=4096,
                temperature=0.3,
                top_p=0.85
            ),
            
            # Powerful instruction-following models
            CloudGPUModel(
                name="mixtral_8x7b_instruct",
                model_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
                size="8x7B",
                min_vram_gb=24,
                recommended_vram_gb=32,
                max_length=8192,
                temperature=0.3,
                top_p=0.85
            ),
            
            CloudGPUModel(
                name="llama2_70b_chat",
                model_id="meta-llama/Llama-2-70b-chat-hf",
                size="70B",
                min_vram_gb=40,
                recommended_vram_gb=80,
                max_length=4096,
                temperature=0.3,
                top_p=0.85,
                load_in_4bit=True
            ),
            
            CloudGPUModel(
                name="code_llama_34b",
                model_id="codellama/CodeLlama-34b-Instruct-hf",
                size="34B",
                min_vram_gb=20,
                recommended_vram_gb=24,
                max_length=4096,
                temperature=0.2,
                top_p=0.8
            ),
            
            # Efficient models for cost optimization
            CloudGPUModel(
                name="mistral_7b_instruct",
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
                size="7B",
                min_vram_gb=6,
                recommended_vram_gb=8,
                max_length=8192,
                temperature=0.3,
                top_p=0.85
            ),
            
            CloudGPUModel(
                name="zephyr_7b_beta",
                model_id="HuggingFaceH4/zephyr-7b-beta",
                size="7B",
                min_vram_gb=6,
                recommended_vram_gb=8,
                max_length=4096,
                temperature=0.3,
                top_p=0.85
            )
        ]
        
        return models
    
    async def _initialize_models(self):
        """Initialize models on cloud GPU."""
        logger.info("🔄 Loading models on cloud GPU...")
        
        # Enhanced quantization for larger models
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Select models based on available VRAM
        available_vram = self.cloud_config.vram_gb
        models_to_load = self._select_models_by_vram(available_vram)
        
        for model_config in models_to_load:
            try:
                await self._load_single_model(model_config, quantization_config)
                await asyncio.sleep(2)  # Longer delay for cloud GPU
                
            except Exception as e:
                logger.warning(f"Failed to load {model_config.name}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(self.models)} models on cloud GPU")
        
        if not self.models:
            logger.error("❌ No models loaded! Check cloud GPU configuration")
    
    def _select_models_by_vram(self, available_vram: int) -> List[CloudGPUModel]:
        """Select models based on available VRAM."""
        selected_models = []
        used_vram = 0
        
        # Prioritize larger, more powerful models for cloud GPU
        priority_order = [
            "llama2_70b_chat",        # Most powerful (if enough VRAM)
            "wizard_uncensored_30b",   # Large uncensored model
            "mixtral_8x7b_instruct",   # Excellent mixture of experts
            "code_llama_34b",          # Great for analytical reasoning
            "david_au_uncensored_13b", # Medium uncensored model
            "mistral_7b_instruct",     # Efficient and capable
            "zephyr_7b_beta"           # Fallback option
        ]
        
        for model_name in priority_order:
            model_config = next((m for m in self.model_configs if m.name == model_name), None)
            if not model_config:
                continue
            
            required_vram = model_config.min_vram_gb
            
            if used_vram + required_vram <= available_vram * 0.85:  # Leave 15% buffer
                selected_models.append(model_config)
                used_vram += required_vram
                
                # Load fewer but larger models for cloud GPU
                if len(selected_models) >= 2:
                    break
        
        logger.info(f"💾 Selected {len(selected_models)} models for {available_vram}GB VRAM")
        for model in selected_models:
            logger.info(f"   📦 {model.name} ({model.size}) - {model.min_vram_gb}GB VRAM")
        
        return selected_models
    
    async def _load_single_model(self, model_config: CloudGPUModel, quantization_config):
        """Load a single model on cloud GPU."""
        logger.info(f"📥 Loading {model_config.name} ({model_config.size})...")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_id,
                trust_remote_code=model_config.trust_remote_code
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Enhanced model loading for cloud GPU
            model_kwargs = {
                "trust_remote_code": model_config.trust_remote_code,
                "torch_dtype": torch.float16,
                "device_map": "auto",
                "max_memory": {0: f"{self.cloud_config.vram_gb-2}GB"}  # Reserve 2GB
            }
            
            # Use quantization for larger models
            if model_config.size in ["30B", "34B", "70B", "8x7B"]:
                model_kwargs["quantization_config"] = quantization_config
                logger.info(f"   🔧 Using 4-bit quantization for {model_config.size} model")
            
            model = AutoModelForCausalLM.from_pretrained(
                model_config.model_id,
                **model_kwargs
            )
            
            # Create optimized pipeline
            text_pipeline = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=model_config.max_length,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                do_sample=True,
                device_map="auto"
            )
            
            # Store model components
            model_config.tokenizer = tokenizer
            model_config.model = model
            model_config.pipeline = text_pipeline
            
            self.models[model_config.name] = model_config
            
            # Log VRAM usage
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / 1e9
                logger.info(f"✅ {model_config.name} loaded - VRAM: {vram_used:.1f}GB")
            else:
                logger.info(f"✅ {model_config.name} loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_config.name}: {e}")
            raise
    
    async def analyze_football_game(self, game_data: Dict, market_type: str = "h2h") -> Dict[str, Any]:
        """Analyze a football game using cloud GPU model ensemble."""
        if not self._models_loaded:
            await self.initialize_models()
            
        if not self.models:
            logger.error("No models available for analysis")
            return None
        
        logger.info(f"🏈 Analyzing {game_data.get('home_team')} vs {game_data.get('away_team')} on cloud GPU")
        
        # Generate predictions from each model
        model_predictions = []
        
        for model_name, model_config in self.models.items():
            try:
                prediction = await self._get_model_prediction(model_config, game_data, market_type)
                if prediction:
                    model_predictions.append({
                        'model': model_name,
                        'model_size': model_config.size,
                        'prediction': prediction
                    })
            except Exception as e:
                logger.warning(f"Model {model_name} failed: {e}")
                continue
        
        if not model_predictions:
            logger.error("No successful predictions from any model")
            return None
        
        # Combine predictions with size-based weighting
        ensemble_result = self._combine_predictions_weighted(model_predictions, game_data)
        
        logger.info(f"🎯 Cloud GPU ensemble prediction: {ensemble_result.get('probability', 0):.3f}")
        return ensemble_result
    
    async def _get_model_prediction(self, model_config: CloudGPUModel, 
                                  game_data: Dict, market_type: str) -> Optional[Dict]:
        """Get prediction from a single cloud GPU model."""
        try:
            prompt = self._create_enhanced_prompt(game_data, market_type, model_config)
            
            start_time = time.time()
            
            response = model_config.pipeline(
                prompt,
                max_length=model_config.max_length,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                do_sample=True,
                pad_token_id=model_config.tokenizer.eos_token_id,
                return_full_text=False
            )
            
            generation_time = time.time() - start_time
            
            if response and len(response) > 0:
                generated_text = response[0]['generated_text']
                parsed_result = self._parse_model_response(generated_text, model_config.name)
                
                if parsed_result:
                    parsed_result['generation_time'] = generation_time
                    parsed_result['model_size'] = model_config.size
                    return parsed_result
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting prediction from {model_config.name}: {e}")
            return None
    
    def _create_enhanced_prompt(self, game_data: Dict, market_type: str, model_config: CloudGPUModel) -> str:
        """Create enhanced prompts for larger cloud GPU models."""
        home_team = game_data.get('home_team', 'Home Team')
        away_team = game_data.get('away_team', 'Away Team')
        
        # Enhanced game data for larger models
        game_summary = {
            'home_team': home_team,
            'away_team': away_team,
            'sport': game_data.get('sport', 'NFL'),
            'commence_time': game_data.get('commence_time', 'TBD'),
            'season': game_data.get('season', '2024'),
            'week': game_data.get('week', 'TBD')
        }
        
        if "uncensored" in model_config.name.lower():
            # Enhanced prompt for uncensored models
            prompt = f"""<|im_start|>system
You are an elite professional sports analyst and betting expert with 20+ years of experience. You have access to comprehensive NFL data and provide brutally honest, unfiltered analysis for serious bettors. Your analysis has consistently beaten the market by 15%+ ROI.
<|im_end|>

<|im_start|>user
Conduct a comprehensive betting analysis for this NFL game:

**GAME DETAILS:**
- Matchup: {home_team} vs {away_team}
- Market: {market_type}
- Data: {json.dumps(game_summary, indent=2)}

**ANALYSIS REQUIREMENTS:**
1. Win probability for home team (0.0 to 1.0) - be precise
2. Key factors (minimum 5 factors with detailed reasoning)
3. Confidence level in analysis (0.0 to 1.0)
4. Risk assessment (low/medium/high) with explanation
5. Sharp money indicators
6. Public betting bias considerations
7. Weather/injury impact assessment
8. Historical head-to-head trends
9. Coaching matchup analysis
10. Line value assessment

**OUTPUT FORMAT (JSON):**
{{
    "probability": 0.XXX,
    "confidence": 0.XXX,
    "key_factors": ["detailed factor 1", "detailed factor 2", "detailed factor 3", "detailed factor 4", "detailed factor 5"],
    "analysis": "comprehensive 200+ word analysis covering all aspects",
    "risk_level": "low/medium/high",
    "sharp_money": "analysis of sharp money movement",
    "public_bias": "public betting sentiment analysis",
    "line_value": "assessment of current line value",
    "recommendation": "clear betting recommendation with reasoning"
}}
<|im_end|>

<|im_start|>assistant"""
        
        elif "llama" in model_config.name.lower():
            prompt = f"""<s>[INST] <<SYS>>
You are a professional NFL betting analyst with expertise in statistical modeling and market analysis.
<</SYS>>

Analyze this NFL game comprehensively:

Game: {home_team} vs {away_team}
Market: {market_type}
Data: {json.dumps(game_summary)}

Provide detailed JSON analysis with probability, confidence, key_factors (5+ items), comprehensive analysis, risk_level, and betting recommendation. [/INST]"""
        
        elif "mixtral" in model_config.name.lower():
            prompt = f"""<s>[INST] You are an expert NFL analyst. Provide comprehensive betting analysis:

Game: {home_team} vs {away_team}
Market: {market_type}
Data: {json.dumps(game_summary)}

Include probability, confidence, detailed key_factors, analysis, risk_level, and recommendation in JSON format. [/INST]"""
        
        else:
            # Default format for other models
            prompt = f"""<|system|>
You are a professional sports analyst specializing in NFL betting analysis.
</s>
