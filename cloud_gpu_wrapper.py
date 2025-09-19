#!/usr/bin/env python3
"""
Cloud GPU Wrapper for Ollama Models
Provides cloud GPU offload using Google Colab with fallback to local Ollama
"""

import asyncio
import aiohttp
import logging
import json
import time
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import os
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ModelProvider(Enum):
    """Model provider types."""
    LOCAL = "local"
    CLOUD = "cloud"
    FALLBACK = "fallback"


@dataclass
class CloudConfig:
    """Cloud GPU configuration."""
    enabled: bool = False
    ngrok_url: Optional[str] = None
    fallback_to_local: bool = True
    timeout_seconds: int = 30
    retry_attempts: int = 3
    health_check_interval: int = 60  # seconds
    models: List[str] = None
    
    def __post_init__(self):
        if self.models is None:
            self.models = [
                "davidau/openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix-gguf:q4_k_m",
                "dolphin-llama3:8b",
                "nous-hermes-2-mixtral-dpo:8x7b-q4_k_m"
            ]


@dataclass
class ModelResponse:
    """Response from model inference."""
    content: str
    provider: ModelProvider
    model_name: str
    response_time: float
    success: bool
    error: Optional[str] = None


class CloudGPUWrapper:
    """Wrapper for cloud GPU Ollama integration with fallback."""
    
    def __init__(self, config: CloudConfig):
        self.config = config
        self.local_url = "http://localhost:11434"
        self.cloud_url = config.ngrok_url
        self.current_provider = ModelProvider.CLOUD if config.enabled and config.ngrok_url else ModelProvider.LOCAL
        self.health_status = {"cloud": False, "local": False}
        self.last_health_check = 0
        
    async def initialize(self) -> bool:
        """Initialize the wrapper and check service availability."""
        try:
            # Check local Ollama
            local_available = await self._check_service_health(self.local_url)
            self.health_status["local"] = local_available
            
            # Check cloud Ollama if configured
            if self.config.enabled and self.cloud_url:
                cloud_available = await self._check_service_health(self.cloud_url)
                self.health_status["cloud"] = cloud_available
                
                if cloud_available:
                    self.current_provider = ModelProvider.CLOUD
                    logger.info(f"Cloud GPU service available at {self.cloud_url}")
                elif local_available and self.config.fallback_to_local:
                    self.current_provider = ModelProvider.LOCAL
                    logger.info("Cloud unavailable, falling back to local Ollama")
                else:
                    logger.error("Neither cloud nor local Ollama services available")
                    return False
            else:
                if local_available:
                    self.current_provider = ModelProvider.LOCAL
                    logger.info("Using local Ollama service")
                else:
                    logger.error("Local Ollama service not available")
                    return False
            
            self.last_health_check = time.time()
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize cloud GPU wrapper: {e}")
            return False
    
    async def _check_service_health(self, base_url: str) -> bool:
        """Check if Ollama service is healthy."""
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5)) as session:
                async with session.get(f"{base_url}/api/tags") as response:
                    return response.status == 200
        except Exception as e:
            logger.debug(f"Health check failed for {base_url}: {e}")
            return False
    
    async def _periodic_health_check(self):
        """Periodic health check to update service status."""
        current_time = time.time()
        if current_time - self.last_health_check < self.config.health_check_interval:
            return
        
        # Check both services
        local_health = await self._check_service_health(self.local_url)
        self.health_status["local"] = local_health
        
        if self.config.enabled and self.cloud_url:
            cloud_health = await self._check_service_health(self.cloud_url)
            self.health_status["cloud"] = cloud_health
            
            # Update current provider based on health status
            if self.current_provider == ModelProvider.LOCAL and cloud_health:
                self.current_provider = ModelProvider.CLOUD
                logger.info("Cloud service recovered, switching back to cloud")
            elif self.current_provider == ModelProvider.CLOUD and not cloud_health and local_health:
                self.current_provider = ModelProvider.LOCAL
                logger.warning("Cloud service failed, falling back to local")
        
        self.last_health_check = current_time
    
    def get_current_base_url(self) -> str:
        """Get the current base URL based on provider."""
        if self.current_provider == ModelProvider.CLOUD and self.cloud_url:
            return self.cloud_url
        return self.local_url
    
    async def generate_completion(self, 
                                prompt: str, 
                                model: str = "dolphin-llama3:8b",
                                temperature: float = 0.7,
                                max_tokens: int = 1000) -> ModelResponse:
        """Generate completion with automatic fallback."""
        await self._periodic_health_check()
        
        start_time = time.time()
        
        for attempt in range(self.config.retry_attempts):
            try:
                base_url = self.get_current_base_url()
                response = await self._call_ollama_api(
                    base_url, prompt, model, temperature, max_tokens
                )
                
                response_time = time.time() - start_time
                
                return ModelResponse(
                    content=response,
                    provider=self.current_provider,
                    model_name=model,
                    response_time=response_time,
                    success=True
                )
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                # Try fallback if cloud fails
                if (self.current_provider == ModelProvider.CLOUD and 
                    self.config.fallback_to_local and 
                    self.health_status["local"]):
                    
                    logger.info("Trying local fallback")
                    try:
                        response = await self._call_ollama_api(
                            self.local_url, prompt, model, temperature, max_tokens
                        )
                        
                        response_time = time.time() - start_time
                        
                        return ModelResponse(
                            content=response,
                            provider=ModelProvider.FALLBACK,
                            model_name=model,
                            response_time=response_time,
                            success=True
                        )
                    except Exception as fallback_error:
                        logger.error(f"Fallback also failed: {fallback_error}")
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        response_time = time.time() - start_time
        return ModelResponse(
            content="",
            provider=self.current_provider,
            model_name=model,
            response_time=response_time,
            success=False,
            error="All retry attempts failed"
        )
    
    async def _call_ollama_api(self, 
                             base_url: str,
                             prompt: str,
                             model: str,
                             temperature: float,
                             max_tokens: int) -> str:
        """Call Ollama API directly."""
        url = f"{base_url}/api/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        
        timeout = aiohttp.ClientTimeout(total=self.config.timeout_seconds)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    raise Exception(f"API call failed with status {response.status}")
                
                result = await response.json()
                return result.get("response", "")
    
    async def test_connection(self, test_prompt: str = "Predict Bills game outcome") -> Dict[str, Any]:
        """Test connection with all available providers."""
        results = {
            "timestamp": time.time(),
            "test_prompt": test_prompt,
            "providers": {}
        }
        
        # Test cloud if available
        if self.config.enabled and self.cloud_url and self.health_status["cloud"]:
            try:
                original_provider = self.current_provider
                self.current_provider = ModelProvider.CLOUD
                
                response = await self.generate_completion(test_prompt)
                results["providers"]["cloud"] = {
                    "success": response.success,
                    "response_time": response.response_time,
                    "content_length": len(response.content),
                    "error": response.error
                }
                
                self.current_provider = original_provider
                
            except Exception as e:
                results["providers"]["cloud"] = {
                    "success": False,
                    "error": str(e)
                }
        
        # Test local if available
        if self.health_status["local"]:
            try:
                original_provider = self.current_provider
                self.current_provider = ModelProvider.LOCAL
                
                response = await self.generate_completion(test_prompt)
                results["providers"]["local"] = {
                    "success": response.success,
                    "response_time": response.response_time,
                    "content_length": len(response.content),
                    "error": response.error
                }
                
                self.current_provider = original_provider
                
            except Exception as e:
                results["providers"]["local"] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get current wrapper status."""
        return {
            "current_provider": self.current_provider.value,
            "cloud_enabled": self.config.enabled,
            "cloud_url": self.cloud_url,
            "local_url": self.local_url,
            "health_status": self.health_status,
            "last_health_check": self.last_health_check,
            "available_models": self.config.models
        }


class AICouncilCloudWrapper:
    """Enhanced AI Council with cloud GPU support."""
    
    def __init__(self, cloud_config: CloudConfig):
        self.cloud_wrapper = CloudGPUWrapper(cloud_config)
        self.council_models = [
            "dolphin-llama3:8b",
            "nous-hermes-2-mixtral-dpo:8x7b-q4_k_m",
            "davidau/openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix-gguf:q4_k_m"
        ]
        self.current_model_index = 0
    
    async def initialize(self) -> bool:
        """Initialize the cloud wrapper."""
        return await self.cloud_wrapper.initialize()
    
    async def get_council_prediction(self, 
                                   prompt: str,
                                   use_rotation: bool = True) -> Dict[str, Any]:
        """Get prediction using council rotation or single model."""
        if use_rotation:
            return await self._get_rotated_prediction(prompt)
        else:
            response = await self.cloud_wrapper.generate_completion(
                prompt, self.council_models[0]
            )
            return {
                "prediction": response.content,
                "model": response.model_name,
                "provider": response.provider.value,
                "response_time": response.response_time,
                "success": response.success
            }
    
    async def _get_rotated_prediction(self, prompt: str) -> Dict[str, Any]:
        """Get prediction using model rotation."""
        model = self.council_models[self.current_model_index]
        self.current_model_index = (self.current_model_index + 1) % len(self.council_models)
        
        response = await self.cloud_wrapper.generate_completion(prompt, model)
        
        return {
            "prediction": response.content,
            "model": response.model_name,
            "provider": response.provider.value,
            "response_time": response.response_time,
            "success": response.success,
            "rotation_index": self.current_model_index
        }
    
    async def get_ensemble_prediction(self, prompt: str) -> Dict[str, Any]:
        """Get predictions from all council models."""
        predictions = []
        total_time = 0
        
        for model in self.council_models:
            response = await self.cloud_wrapper.generate_completion(prompt, model)
            predictions.append({
                "model": model,
                "prediction": response.content,
                "provider": response.provider.value,
                "response_time": response.response_time,
                "success": response.success
            })
            total_time += response.response_time
        
        return {
            "ensemble_predictions": predictions,
            "total_response_time": total_time,
            "successful_predictions": sum(1 for p in predictions if p["success"]),
            "timestamp": time.time()
        }


# Configuration loader
def load_cloud_config() -> CloudConfig:
    """Load cloud configuration from environment variables."""
    return CloudConfig(
        enabled=os.getenv("CLOUD_GPU_ENABLED", "false").lower() == "true",
        ngrok_url=os.getenv("NGROK_URL"),
        fallback_to_local=os.getenv("FALLBACK_TO_LOCAL", "true").lower() == "true",
        timeout_seconds=int(os.getenv("CLOUD_TIMEOUT", "30")),
        retry_attempts=int(os.getenv("CLOUD_RETRY_ATTEMPTS", "3")),
        health_check_interval=int(os.getenv("HEALTH_CHECK_INTERVAL", "60"))
    )


# Example usage and testing
async def test_cloud_wrapper():
    """Test the cloud wrapper functionality."""
    config = load_cloud_config()
    wrapper = CloudGPUWrapper(config)
    
    print("Initializing cloud GPU wrapper...")
    if await wrapper.initialize():
        print("✅ Wrapper initialized successfully")
        
        # Test connection
        test_results = await wrapper.test_connection("Predict Bills game outcome")
        print(f"🧪 Test results: {json.dumps(test_results, indent=2)}")
        
        # Get status
        status = wrapper.get_status()
        print(f"📊 Status: {json.dumps(status, indent=2)}")
        
        # Test AI Council
        council = AICouncilCloudWrapper(config)
        await council.initialize()
        
        prediction = await council.get_council_prediction(
            "Analyze the Buffalo Bills vs Kansas City Chiefs matchup. Consider recent performance, injuries, and weather."
        )
        print(f"🏈 Council prediction: {json.dumps(prediction, indent=2)}")
        
    else:
        print("❌ Failed to initialize wrapper")


if __name__ == "__main__":
    asyncio.run(test_cloud_wrapper())
