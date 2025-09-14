#!/usr/bin/env python3
"""
MLB Betting System Agent Coordinator
===================================
Coordinates background agents and integrates with n8n workflows and Supabase.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from supabase_client import MLBSupabaseClient
from supabase_config import SupabaseConfig

logger = logging.getLogger(__name__)


@dataclass
class AgentStatus:
    """Status information for an agent."""

    agent_id: str
    status: str  # 'running', 'stopped', 'error'
    last_run: datetime | None = None
    next_run: datetime | None = None
    success_count: int = 0
    error_count: int = 0
    last_error: str | None = None


class AgentCoordinator:
    """Coordinates background agents for the MLB betting system."""

    def __init__(self):
        self.supabase_client = MLBSupabaseClient()
        self.agents: dict[str, AgentStatus] = {}
        self.running = False

        # Initialize agent statuses
        self._init_agents()

    def _init_agents(self):
        """Initialize agent statuses."""
        agent_configs = [
            {
                "id": "odds_monitor_001",
                "name": "Real-time Odds Monitor",
                "schedule": "30s",
                "type": "data_collection",
            },
            {
                "id": "sentiment_analyzer_001",
                "name": "Sentiment Analysis",
                "schedule": "15m",
                "type": "data_collection",
            },
            {
                "id": "performance_monitor_001",
                "name": "Performance Monitor",
                "schedule": "5m",
                "type": "monitoring",
            },
            {
                "id": "model_trainer_001",
                "name": "AI Model Trainer",
                "schedule": "1d",
                "type": "ai_enhancement",
            },
            {
                "id": "risk_manager_001",
                "name": "Risk Manager",
                "schedule": "10m",
                "type": "monitoring",
            },
            {
                "id": "market_analyzer_001",
                "name": "Market Analyzer",
                "schedule": "1h",
                "type": "analysis",
            },
        ]

        for config in agent_configs:
            self.agents[config["id"]] = AgentStatus(
                agent_id=config["id"], status="stopped"
            )

    async def start_agent_monitoring(self):
        """Start monitoring all agents."""
        self.running = True
        logger.info("🚀 Starting agent coordinator...")

        try:
            # Test Supabase connection
            if not await self.supabase_client.test_connection():
                logger.error("❌ Cannot connect to Supabase")
                return

            # Start monitoring loop
            while self.running:
                await self._monitor_agents()
                await asyncio.sleep(60)  # Check every minute

        except Exception as e:
            logger.error(f"❌ Agent coordinator error: {e}")
        finally:
            self.running = False

    async def _monitor_agents(self):
        """Monitor agent statuses and trigger actions."""
        try:
            # Get agent activity from Supabase
            agent_activity = await self._get_agent_activity()

            for agent_id, agent_status in self.agents.items():
                # Check if agent needs to be triggered
                if await self._should_trigger_agent(agent_id, agent_status):
                    await self._trigger_agent(agent_id)

                # Update agent status based on recent activity
                await self._update_agent_status(agent_id, agent_activity)

        except Exception as e:
            logger.error(f"❌ Error monitoring agents: {e}")

    async def _get_agent_activity(self) -> list[dict[str, Any]]:
        """Get recent agent activity from Supabase."""
        try:
            # Get activity from last hour
            one_hour_ago = (datetime.now() - timedelta(hours=1)).isoformat()

            # This would query the agent_activity table
            # For now, return empty list
            return []

        except Exception as e:
            logger.error(f"❌ Error getting agent activity: {e}")
            return []

    async def _should_trigger_agent(
        self, agent_id: str, agent_status: AgentStatus
    ) -> bool:
        """Determine if an agent should be triggered."""
        try:
            # Get agent schedule
            schedule = self._get_agent_schedule(agent_id)

            if not agent_status.last_run:
                return True

            # Calculate next run time
            next_run = agent_status.last_run + schedule

            if datetime.now() >= next_run:
                return True

            return False

        except Exception as e:
            logger.error(f"❌ Error checking agent trigger: {e}")
            return False

    def _get_agent_schedule(self, agent_id: str) -> timedelta:
        """Get the schedule for an agent."""
        schedules = {
            "odds_monitor_001": timedelta(seconds=30),
            "sentiment_analyzer_001": timedelta(minutes=15),
            "performance_monitor_001": timedelta(minutes=5),
            "model_trainer_001": timedelta(days=1),
            "risk_manager_001": timedelta(minutes=10),
            "market_analyzer_001": timedelta(hours=1),
        }

        return schedules.get(agent_id, timedelta(minutes=30))

    async def _trigger_agent(self, agent_id: str):
        """Trigger an agent via n8n webhook."""
        try:
            # Update agent status
            self.agents[agent_id].status = "running"
            self.agents[agent_id].last_run = datetime.now()

            # Trigger n8n workflow
            webhook_url = self._get_agent_webhook_url(agent_id)

            if webhook_url:
                # This would make an HTTP request to trigger the n8n workflow
                logger.info(f"🔗 Triggering agent {agent_id} via webhook")

                # For now, just log the trigger
                await self._log_agent_trigger(agent_id)

        except Exception as e:
            logger.error(f"❌ Error triggering agent {agent_id}: {e}")
            self.agents[agent_id].status = "error"
            self.agents[agent_id].last_error = str(e)
            self.agents[agent_id].error_count += 1

    def _get_agent_webhook_url(self, agent_id: str) -> str | None:
        """Get the webhook URL for an agent."""
        # These would be your n8n webhook URLs
        webhook_urls = {
            "odds_monitor_001": "https://your-n8n-instance.com/webhook/odds-monitor",
            "sentiment_analyzer_001": "https://your-n8n-instance.com/webhook/sentiment-analyzer",
            "performance_monitor_001": "https://your-n8n-instance.com/webhook/performance-monitor",
            "model_trainer_001": "https://your-n8n-instance.com/webhook/model-trainer",
            "risk_manager_001": "https://your-n8n-instance.com/webhook/risk-manager",
            "market_analyzer_001": "https://your-n8n-instance.com/webhook/market-analyzer",
        }

        return webhook_urls.get(agent_id)

    async def _log_agent_trigger(self, agent_id: str):
        """Log agent trigger in Supabase."""
        try:
            trigger_data = {
                "agent_id": agent_id,
                "activity_type": "agent_triggered",
                "data": {
                    "trigger_time": datetime.now().isoformat(),
                    "status": "triggered",
                },
                "status": "active",
            }

            # Store in agent_activity table
            await self.supabase_client.supabase.table("agent_activity").insert(
                trigger_data
            ).execute()

        except Exception as e:
            logger.error(f"❌ Error logging agent trigger: {e}")

    async def _update_agent_status(
        self, agent_id: str, agent_activity: list[dict[str, Any]]
    ):
        """Update agent status based on recent activity."""
        try:
            # Find recent activity for this agent
            recent_activity = [
                activity
                for activity in agent_activity
                if activity.get("agent_id") == agent_id
            ]

            if recent_activity:
                latest_activity = recent_activity[-1]

                if latest_activity.get("status") == "completed":
                    self.agents[agent_id].status = "stopped"
                    self.agents[agent_id].success_count += 1
                elif latest_activity.get("status") == "error":
                    self.agents[agent_id].status = "error"
                    self.agents[agent_id].error_count += 1
                    self.agents[agent_id].last_error = latest_activity.get(
                        "data", {}
                    ).get("error")

        except Exception as e:
            logger.error(f"❌ Error updating agent status: {e}")

    async def get_agent_status(self) -> dict[str, AgentStatus]:
        """Get status of all agents."""
        return self.agents

    async def get_system_health(self) -> dict[str, Any]:
        """Get overall system health."""
        try:
            total_agents = len(self.agents)
            running_agents = sum(
                1 for agent in self.agents.values() if agent.status == "running"
            )
            error_agents = sum(
                1 for agent in self.agents.values() if agent.status == "error"
            )

            # Get recent performance metrics
            recent_performance = await self._get_recent_performance()

            return {
                "timestamp": datetime.now().isoformat(),
                "total_agents": total_agents,
                "running_agents": running_agents,
                "error_agents": error_agents,
                "success_rate": (
                    (total_agents - error_agents) / total_agents
                    if total_agents > 0
                    else 0
                ),
                "recent_performance": recent_performance,
                "system_status": "healthy" if error_agents == 0 else "degraded",
            }

        except Exception as e:
            logger.error(f"❌ Error getting system health: {e}")
            return {
                "timestamp": datetime.now().isoformat(),
                "system_status": "error",
                "error": str(e),
            }

    async def _get_recent_performance(self) -> dict[str, Any]:
        """Get recent performance metrics."""
        try:
            # Get performance from last 24 hours
            yesterday = (datetime.now() - timedelta(days=1)).isoformat()

            # This would query performance metrics from Supabase
            # For now, return placeholder data
            return {
                "total_predictions": 0,
                "accuracy_rate": 0.0,
                "roi_percentage": 0.0,
                "total_pnl": 0.0,
            }

        except Exception as e:
            logger.error(f"❌ Error getting recent performance: {e}")
            return {}

    async def stop(self):
        """Stop the agent coordinator."""
        self.running = False
        logger.info("🛑 Stopping agent coordinator...")

    async def restart_agent(self, agent_id: str):
        """Restart a specific agent."""
        try:
            if agent_id in self.agents:
                self.agents[agent_id].status = "stopped"
                self.agents[agent_id].last_error = None
                logger.info(f"🔄 Restarted agent {agent_id}")
            else:
                logger.error(f"❌ Agent {agent_id} not found")

        except Exception as e:
            logger.error(f"❌ Error restarting agent {agent_id}: {e}")


# Convenience functions for external use
async def start_agent_coordinator():
    """Start the agent coordinator."""
    coordinator = AgentCoordinator()
    await coordinator.start_agent_monitoring()


async def get_agent_status():
    """Get status of all agents."""
    coordinator = AgentCoordinator()
    return await coordinator.get_agent_status()


async def get_system_health():
    """Get system health status."""
    coordinator = AgentCoordinator()
    return await coordinator.get_system_health()


# Example usage
async def main():
    """Example usage of the agent coordinator."""
    print("🤖 MLB Betting System Agent Coordinator")
    print("=" * 40)

    # Validate configuration
    try:
        SupabaseConfig.validate_config()
        print("✅ Supabase configuration is valid")
    except ValueError as e:
        print(f"❌ Configuration error: {e}")
        return

    # Start coordinator
    coordinator = AgentCoordinator()

    try:
        # Start monitoring
        await coordinator.start_agent_monitoring()
    except KeyboardInterrupt:
        print("\n🛑 Stopping coordinator...")
        await coordinator.stop()
    except Exception as e:
        print(f"❌ Coordinator error: {e}")


if __name__ == "__main__":
    asyncio.run(main())
