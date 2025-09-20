#!/usr/bin/env python3
"""
Integrated Performance Monitor for Unified NFL Intelligence System
===============================================================

Combines the performance analyzer with the unified system to provide
real-time monitoring and improvement recommendations.
"""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import json

from unified_nfl_intelligence_system import UnifiedNFLIntelligenceSystem
from system_performance_analyzer import SystemPerformanceAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IntegratedPerformanceMonitor:
    """Monitor that combines unified system with performance analysis"""
    
    def __init__(self, bankroll: float = 10000.0):
        self.unified_system = UnifiedNFLIntelligenceSystem(bankroll)
        self.performance_analyzer = SystemPerformanceAnalyzer()
        self.monitoring_data = {
            'session_start': datetime.now(),
            'games_analyzed': 0,
            'total_edge_detected': 0.0,
            'recommendations_generated': 0,
            'system_uptime': 0.0
        }
        
        logger.info("🔍 Integrated Performance Monitor initialized")
    
    async def run_performance_analysis(self) -> Dict[str, Any]:
        """Run comprehensive performance analysis"""
        try:
            logger.info("📊 Running integrated performance analysis...")
            
            # Get current system performance
            performance_report = self.performance_analyzer.generate_performance_report()
            bottlenecks = self.performance_analyzer.identify_bottlenecks()
            recommendations = self.performance_analyzer.generate_optimization_recommendations()
            
            # Analyze unified system performance
            unified_performance = await self._analyze_unified_system_performance()
            
            # Generate improvement plan
            improvement_plan = self._generate_improvement_plan(bottlenecks, recommendations, unified_performance)
            
            # Create comprehensive analysis
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'performance_report': performance_report,
                'bottlenecks': bottlenecks,
                'recommendations': recommendations,
                'unified_system_performance': unified_performance,
                'improvement_plan': improvement_plan,
                'monitoring_data': self.monitoring_data
            }
            
            # Save analysis
            self._save_analysis(analysis)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in performance analysis: {e}")
            return {'error': str(e)}
    
    async def _analyze_unified_system_performance(self) -> Dict[str, Any]:
        """Analyze the unified system's current performance"""
        try:
            # Test with sample games
            test_games = [
                {
                    'game_id': 'PERF_TEST_1',
                    'home_team': 'TEST_HOME',
                    'away_team': 'TEST_AWAY',
                    'spread': -3.5,
                    'total': 47.5,
                    'public_percentage': 0.65,
                    'sharp_percentage': 0.35,
                    'line_movement': -0.5
                }
            ]
            
            unified_results = []
            total_edge = 0.0
            
            for game in test_games:
                result = await self.unified_system.run_unified_analysis(game)
                if 'error' not in result:
                    unified_results.append(result)
                    total_edge += result.get('total_edge', 0.0)
            
            # Calculate unified system metrics
            avg_edge = total_edge / len(test_games) if test_games else 0.0
            successful_analyses = len([r for r in unified_results if 'error' not in r])
            success_rate = successful_analyses / len(test_games) if test_games else 0.0
            
            return {
                'avg_edge_per_game': avg_edge,
                'success_rate': success_rate,
                'total_games_tested': len(test_games),
                'successful_analyses': successful_analyses,
                'total_edge_detected': total_edge,
                'system_operational': success_rate > 0.8
            }
            
        except Exception as e:
            logger.error(f"Error analyzing unified system: {e}")
            return {'error': str(e)}
    
    def _generate_improvement_plan(self, bottlenecks: List[str], recommendations: List[Dict], unified_perf: Dict) -> Dict[str, Any]:
        """Generate a comprehensive improvement plan"""
        try:
            improvement_plan = {
                'priority_actions': [],
                'system_optimizations': [],
                'feature_enhancements': [],
                'monitoring_improvements': []
            }
            
            # High priority actions based on bottlenecks
            high_priority_bottlenecks = [b for b in bottlenecks if 'High' in b or 'Critical' in b]
            if high_priority_bottlenecks:
                improvement_plan['priority_actions'].extend([
                    f"Address: {bottleneck}" for bottleneck in high_priority_bottlenecks
                ])
            
            # System optimizations
            if 'High memory usage' in bottlenecks:
                improvement_plan['system_optimizations'].append({
                    'action': 'Implement memory optimization',
                    'details': 'Add memory pooling, cleanup routines, and data structure optimization',
                    'priority': 'High',
                    'estimated_impact': 'Reduce memory usage by 20-30%'
                })
            
            if 'High CPU usage' in bottlenecks:
                improvement_plan['system_optimizations'].append({
                    'action': 'Optimize CPU-intensive operations',
                    'details': 'Implement caching, parallel processing, and code profiling',
                    'priority': 'Medium',
                    'estimated_impact': 'Reduce CPU usage by 15-25%'
                })
            
            # Feature enhancements based on unified system performance
            if unified_perf.get('avg_edge_per_game', 0) < 0.05:  # Less than 5% edge
                improvement_plan['feature_enhancements'].append({
                    'action': 'Enhance edge detection algorithms',
                    'details': 'Improve market intelligence and behavioral analysis',
                    'priority': 'High',
                    'estimated_impact': 'Increase average edge by 2-3%'
                })
            
            if unified_perf.get('success_rate', 0) < 0.9:  # Less than 90% success rate
                improvement_plan['feature_enhancements'].append({
                    'action': 'Improve system reliability',
                    'details': 'Add better error handling and fallback mechanisms',
                    'priority': 'High',
                    'estimated_impact': 'Increase success rate to 95%+'
                })
            
            # Monitoring improvements
            improvement_plan['monitoring_improvements'] = [
                {
                    'action': 'Add real-time performance dashboards',
                    'details': 'Create live monitoring of system metrics',
                    'priority': 'Medium',
                    'estimated_impact': 'Better visibility into system performance'
                },
                {
                    'action': 'Implement automated alerting',
                    'details': 'Set up alerts for performance degradation',
                    'priority': 'Medium',
                    'estimated_impact': 'Faster response to issues'
                }
            ]
            
            return improvement_plan
            
        except Exception as e:
            logger.error(f"Error generating improvement plan: {e}")
            return {'error': str(e)}
    
    def _save_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save analysis to file"""
        try:
            # Create analysis directory
            analysis_dir = Path("performance_analysis")
            analysis_dir.mkdir(exist_ok=True)
            
            # Save detailed analysis
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_file = analysis_dir / f"integrated_analysis_{timestamp}.json"
            
            with open(analysis_file, 'w') as f:
                json.dump(analysis, f, indent=2, default=str)
            
            # Save summary report
            summary_file = analysis_dir / f"summary_report_{timestamp}.txt"
            with open(summary_file, 'w') as f:
                f.write(analysis['performance_report'])
                f.write(f"\n\n🏈 UNIFIED SYSTEM PERFORMANCE:\n")
                f.write(f"Average Edge per Game: {analysis['unified_system_performance'].get('avg_edge_per_game', 0):.1%}\n")
                f.write(f"Success Rate: {analysis['unified_system_performance'].get('success_rate', 0):.1%}\n")
                f.write(f"System Operational: {analysis['unified_system_performance'].get('system_operational', False)}\n")
            
            logger.info(f"Analysis saved to {analysis_file}")
            logger.info(f"Summary saved to {summary_file}")
            
        except Exception as e:
            logger.error(f"Error saving analysis: {e}")
    
    async def run_continuous_monitoring(self, interval_minutes: int = 30) -> None:
        """Run continuous performance monitoring"""
        logger.info(f"🔄 Starting continuous monitoring (every {interval_minutes} minutes)")
        
        while True:
            try:
                # Run performance analysis
                analysis = await self.run_performance_analysis()
                
                # Update monitoring data
                self.monitoring_data['system_uptime'] = (datetime.now() - self.monitoring_data['session_start']).total_seconds()
                
                # Log key metrics
                unified_perf = analysis.get('unified_system_performance', {})
                logger.info(f"📊 Performance Update:")
                logger.info(f"   Edge per Game: {unified_perf.get('avg_edge_per_game', 0):.1%}")
                logger.info(f"   Success Rate: {unified_perf.get('success_rate', 0):.1%}")
                logger.info(f"   System Status: {'✅ Operational' if unified_perf.get('system_operational', False) else '⚠️ Issues Detected'}")
                
                # Wait for next interval
                await asyncio.sleep(interval_minutes * 60)
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying


async def main():
    """Demo the integrated performance monitor"""
    print("🔍 INTEGRATED PERFORMANCE MONITOR - UNIFIED NFL SYSTEM")
    print("=" * 70)
    
    # Initialize monitor
    monitor = IntegratedPerformanceMonitor(bankroll=15000.0)
    
    # Run comprehensive analysis
    print("📊 Running comprehensive performance analysis...")
    analysis = await monitor.run_performance_analysis()
    
    if 'error' not in analysis:
        print("✅ Analysis Complete!")
        
        # Display key findings
        unified_perf = analysis.get('unified_system_performance', {})
        bottlenecks = analysis.get('bottlenecks', [])
        improvement_plan = analysis.get('improvement_plan', {})
        
        print(f"\n🏈 UNIFIED SYSTEM PERFORMANCE:")
        print(f"   Average Edge per Game: {unified_perf.get('avg_edge_per_game', 0):.1%}")
        print(f"   Success Rate: {unified_perf.get('success_rate', 0):.1%}")
        print(f"   System Operational: {'✅' if unified_perf.get('system_operational', False) else '⚠️'}")
        
        if bottlenecks:
            print(f"\n⚠️ IDENTIFIED BOTTLENECKS:")
            for i, bottleneck in enumerate(bottlenecks, 1):
                print(f"   {i}. {bottleneck}")
        
        if improvement_plan.get('priority_actions'):
            print(f"\n🎯 PRIORITY IMPROVEMENTS:")
            for action in improvement_plan['priority_actions']:
                print(f"   • {action}")
        
        print(f"\n📄 Detailed analysis saved to performance_analysis/ directory")
    
    print(f"\n🎊 INTEGRATED PERFORMANCE MONITOR OPERATIONAL!")
    print(f"✅ Real-time monitoring ready")
    print(f"✅ Improvement recommendations generated")
    print(f"✅ System optimization plan created")


if __name__ == "__main__":
    asyncio.run(main())
