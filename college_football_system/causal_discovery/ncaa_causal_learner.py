#!/usr/bin/env python3
"""
NCAA Causal Discovery Pipeline
===============================
Uses causal-learn and DoWhy to discover actual cause-effect relationships
in your 10 years of NCAA football data (2015-2024).

Not just correlations - actual causal paths that predict game outcomes.
"""

import pandas as pd
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pickle

# Causal discovery
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ScoreBased.GES import ges
from causallearn.utils.cit import fisherz

logger = logging.getLogger(__name__)


class NCAACousalLearner:
    """
    Discovers causal relationships in NCAA game data.
    
    Example causal paths it might discover:
    - temperature → passing_yards → total_score
    - rest_days → injury_risk → team_performance
    - referee_penalty_rate → red_team_aggressive → total_yards
    - key_injury → defensive_efficiency → allowed_points
    """
    
    def __init__(self, 
                 data_path: Optional[str] = None,
                 cache_dir: str = "models/causal_models"):
        self.data_path = data_path
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.causal_graph = None
        self.causal_paths = {}
        self.load_cache()
    
    def load_cache(self):
        """Load cached causal models"""
        graph_file = self.cache_dir / "causal_graph.pkl"
        paths_file = self.cache_dir / "causal_paths.json"
        
        if graph_file.exists():
            try:
                with open(graph_file, 'rb') as f:
                    self.causal_graph = pickle.load(f)
                    logger.info("✅ Loaded cached causal graph")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load causal graph: {e}")
        
        if paths_file.exists():
            try:
                with open(paths_file, 'r') as f:
                    self.causal_paths = json.load(f)
                    logger.info("✅ Loaded cached causal paths")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load causal paths: {e}")
    
    def discover_causality(self, 
                          data: pd.DataFrame,
                          method: str = 'pc',
                          confidence_level: float = 0.05) -> Dict[str, Any]:
        """
        Discover causal relationships using constraint-based (PC) or score-based (GES) methods.
        
        Args:
            data: DataFrame with NCAA game features and outcomes
            method: 'pc' (default, faster) or 'ges' (more conservative)
            confidence_level: alpha value for conditional independence tests
        
        Returns:
            Dictionary with causal graph, discovered paths, and statistics
        """
        logger.info(f"🔍 Starting causal discovery ({method.upper()} algorithm)...")
        logger.info(f"   Data shape: {data.shape}")
        logger.info(f"   Confidence level: {confidence_level}")
        
        # Standardize data for causal learning
        data_std = (data - data.mean()) / data.std()
        
        # Run causal discovery
        if method == 'pc':
            cg = self._run_pc_algorithm(data_std, confidence_level)
        elif method == 'ges':
            cg = self._run_ges_algorithm(data_std)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        self.causal_graph = cg
        
        # Extract causal relationships
        results = self._extract_causal_relationships(cg, data.columns)
        
        # Save results
        self._save_results(cg, results)
        
        return results
    
    def _run_pc_algorithm(self, data: np.ndarray, alpha: float) -> Any:
        """
        PC Algorithm (Constraint-Based)
        - Faster, good for large datasets
        - Discovers conditional independencies
        - Good for discovering d-separation relationships
        """
        logger.info("   🔄 Running PC algorithm...")
        try:
            cg = pc.pc(
                data.values,
                alpha=alpha,
                indep_test=fisherz,  # Use Fisher Z test for continuous data
                stable=True,  # Stable version
                uc_rule=0  # Conservative
            )
            logger.info("   ✅ PC algorithm completed")
            return cg
        except Exception as e:
            logger.error(f"   ❌ PC algorithm failed: {e}")
            raise
    
    def _run_ges_algorithm(self, data: np.ndarray) -> Any:
        """
        GES Algorithm (Score-Based)
        - More conservative, fewer false positives
        - Uses BIC or other scoring function
        - Good for ground truth discovery
        """
        logger.info("   🔄 Running GES algorithm...")
        try:
            cg = ges.ges(
                data.values,
                score_func='bic'  # BIC score for continuous data
            )
            logger.info("   ✅ GES algorithm completed")
            return cg
        except Exception as e:
            logger.error(f"   ❌ GES algorithm failed: {e}")
            raise
    
    def _extract_causal_relationships(self, 
                                     cg: Any,
                                     columns: List[str]) -> Dict[str, Any]:
        """
        Extract interpretable causal relationships from the causal graph.
        """
        logger.info("   📊 Extracting causal paths...")
        
        # Get adjacency matrix
        adj_matrix = cg.G
        
        # Find causal edges (directed edges, not bidirectional)
        causal_edges = []
        for i in range(len(columns)):
            for j in range(len(columns)):
                if i != j:
                    # Check for causal relationship: i → j
                    if adj_matrix[i, j] == 1 and adj_matrix[j, i] == 0:
                        causal_edges.append({
                            'cause': columns[i],
                            'effect': columns[j],
                            'type': 'direct_causal'
                        })
        
        # Find confounders (i ← ← j)
        confounders = []
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                if adj_matrix[i, j] == 2 and adj_matrix[j, i] == 2:
                    confounders.append({
                        'var1': columns[i],
                        'var2': columns[j],
                        'type': 'confounded'
                    })
        
        # Find mediators (paths like i → k → j)
        mediators = []
        for i in range(len(columns)):
            for k in range(len(columns)):
                for j in range(len(columns)):
                    if i != k != j:
                        if (adj_matrix[i, k] == 1 and adj_matrix[k, i] == 0 and
                            adj_matrix[k, j] == 1 and adj_matrix[j, k] == 0):
                            mediators.append({
                                'source': columns[i],
                                'mediator': columns[k],
                                'target': columns[j],
                                'type': 'mediation'
                            })
        
        results = {
            'direct_causal_edges': causal_edges,
            'confounders': confounders,
            'mediators': mediators,
            'edge_count': len(causal_edges),
            'confounder_count': len(confounders),
            'mediator_count': len(mediators),
            'total_relationships': len(causal_edges) + len(confounders) + len(mediators)
        }
        
        logger.info(f"   Found {len(causal_edges)} direct causal edges")
        logger.info(f"   Found {len(confounders)} confounder pairs")
        logger.info(f"   Found {len(mediators)} mediation paths")
        
        self.causal_paths = results
        return results
    
    def _save_results(self, cg: Any, results: Dict[str, Any]):
        """Save causal models to disk"""
        try:
            # Save graph
            graph_file = self.cache_dir / "causal_graph.pkl"
            with open(graph_file, 'wb') as f:
                pickle.dump(cg, f)
            
            # Save relationships
            paths_file = self.cache_dir / "causal_paths.json"
            with open(paths_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"   ✅ Saved to {self.cache_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to save results: {e}")
    
    def get_predictive_paths(self) -> Dict[str, List[Dict]]:
        """
        Get causal paths most relevant to game outcome prediction.
        Filters for paths that include game result variables.
        """
        if not self.causal_paths:
            return {'causal_edges': [], 'mediation_paths': []}
        
        game_outcome_vars = ['final_score', 'spread_prediction', 'cover', 
                            'total_points', 'home_score', 'away_score']
        
        predictive = {
            'causal_edges': [],
            'mediation_paths': []
        }
        
        # Find causal edges pointing to game outcomes
        for edge in self.causal_paths.get('direct_causal_edges', []):
            if edge['effect'] in game_outcome_vars:
                predictive['causal_edges'].append({
                    'predictor': edge['cause'],
                    'target': edge['effect'],
                    'strength': 'direct',
                    'action': f"Monitor {edge['cause']} as direct predictor of {edge['effect']}"
                })
        
        # Find mediation paths to game outcomes
        for med in self.causal_paths.get('mediators', []):
            if med['target'] in game_outcome_vars:
                predictive['mediation_paths'].append({
                    'path': f"{med['source']} → {med['mediator']} → {med['target']}",
                    'intermediate': med['mediator'],
                    'strength': 'mediated',
                    'action': f"Model {med['source']} effect through {med['mediator']}"
                })
        
        return predictive
    
    def apply_causal_adjustments(self, 
                               prediction: float,
                               causal_context: Dict[str, float]) -> Tuple[float, Dict[str, Any]]:
        """
        Adjust a prediction based on active causal paths.
        
        Args:
            prediction: Base confidence score
            causal_context: Current values of causal variables
        
        Returns:
            (adjusted_prediction, adjustment_details)
        """
        adjusted = prediction
        details = {
            'base': prediction,
            'adjustments': [],
            'total_adjustment': 0.0
        }
        
        predictive = self.get_predictive_paths()
        
        for edge in predictive.get('causal_edges', []):
            var_name = edge['predictor']
            if var_name in causal_context:
                # Strong causal edges get higher weight
                adjustment = causal_context[var_name] * 0.08  # 8% per causal edge
                adjusted *= (1 + adjustment)
                details['adjustments'].append({
                    'variable': var_name,
                    'value': causal_context[var_name],
                    'adjustment': adjustment
                })
                details['total_adjustment'] += adjustment
        
        # Cap at 98%
        adjusted = min(adjusted, 0.98)
        details['final'] = adjusted
        
        return adjusted, details
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize causal model state"""
        return {
            'total_relationships': self.causal_paths.get('total_relationships', 0),
            'causal_edges': len(self.causal_paths.get('direct_causal_edges', [])),
            'confounders': len(self.causal_paths.get('confounders', [])),
            'mediators': len(self.causal_paths.get('mediators', [])),
            'predictive_paths': len(self.get_predictive_paths().get('causal_edges', [])),
            'sample_edges': self.causal_paths.get('direct_causal_edges', [])[:3]
        }


# Utility function for batch causal discovery
def discover_causality_from_file(csv_path: str, 
                               output_dir: str = "models/causal_models",
                               method: str = 'pc') -> Dict[str, Any]:
    """
    Quick wrapper to discover causality from a CSV file.
    
    Usage:
        results = discover_causality_from_file('data/ncaa_history_2015_2024.csv')
        print(results)
    """
    logger.info(f"📂 Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)
    
    learner = NCAACousalLearner(data_path=csv_path, cache_dir=output_dir)
    results = learner.discover_causality(data, method=method)
    
    return results
