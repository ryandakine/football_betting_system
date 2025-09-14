"""
Neural Network for Team Chemistry Analysis
YOLO Mode Implementation - Advanced Deep Learning for Team Dynamics

Uses cutting-edge neural network architectures to analyze:
- Team chemistry and player interactions
- Formation effectiveness and matchup advantages
- Coaching tendencies and strategic patterns
- Player role adaptability and team synergy
- Hidden performance factors beyond traditional stats
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import logging
from datetime import datetime
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

class TeamChemistryDataset(Dataset):
    """Custom dataset for team chemistry analysis"""
    
    def __init__(self, features: pd.DataFrame, targets: pd.DataFrame, sequence_length: int = 5):
        self.features = features
        self.targets = targets
        self.sequence_length = sequence_length
        
        # Create sequences for temporal analysis
        self.sequences = []
        self.sequence_targets = []
        
        # Group by team and create rolling sequences
        for team in features["team_id"].unique():
            team_data = features[features["team_id"] == team].sort_values("week")
            team_targets = targets[targets["team_id"] == team].sort_values("week")
            
            if len(team_data) >= sequence_length:
                for i in range(len(team_data) - sequence_length + 1):
                    seq_features = team_data.iloc[i:i+sequence_length].drop("team_id", axis=1).values
                    seq_target = team_targets.iloc[i+sequence_length-1]["win_probability"]
                    
                    self.sequences.append(seq_features)
                    self.sequence_targets.append(seq_target)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "features": torch.tensor(self.sequences[idx], dtype=torch.float32),
            "target": torch.tensor(self.sequence_targets[idx], dtype=torch.float32)
        }

class TeamChemistryNetwork(nn.Module):
    """
    Advanced neural network for team chemistry analysis.
    YOLO Mode: Multiple specialized layers for different aspects of team dynamics.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 128, num_layers: int = 3, dropout: float = 0.3):
        super(TeamChemistryNetwork, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Multi-head attention for player interactions
        self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=8, dropout=dropout)
        
        # LSTM layers for temporal team dynamics
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )
        
        # Convolutional layers for pattern recognition
        self.conv1d = nn.Conv1d(in_channels=hidden_dim*2, out_channels=hidden_dim, kernel_size=3, padding=1)
        
        # Chemistry analysis layers
        self.chemistry_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()  # Output win probability
        )
        
        # Chemistry factors (interpretable outputs)
        self.chemistry_factors = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 5)  # 5 chemistry factors
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights using Xavier initialization"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight" in name:
                    nn.init.xavier_uniform_(param)
                elif "bias" in name:
                    nn.init.constant_(param, 0)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the team chemistry network.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
            Tuple of (win_probability, chemistry_factors)
        """
        batch_size, seq_len, _ = x.shape
        
        # Multi-head attention for player interactions
        # Reshape for attention: (seq_len, batch_size, input_dim)
        x_attn = x.permute(1, 0, 2)
        attn_output, _ = self.attention(x_attn, x_attn, x_attn)
        x_attn = attn_output.permute(1, 0, 2)  # Back to (batch_size, seq_len, input_dim)
        
        # LSTM for temporal patterns
        lstm_out, (h_n, c_n) = self.lstm(x_attn)
        
        # Use last hidden state from both directions
        last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch_size, hidden_dim * 2)
        
        # Convolutional feature extraction
        conv_input = lstm_out.permute(0, 2, 1)  # (batch_size, hidden_dim*2, seq_len)
        conv_output = self.conv1d(conv_input)
        conv_pooled = torch.mean(conv_output, dim=2)  # Global average pooling
        
        # Combine LSTM and convolutional features
        combined_features = last_hidden + conv_pooled
        
        # Chemistry prediction
        win_probability = self.chemistry_net(combined_features)
        
        # Chemistry factors (interpretable components)
        chemistry_factors = self.chemistry_factors(combined_features)
        
        return win_probability.squeeze(), chemistry_factors

class ChemistryFactorInterpreter:
    """Interpret chemistry factors into human-readable insights"""
    
    def __init__(self):
        self.factor_names = [
            "team_synergy",
            "coaching_effectiveness", 
            "player_adaptability",
            "defensive_chemistry",
            "situational_awareness"
        ]
    
    def interpret_factors(self, factors: np.ndarray) -> Dict[str, float]:
        """Convert raw factors to interpretable scores"""
        # Normalize factors to 0-100 scale
        normalized = (factors - factors.min()) / (factors.max() - factors.min()) * 100
        
        return dict(zip(self.factor_names, normalized))
    
    def get_chemistry_insights(self, factors: Dict[str, float]) -> List[str]:
        """Generate human-readable chemistry insights"""
        insights = []
        
        if factors["team_synergy"] > 75:
            insights.append("Excellent team synergy - players work together seamlessly")
        elif factors["team_synergy"] < 25:
            insights.append("Poor team synergy - players struggle to coordinate")
        
        if factors["coaching_effectiveness"] > 75:
            insights.append("Strong coaching impact - strategic decisions paying off")
        elif factors["coaching_effectiveness"] < 25:
            insights.append("Coaching struggles - tactical execution lacking")
        
        if factors["player_adaptability"] > 75:
            insights.append("High player adaptability - team adjusts well to situations")
        
        if factors["defensive_chemistry"] > 75:
            insights.append("Outstanding defensive chemistry - unit plays as one")
        
        if factors["situational_awareness"] > 75:
            insights.append("Excellent situational awareness - smart decision-making")
        
        return insights

class TeamChemistryAnalyzer:
    """
    Complete team chemistry analysis system.
    YOLO Mode: Production-ready deep learning pipeline.
    """
    
    def __init__(self, model_dir: str = "models/team_chemistry"):
        from pathlib import Path
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.interpreter = ChemistryFactorInterpreter()
        
        # Training parameters
        self.batch_size = 32
        self.learning_rate = 0.001
        self.num_epochs = 100
        self.patience = 10
        
    def prepare_team_features(self, player_df: pd.DataFrame, game_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare comprehensive team features from player and game data.
        """
        logger.info("Preparing team chemistry features")
        
        # Aggregate player stats by team and week
        team_features = []
        
        for _, game in game_df.iterrows():
            game_features = {
                "team_id": game["home_team"],
                "opponent_id": game["away_team"],
                "week": game["week"],
                "season": game["season"],
                "home_advantage": 1,
                "venue": game.get("venue", "unknown"),
                "weather_temp": game.get("weather_temp", 70),
                "spread_line": game.get("spread_line", 0),
                "game_result": game.get("home_score", 0) > game.get("away_score", 0)
            }
            
            # Add player stats for home team
            home_players = player_df[
                (player_df["team"] == game["home_team"]) & 
                (player_df["season"] == game["season"]) &
                (player_df["week"] <= game["week"])
            ].tail(20)  # Last 20 player records
            
            if not home_players.empty:
                # QB stats
                qb_stats = home_players[home_players["position"] == "QB"]
                game_features.update({
                    "qb_avg_rating": qb_stats["qb_rating"].mean() if not qb_stats.empty else 85,
                    "qb_avg_yds": qb_stats["passing_yds"].mean() if not qb_stats.empty else 220,
                    "qb_avg_tds": qb_stats["passing_tds"].mean() if not qb_stats.empty else 1.5,
                })
                
                # RB stats
                rb_stats = home_players[home_players["position"] == "RB"]
                game_features.update({
                    "rb_avg_yds": rb_stats["rushing_yds"].mean() if not rb_stats.empty else 100,
                    "rb_avg_tds": rb_stats["rushing_tds"].mean() if not rb_stats.empty else 0.8,
                })
                
                # WR/TE stats
                skill_stats = home_players[home_players["position"].isin(["WR", "TE"])]
                game_features.update({
                    "skill_avg_yds": skill_stats["receiving_yds"].mean() if not skill_stats.empty else 80,
                    "skill_avg_tds": skill_stats["receiving_tds"].mean() if not skill_stats.empty else 0.7,
                    "skill_avg_receptions": skill_stats["receptions"].mean() if not skill_stats.empty else 4.5,
                })
            
            team_features.append(game_features)
            
            # Add away team features (symmetric)
            away_features = game_features.copy()
            away_features["team_id"] = game["away_team"]
            away_features["opponent_id"] = game["home_team"]
            away_features["home_advantage"] = 0
            away_features["game_result"] = game.get("away_score", 0) > game.get("home_score", 0)
            team_features.append(away_features)
        
        features_df = pd.DataFrame(team_features)
        
        # Encode categorical features
        categorical_cols = ["venue"]
        for col in categorical_cols:
            if col in features_df.columns:
                le = LabelEncoder()
                features_df[f"{col}_encoded"] = le.fit_transform(features_df[col].fillna("unknown"))
                self.label_encoders[col] = le
        
        # Fill missing values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(features_df[numeric_cols].mean())
        
        # Create win probability target
        features_df["win_probability"] = features_df["game_result"].astype(float)
        
        logger.info(f"Prepared {len(features_df)} team feature records")
        return features_df
    
    def train_chemistry_model(self, features_df: pd.DataFrame, test_size: float = 0.2):
        """
        Train the team chemistry neural network.
        """
        logger.info("Training team chemistry neural network")
        
        # Prepare features and targets
        feature_cols = [col for col in features_df.columns 
                       if col not in ["team_id", "opponent_id", "game_result", "win_probability", "venue"]]
        
        X = features_df[feature_cols]
        y = features_df[["win_probability"]]
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Add back team_id for dataset creation
        X_scaled["team_id"] = features_df["team_id"]
        y["team_id"] = features_df["team_id"]
        
        # Split data
        train_data, val_data = train_test_split(
            pd.concat([X_scaled, y], axis=1), 
            test_size=test_size, 
            random_state=42,
            stratify=y["win_probability"].round()
        )
        
        # Create datasets
        train_features = train_data[feature_cols + ["team_id"]]
        train_targets = train_data[["team_id", "win_probability"]]
        
        val_features = val_data[feature_cols + ["team_id"]]
        val_targets = val_data[["team_id", "win_probability"]]
        
        train_dataset = TeamChemistryDataset(train_features, train_targets)
        val_dataset = TeamChemistryDataset(val_features, val_targets)
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size)
        
        # Initialize model
        input_dim = len(feature_cols)
        self.model = TeamChemistryNetwork(input_dim=input_dim)
        
        # Loss and optimizer
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        # Training loop
        best_loss = float(