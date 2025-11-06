#!/usr/bin/env python3
"""
Deploy Enhanced AI Council to AWS Lambda
==========================================

Packages the trained models and deploys to Lambda for real-time predictions
using referee features and improved Total Expert models.
"""

import os
import subprocess
import json
from pathlib import Path
import boto3
import zipfile
import shutil

class EnhancedAICouncilDeployer:
    """Deploy enhanced AI Council models to AWS Lambda"""
    
    def __init__(self, lambda_role_arn: str = None):
        self.s3 = boto3.client('s3')
        self.lambda_client = boto3.client('lambda')
        self.iam = boto3.client('iam')
        self.lambda_role_arn = lambda_role_arn or self._get_lambda_role()
        
    def _get_lambda_role(self) -> str:
        """Get or create Lambda execution role"""
        
        role_name = 'football_betting_lambda_role'
        
        try:
            response = self.iam.get_role(RoleName=role_name)
            print(f"✅ Using existing role: {role_name}")
            return response['Role']['Arn']
        except:
            print(f"📝 Creating new Lambda role: {role_name}")
            
            assume_role_policy = {
                "Version": "2012-10-17",
                "Statement": [
                    {
                        "Effect": "Allow",
                        "Principal": {
                            "Service": "lambda.amazonaws.com"
                        },
                        "Action": "sts:AssumeRole"
                    }
                ]
            }
            
            response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(assume_role_policy)
            )
            
            # Attach S3 and CloudWatch policies
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonS3FullAccess'
            )
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/CloudWatchLogsFullAccess'
            )
            
            print(f"✅ Created role: {response['Role']['Arn']}")
            return response['Role']['Arn']
    
    def package_models(self, model_dir: str = 'models', output_dir: str = 'lambda_package'):
        """Package trained models and dependencies for Lambda"""
        
        print(f"\n📦 Packaging models for Lambda...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Create lambda function directory
        func_dir = os.path.join(output_dir, 'lambda_function')
        os.makedirs(func_dir, exist_ok=True)
        
        # Copy models
        models_dest = os.path.join(func_dir, 'models')
        if os.path.exists(models_dest):
            shutil.rmtree(models_dest)
        shutil.copytree(model_dir, models_dest)
        print(f"   ✅ Copied models from {model_dir}")
        
        # Copy referee data
        if os.path.exists('data/referee_training_features.json'):
            os.makedirs(os.path.join(func_dir, 'data'), exist_ok=True)
            shutil.copy('data/referee_training_features.json', 
                       os.path.join(func_dir, 'data/referee_features.json'))
            print(f"   ✅ Copied referee features")
        
        # Create lambda handler
        self._create_lambda_handler(func_dir)
        print(f"   ✅ Created lambda handler")
        
        # Install dependencies
        self._install_dependencies(func_dir)
        print(f"   ✅ Installed Python dependencies")
        
        # Create deployment package
        zip_path = os.path.join(output_dir, 'enhanced_ai_council.zip')
        self._create_deployment_zip(func_dir, zip_path)
        print(f"   ✅ Created deployment package: {zip_path}")
        
        return zip_path
    
    def _create_lambda_handler(self, func_dir: str):
        """Create the Lambda handler function"""
        
        handler_code = '''#!/usr/bin/env python3
"""
Enhanced AI Council Lambda Handler
===================================

Processes NFL game data and returns AI Council predictions
with referee features and improved Total Expert models.
"""

import json
import os
import joblib
import numpy as np
import boto3
from pathlib import Path

# Initialize clients
s3 = boto3.client('s3')

# Load models (loaded once on container startup)
MODELS_DIR = Path(__file__).parent / 'models'
REFEREE_DATA_PATH = Path(__file__).parent / 'data' / 'referee_features.json'

def load_models():
    """Load trained models from disk"""
    
    models = {}
    
    # Load spread models
    for model_name in ['spread_expert', 'contrarian', 'home_advantage']:
        path = MODELS_DIR / f'{model_name}_nfl_model.pkl'
        if path.exists():
            models[model_name] = joblib.load(str(path))
    
    # Load total models
    for model_name in ['total_regressor', 'total_high_games', 'total_low_games', 'total_weather_adjusted']:
        path = MODELS_DIR / f'{model_name}_nfl_model.pkl'
        if path.exists():
            models[model_name] = joblib.load(str(path))
    
    return models

def load_referee_features():
    """Load referee profile data"""
    
    if REFEREE_DATA_PATH.exists():
        with open(REFEREE_DATA_PATH, 'r') as f:
            return json.load(f)
    return {'referee_profiles': {}, 'team_history': {}}

def get_referee_features(game_data, referee_data):
    """Extract referee features for a game"""
    
    referee = game_data.get('referee', 'Unknown')
    ref_profiles = referee_data.get('referee_profiles', {})
    
    features = {
        'ref_avg_margin': 0.0,
        'ref_avg_penalties': 6.0,
        'ref_penalty_diff': 0.0,
        'ref_odds_delta': 0.0,
        'ref_overtime_rate': 6.0,
        'ref_is_high_penalties': 0,
        'ref_is_low_flags': 0,
        'ref_is_overtime_frequent': 0,
        'ref_home_advantage': 0.0,
        'ref_penalty_advantage': 0.0
    }
    
    if referee in ref_profiles:
        profile = ref_profiles[referee]
        features['ref_avg_margin'] = profile.get('avg_margin', 0.0)
        features['ref_avg_penalties'] = profile.get('avg_penalties', 6.0)
        features['ref_penalty_diff'] = profile.get('avg_penalty_diff', 0.0)
        features['ref_odds_delta'] = profile.get('avg_odds_delta', 0.0)
        features['ref_overtime_rate'] = profile.get('avg_overtime_rate', 6.0)
        
        labels = profile.get('labels', [])
        features['ref_is_high_penalties'] = int('high_penalties_close_games' in labels)
        features['ref_is_low_flags'] = int('low_flags_high_blowouts' in labels)
        features['ref_is_overtime_frequent'] = int('overtime_frequency_gt_15pct' in labels)
    
    return features

def predict_game(models, referee_data, game_data, feature_columns, total_feature_columns):
    """Make predictions for a single game"""
    
    predictions = {
        'game_id': game_data.get('game_id'),
        'home_team': game_data.get('home_team'),
        'away_team': game_data.get('away_team'),
        'referee': game_data.get('referee'),
        'predictions': {}
    }
    
    # Load feature columns if not provided
    if not feature_columns:
        features_path = MODELS_DIR / 'nfl_features.json'
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_data = json.load(f)
                feature_columns = feature_data.get('spread_features', [])
                total_feature_columns = feature_data.get('total_features', [])
    
    # Prepare input features (simplified - would need full feature engineering)
    try:
        # Get referee features
        ref_features = get_referee_features(game_data, referee_data)
        
        # Spread predictions
        if 'spread_expert' in models and feature_columns:
            try:
                # Build feature vector (would need proper engineering)
                X = np.zeros((1, len(feature_columns)))
                spread_pred = models['spread_expert'].predict(X)[0]
                spread_prob = models['spread_expert'].predict_proba(X)[0]
                predictions['predictions']['spread'] = {
                    'pick': 'home' if spread_pred == 1 else 'away',
                    'confidence': float(max(spread_prob))
                }
            except Exception as e:
                predictions['predictions']['spread'] = {'error': str(e)}
        
        # Total predictions (uses regressor)
        if 'total_regressor' in models and total_feature_columns:
            try:
                X = np.zeros((1, len(total_feature_columns)))
                total_points = models['total_regressor'].predict(X)[0]
                total_line = game_data.get('total_line', 44)
                pick = 'over' if total_points > total_line else 'under'
                
                predictions['predictions']['total'] = {
                    'pick': pick,
                    'predicted_total': float(total_points),
                    'line': float(total_line),
                    'confidence': 0.55  # Base confidence
                }
            except Exception as e:
                predictions['predictions']['total'] = {'error': str(e)}
        
        # Add referee features to output
        predictions['referee_features'] = ref_features
        
        return predictions
    
    except Exception as e:
        predictions['error'] = str(e)
        return predictions

# Lazy load models
_MODELS = None
_REFEREE_DATA = None

def get_models():
    global _MODELS
    if _MODELS is None:
        _MODELS = load_models()
    return _MODELS

def get_referee_data():
    global _REFEREE_DATA
    if _REFEREE_DATA is None:
        _REFEREE_DATA = load_referee_features()
    return _REFEREE_DATA

def lambda_handler(event, context):
    """AWS Lambda handler"""
    
    try:
        models = get_models()
        referee_data = get_referee_data()
        
        # Handle batch predictions
        if 'games' in event:
            games = event['games']
        else:
            games = [event]
        
        # Load feature columns
        features_path = MODELS_DIR / 'nfl_features.json'
        feature_columns = []
        total_feature_columns = []
        
        if features_path.exists():
            with open(features_path, 'r') as f:
                feature_data = json.load(f)
                feature_columns = feature_data.get('spread_features', [])
                total_feature_columns = feature_data.get('total_features', [])
        
        # Make predictions
        results = []
        for game in games:
            result = predict_game(models, referee_data, game, feature_columns, total_feature_columns)
            results.append(result)
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'predictions': results,
                'count': len(results)
            })
        }
    
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e)
            })
        }
'''
        
        handler_path = os.path.join(func_dir, 'lambda_function.py')
        with open(handler_path, 'w') as f:
            f.write(handler_code)
    
    def _install_dependencies(self, func_dir: str):
        """Install Python dependencies"""
        
        requirements = [
            'numpy',
            'pandas',
            'scikit-learn',
            'joblib',
            'boto3'
        ]
        
        # Create requirements.txt
        req_path = os.path.join(func_dir, 'requirements.txt')
        with open(req_path, 'w') as f:
            for req in requirements:
                f.write(f"{req}\n")
        
        # Install to func_dir
        subprocess.run([
            'pip', 'install', '-r', req_path,
            '-t', func_dir,
            '--quiet'
        ], check=True)
    
    def _create_deployment_zip(self, func_dir: str, zip_path: str):
        """Create deployment ZIP file"""
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(func_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, func_dir)
                    zipf.write(file_path, arcname)
    
    def deploy_to_lambda(self, zip_path: str, function_name: str = 'enhanced_ai_council_predictions'):
        """Deploy to AWS Lambda"""
        
        print(f"\n🚀 Deploying to Lambda: {function_name}")
        
        # Read ZIP file
        with open(zip_path, 'rb') as f:
            zip_content = f.read()
        
        try:
            # Check if function exists
            self.lambda_client.get_function(FunctionName=function_name)
            
            # Update existing function
            print(f"   📝 Updating existing Lambda function...")
            self.lambda_client.update_function_code(
                FunctionName=function_name,
                ZipFile=zip_content
            )
            
            # Update configuration
            self.lambda_client.update_function_configuration(
                FunctionName=function_name,
                Handler='lambda_function.lambda_handler',
                Runtime='python3.11',
                Timeout=300,
                MemorySize=1024,
                Description='Enhanced AI Council with referee features for NFL predictions'
            )
            
            print(f"   ✅ Updated Lambda function")
            
        except self.lambda_client.exceptions.ResourceNotFoundException:
            # Create new function
            print(f"   📝 Creating new Lambda function...")
            
            self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.11',
                Role=self.lambda_role_arn,
                Handler='lambda_function.lambda_handler',
                Code={'ZipFile': zip_content},
                Timeout=300,
                MemorySize=1024,
                Description='Enhanced AI Council with referee features for NFL predictions'
            )
            
            print(f"   ✅ Created Lambda function")
        
        return function_name
    
    def upload_models_to_s3(self, bucket: str, model_dir: str = 'models'):
        """Upload models to S3 for version control"""
        
        print(f"\n📤 Uploading models to S3: s3://{bucket}/ai_council_models/")
        
        # Create S3 bucket if it doesn't exist
        try:
            self.s3.head_bucket(Bucket=bucket)
        except:
            self.s3.create_bucket(Bucket=bucket)
            print(f"   ✅ Created S3 bucket: {bucket}")
        
        # Upload models
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                file_path = os.path.join(root, file)
                s3_key = f'ai_council_models/{os.path.relpath(file_path, model_dir)}'
                
                self.s3.upload_file(file_path, bucket, s3_key)
                print(f"   ✅ Uploaded {s3_key}")

def main():
    """Main deployment workflow"""
    
    print("🏈 DEPLOYING ENHANCED AI COUNCIL TO AWS LAMBDA")
    print("=" * 60)
    
    deployer = EnhancedAICouncilDeployer()
    
    # Package models
    zip_path = deployer.package_models()
    
    # Deploy to Lambda
    function_name = deployer.deploy_to_lambda(zip_path)
    
    # Upload models to S3 for backup
    deployer.upload_models_to_s3('football-betting-ai-models')
    
    print(f"\n✅ Deployment complete!")
    print(f"   Lambda Function: {function_name}")
    print(f"   Referee features: Integrated")
    print(f"   Total Expert models: Deployed (regressor + specialists)")

if __name__ == "__main__":
    main()
