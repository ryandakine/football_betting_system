#!/usr/bin/env python3
"""Test AI provider status"""

import sys
sys.path.append('.')

from football_master_gui import UnifiedAIProvider
from api_config import get_api_keys

print('🤖 Testing AI Providers Initialization...')
api_keys = get_api_keys()

provider = UnifiedAIProvider(api_keys)
status = provider.get_provider_status()

print('\n📊 AI Provider Status:')
active_count = 0
for name, info in status.items():
    status_icon = '✅' if info['status'] == 'active' else '❌' if info['status'] == 'no_api_key' else '⚠️'
    print(f'{status_icon} {info["name"]} ({name}): {info["status"]}')
    if info['status'] == 'active':
        active_count += 1

print(f'\n🎯 Total Active Providers: {active_count}/8')
premium_count = len([p for p in status.values() if p['status'] == 'active' and p['name'] not in ['Ollama', 'HuggingFace']])
free_count = len([p for p in status.values() if p['status'] == 'active' and p['name'] in ['Ollama', 'HuggingFace']])

print(f'💰 Premium AI Providers: {premium_count}')
print(f'🆓 Free Fallbacks: {free_count}')

if active_count >= 3:
    print('\n🚀 SYSTEM READY! You have enough AI providers for intelligent predictions.')
else:
    print('\n⚠️ Limited AI providers - system will rely more on fallbacks.')
