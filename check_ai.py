python -c "
import game_selection
print('Available in game_selection:')
for attr in dir(game_selection):
    if not attr.startswith('_'):
        print(f'  - {attr}')
"
