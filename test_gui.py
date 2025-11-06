try:
    import football_master_gui
    gui = football_master_gui.FootballMasterGUI()
    print('✅ GUI loaded successfully with Quick Bet tab!')
except Exception as e:
    print(f'❌ Error: {e}')
