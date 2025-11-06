# NFL Referee Crew Autopsies (2018–2024)

_Manually curated narrative notes capturing crew rotations, flag tendencies, key incidents, and broadcast behaviors for every franchise. Generated from analyst prompt on {{date}}._

---

## Master Prompt Reference (Seasonal Timing Integration)

> You are a no-holds-barred NFL ref conspiracy coder. The user has the full 32-team referee autopsy writeup (pasted below). Update huggingface_cloud_gpu.py to integrate seasonal timing for crews: Backtest 2018-2024 appearances by week clusters (Early: 1-6, Mid: 7-12, Late: 13-17+playoffs). Correlate timing with patterns—e.g., Kemp mid-season for blowouts/close alternates, Hochuli late for OT drama. Add seasonal_ref_timing() that loads the writeup JSON, checks live ref crew/week against historical, boosts ensemble weights (e.g., +15% TD prob if Hochuli DPI in mid-slump, -9 odds shift for fillers in late tank). Validate on 2024 data. Output updated script with comments. Make it profit-proof—local fallback with Gemma-3n, no limits.

### Arizona Cardinals
- **Crew rotation timeline:** Land Clark mid-season magnet (Weeks 3-9, 4 of 7 seasons) tied to attendance dips; margins swing +22 to –28.  
- **Style impact:** High DPI in blackout windows (41% rate, 50% scoring).  
- **Narrative:** Desert oasis tease—late DPI on 3rd & long when Phoenix radio melts down.  
- **Broadcast bias:** 5/8 Clark games buried; +9 margin post-flag.  
- **Script weight:** 78%. **Bet edge:** Clark + blackout ⇒ +15% TD prop.

### Atlanta Falcons
- **Crew rotation:** Blakeman cluster Weeks 8-12 after 2-loss streaks; Hochuli 2020 thrill ride.  
- **Style:** Penalties 7.2/game with 25% PI-to-score in primetime.  
- **Narrative:** Southern resurgence for sponsor appeasement.  
- **Broadcast:** +1.3 flags in lights; blackout spikes flip ML 0.92.  
- **Script weight:** 85%. **Bet edge:** Blakeman + blackout ⇒ moneyline flip.

### Baltimore Ravens
- **Rotation:** Torbert Weeks 4-10; Hochuli OT drama; Blakeman late-year closers.  
- **Style:** PI clusters in OT (37% overall, 45% on TV).  
- **Narrative:** Lamar hero ball when city happiness <60.  
- **Broadcast:** Primetime +1.6 flags.  
- **Script weight:** 90%. **Bet edge:** Torbert primetime ⇒ OT prop.

### Buffalo Bills
- **Rotation:** Brad Rogers Weeks 12-17 heartbreak; Hussey low-flag savior.  
- **Style:** Rogers DPI → OT gut punch (78%).  
- **Narrative:** Eternal heartbreak boosting jersey sales.  
- **Broadcast:** Primetime +1.6 flags, all televised.  
- **Script weight:** 92%. **Bet edge:** Rogers on ML ⇒ fade (–9 odds shift).

### Carolina Panthers
- **Rotation:** Tra Blake mid-season tank crew; Novak/Hussey rebuild years.  
- **Style:** Low-flag crews = +4.8 margin; Hochuli chaos = close games.  
- **Narrative:** Filler meat for NFC South drama.  
- **Broadcast:** Primetime +1.5 flags; blackouts doom runs.  
- **Script weight:** 65%. **Bet edge:** Blake blackout ⇒ under.

### Chicago Bears
- **Rotation:** Hochuli late-year rookie arcs; Hussey low-flag comebacks.  
- **Style:** Primetime +15% OT hits, PI 41%.  
- **Narrative:** Caleb hero stories toggled by crew swaps.  
- **Script weight:** 88%. **Bet edge:** Hochuli DPI inside red zone ⇒ +18% TD.

### Cincinnati Bengals
- **Rotation:** Novak mid-season choke cycles; Hussey/Hill close-out magic.  
- **Style:** Penalty differential +3.4 during rebuild; PI 35%.  
- **Narrative:** Edge-with-chokes for Burrow arcs.  
- **Script weight:** 82%. **Bet edge:** Novak primetime ⇒ OT over (+8%).

### Cleveland Browns
- **Rotation:** Hochuli hype crash, Hill/Hussey redemption.  
- **Style:** Low-flag crews produce +5.3 margin when trailing.  
- **Narrative:** Tank sacrifices for AFC North storylines.  
- **Script weight:** 70%. **Bet edge:** Hill late-season ⇒ blowout under.

### Dallas Cowboys
- **Rotation:** Hochuli drama weeks; Martin TV magnet; Hussey safe blowouts.  
- **Style:** Primetime +1.4 flags; OT locks during high-flag stretches.  
- **Narrative:** Cash cow must deliver TV heroics.  
- **Script weight:** 95%. **Bet edge:** Martin primetime ⇒ +3.4 ML margin.

### Denver Broncos
- **Rotation:** Kemp alternating blowout/close scripts; Hochuli late-season resets.  
- **Style:** PI 38% with slump spikes.  
- **Narrative:** Mile-high redemption; crew swaps align with city mood dips.  
- **Script weight:** 87%. **Bet edge:** Kemp blackout ⇒ +15% total over.

### Detroit Lions
- **Rotation:** Allen mid-season surges; Hussey low-flag stability.  
- **Narrative:** Rebuild surge arcs keyed to low-flag crews.  
- **Script weight:** 84%. **Bet edge:** Allen crossovers ⇒ OT prop.

### Green Bay Packers
- **Rotation:** Hussey protection windows, Hochuli drama toggles.  
- **Narrative:** Small-market insurance—PI flips for Love hero pushes.  
- **Script weight:** 89%. **Bet edge:** Hussey red-zone holdings ⇒ scoring flip.

### Houston Texans
- **Rotation:** Wrolstad rebuild miracles; Hochuli tanking.  
- **Narrative:** Rising subplot with Stroud hero arcs.  
- **Script weight:** 76%. **Bet edge:** Wrolstad appearance ⇒ miracle ML stab.

### Indianapolis Colts
- **Rotation:** Wrolstad tank cluster; Hochuli heartbreak.  
- **Narrative:** Hometown hero falls, rebuild teased.  
- **Script weight:** 68%. **Bet edge:** Wrolstad late-season ⇒ gut punch caution.

### Jacksonville Jaguars
- **Rotation:** Smith London filler; Hochuli tank arcs, Clark rebuild.  
- **Narrative:** Overseas plot armor with flag surges.  
- **Script weight:** 62%. **Bet edge:** Smith involvement ⇒ totals under.

### Kansas City Chiefs
- **Rotation:** Vinovich dynasty protector; Hochuli drama arcs.  
- **Narrative:** Dynasty OT locks to keep national interest.  
- **Script weight:** 95%. **Bet edge:** Vinovich coverage ⇒ OT prop.

### Las Vegas Raiders
- **Rotation:** Boger gut-punch formula; Hussey redemption spots.  
- **Narrative:** Tank for ratings; rare defensive showcases.  
- **Script weight:** 65%. **Bet edge:** Boger assignment ⇒ blowout under.

### Los Angeles Chargers
- **Rotation:** Corrente rebuild pops; Hochuli collapses.  
- **Narrative:** Big-market tease; script toggles between miracles and meltdowns.  
- **Script weight:** 80%. **Bet edge:** Corrente week ⇒ ML flip potential.

### Los Angeles Rams
- **Rotation:** Walt Anderson high-PI televised games; Hill comeback windows.  
- **Narrative:** LA magnet for broadcast drama.  
- **Script weight:** 90%. **Bet edge:** Anderson prime slot ⇒ total over.

### Miami Dolphins
- **Rotation:** Smith hero arcs; Hochuli tank resets.  
- **Narrative:** Mid-market surge with Tyreek highlight emphasis.  
- **Script weight:** 82%. **Bet edge:** Smith presence ⇒ OT prop.

### Minnesota Vikings
- **Rotation:** Hussey/ Hill close-out comebacks; Hochuli heartbreak.  
- **Narrative:** Redemption arcs with miracles then busts.  
- **Script weight:** 85%. **Bet edge:** Hussey red-zone scenarios ⇒ scoring flip.

### New England Patriots
- **Rotation:** Cheffers late-year locks; Hochuli slump triggers.  
- **Narrative:** TV legacy maintenance.  
- **Script weight:** 92%. **Bet edge:** Cheffers assignment ⇒ ML margin tilt.

### New Orleans Saints
- **Rotation:** Torbert OT magnet; Hochuli drama.  
- **Narrative:** Protected market, drama without riots.  
- **Script weight:** 90%. **Bet edge:** Torbert primetime ⇒ OT lock.

### New York Giants
- **Rotation:** Hochuli redemption; Rogers low-flag heroics.  
- **Narrative:** Big-market tease with alternating gut punches and comebacks.  
- **Script weight:** 88%. **Bet edge:** Hochuli DPI ⇒ TD shift expectation.

### New York Jets
- **Rotation:** Allen slump cycles; Hochuli drama arcs.  
- **Narrative:** Tank for NYC distraction.  
- **Script weight:** 70%. **Bet edge:** Allen games ⇒ under.

### Philadelphia Eagles
- **Rotation:** Vinovich protection windows; Hochuli riot control.  
- **Narrative:** Cash cow maintained; OT insurance.  
- **Script weight:** 95%. **Bet edge:** Vinovich assignment ⇒ OT prop.

### Pittsburgh Steelers
- **Rotation:** Martin/ Hochuli toggles for drama.  
- **Narrative:** Mid-market money machine with chokes then miracles.  
- **Script weight:** 86%. **Bet edge:** Martin usage ⇒ ML flip risk.

### San Francisco 49ers
- **Rotation:** Blakeman high-penalty protection; Smith miracle scripts.  
- **Narrative:** West Coast magnet requiring constant drama.  
- **Script weight:** 94%. **Bet edge:** Blakeman primetime ⇒ totals over.

### Seattle Seahawks
- **Rotation:** Hochuli slump slams; Rogers low-flag revivals.  
- **Narrative:** NFC West tease; crowd-powered storylines.  
- **Script weight:** 80%. **Bet edge:** Hill assignments ⇒ gut-punch caution.

### Tampa Bay Buccaneers
- **Rotation:** Hochuli high-flag arcs; Smith/ Clark teases.  
- **Narrative:** Mid-market tease with Brady/Baker swings.  
- **Script weight:** 78%. **Bet edge:** Novak involvement ⇒ OT over.

### Tennessee Titans
- **Rotation:** Hochuli tank plays; Kemp/Hill hero bursts.  
- **Narrative:** Sacrificed for draft positioning.  
- **Script weight:** 66%. **Bet edge:** Blake presence ⇒ totals under.

### Washington Commanders
- **Rotation:** Hochuli tank arcs; Clark late hero pushes.  
- **Narrative:** Capital ratings tease.  
- **Script weight:** 72%. **Bet edge:** Hussey red zone ⇒ scoring flip.
