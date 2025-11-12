# Football Betting System - Backup Guide

## 🎯 What You Need to Protect

**Your current data:**
- ✅ 11 trained models (`models/ncaa/*.pkl`)
- ✅ 42MB historical data (`data/`)
- ✅ 123 game data files
- ✅ 21,522 games collected (2015-2024)
- ✅ Backtest results (60.7% validation)

**If HD fails without backup = YOU LOSE EVERYTHING** ❌

---

## 💾 Best Backup Options (Ranked)

### **Option 1: Cloud Storage (BEST for most people)**

**Services:**
- Google Drive (15GB free)
- Dropbox (2GB free, $12/month for 2TB)
- OneDrive (5GB free)
- iCloud (5GB free)

**Pros:**
- ✅ Automatic sync
- ✅ Access anywhere
- ✅ Version history
- ✅ Survives HD failure
- ✅ Easy to setup

**Cons:**
- ⚠️ Needs internet
- ⚠️ Slight privacy concern (encrypted though)

**Setup:**
```bash
# If you have Google Drive installed:
ln -s ~/GoogleDrive/football_betting_backup ~/Backups/football_betting

# Or Dropbox:
ln -s ~/Dropbox/football_betting_backup ~/Backups/football_betting

# Then run backup:
./backup_system.sh
```

**Your data (42MB) fits easily in free tier!**

---

### **Option 2: External Hard Drive (GOOD for privacy)**

**Pros:**
- ✅ Complete control
- ✅ Fast backups
- ✅ No recurring cost
- ✅ No privacy concerns

**Cons:**
- ⚠️ Drive can also fail
- ⚠️ Need to remember to backup
- ⚠️ Only accessible when plugged in

**Setup:**
```bash
# Edit backup_system.sh
# Change: BACKUP_DIR="$HOME/Backups/football_betting"
# To: BACKUP_DIR="/Volumes/ExternalDrive/football_betting"

# Or for Linux:
# To: BACKUP_DIR="/media/username/ExternalDrive/football_betting"

# Run backup weekly:
./backup_system.sh
```

**Cost:** $50-100 for 1-2TB drive

---

### **Option 3: Remote Server/VPS (BEST for automation)**

**Services:**
- DigitalOcean ($6/month)
- Linode ($5/month)
- AWS Lightsail ($5/month)

**Pros:**
- ✅ Always accessible
- ✅ Can run monitoring 24/7
- ✅ Professional setup
- ✅ Automatic backups

**Cons:**
- ⚠️ Costs money
- ⚠️ Requires setup knowledge
- ⚠️ Overkill for just backups

**Setup:**
```bash
# One-time: Copy to server
scp -r models/ user@yourserver.com:/backups/
scp -r data/ user@yourserver.com:/backups/

# Automatic: Add to cron
0 2 * * * rsync -avz -e ssh models/ user@yourserver.com:/backups/models/
```

**Good if:** Running system 24/7 or want enterprise setup

---

### **Option 4: Git LFS Alternative (Backblaze B2)**

**Service:** Backblaze B2 ($5/TB/month)

**Pros:**
- ✅ Cheap ($5/TB)
- ✅ Version control
- ✅ Integrates with git
- ✅ Unlimited storage

**Cons:**
- ⚠️ Requires setup
- ⚠️ Monthly cost

**Setup:**
```bash
# Install B2 CLI
pip install b2

# Configure
b2 authorize-account

# Backup
b2 sync models/ b2://your-bucket/models/
b2 sync data/ b2://your-bucket/data/
```

**Good if:** You want git-like versioning for data

---

## 🚀 Recommended Setup (3-2-1 Rule)

**3 copies, 2 different media types, 1 offsite**

### **For Your System:**

1. **Primary** - Local HD (where you work)
2. **Secondary** - Google Drive/Dropbox (automatic cloud)
3. **Tertiary** - External drive (weekly manual)

**Why this works:**
- HD fails → Restore from cloud (minutes)
- Cloud account issues → Restore from external (hours)
- Both fail → You have 3rd copy somewhere

---

## ⚡ Quick Start (5 Minutes)

### **Step 1: Run First Backup**

```bash
# This creates backup in ~/Backups/football_betting/
./backup_system.sh
```

**Output:**
```
🔄 Football Betting System Backup
==================================

📊 Backing up historical data...
   ✅ Data backed up (42M)
🤖 Backing up trained models...
   ✅ Models backed up (11 .pkl files)
📈 Backing up backtest results...
   ✅ Backtest results backed up

✅ BACKUP COMPLETE
==================================

📊 Backup Summary:
   Total size: 43M
   Files: 145
```

### **Step 2: Setup Cloud Sync (Optional but Recommended)**

**If you have Google Drive:**
```bash
# One-time setup
mkdir -p ~/GoogleDrive/football_betting_backup

# Symlink for automatic sync
ln -s ~/Backups/football_betting ~/GoogleDrive/football_betting_backup

# Future backups auto-sync!
```

**If you have Dropbox:**
```bash
mkdir -p ~/Dropbox/football_betting_backup
ln -s ~/Backups/football_betting ~/Dropbox/football_betting_backup
```

### **Step 3: Automate (Optional)**

**Run backup daily at 2am:**
```bash
crontab -e

# Add this line:
0 2 * * * cd /home/user/football_betting_system && ./backup_system.sh
```

---

## 🆘 Disaster Recovery

### **If Your HD Fails:**

**You have cloud backup:**
```bash
# 1. Setup fresh system
git clone https://github.com/youruser/football_betting_system.git

# 2. Restore from cloud
cp -r ~/GoogleDrive/football_betting_backup/latest/* .

# 3. Verify
ls -lh models/ncaa/*.pkl  # Should show 11 models
python validate_system.py  # Should pass

# 4. Resume betting!
```

**You have external drive backup:**
```bash
# 1. Setup fresh system
git clone https://github.com/youruser/football_betting_system.git

# 2. Restore from external drive
cp -r /Volumes/ExternalDrive/football_betting/backup_YYYYMMDD/* .

# 3. Verify and resume
```

**You have NO backup:**
```bash
# 😱 You lose:
- 11 trained models (5 min to retrain)
- 21,522 historical games (hours to re-scrape)
- Backtest validation (hours to regenerate)
- CLV tracking history (GONE FOREVER)
- All custom tweaks (GONE FOREVER)

# Can recover:
- Code (git clone)
- Models (retrain)
- Historical data (re-scrape)

# But you lose weeks of work! DON'T LET THIS HAPPEN
```

---

## 📊 What to Backup

### **CRITICAL (Must backup):**

1. **Historical game data** (`data/`)
   - 21,522 games collected
   - Hard to recreate
   - **Size:** 42MB

2. **Trained models** (`models/ncaa/`)
   - 11 trained models
   - Takes 5 min to retrain (but annoying)
   - **Size:** ~50MB

### **IMPORTANT (Should backup):**

3. **Backtest results**
   - 60.7% win rate validation
   - Can regenerate but time-consuming
   - **Size:** ~5MB

4. **CLV/Line shopping history**
   - Tracks your betting performance
   - Shows if you're sharp
   - **Size:** <1MB

### **NICE (Optional):**

5. **Configuration files**
   - `*.json`, `*.py` config
   - Already in git
   - **Size:** <1MB

### **DON'T NEED to backup:**

- Code files (in git)
- System state (regenerates)
- Cache files (`__pycache__/`)
- Temporary files

---

## 💡 Pro Tips

### **Verify Backups Work:**

```bash
# Test restore
cp -r ~/Backups/football_betting/backup_latest/models /tmp/test_restore/
ls -lh /tmp/test_restore/models/ncaa/*.pkl

# Should see 11 .pkl files
```

### **Check Backup Size:**

```bash
# How much space do you need?
du -sh ~/Backups/football_betting/

# Typical: 50-100MB (fits anywhere!)
```

### **Backup Before Big Changes:**

```bash
# Before retraining models:
./backup_system.sh

# Before major code changes:
git commit -am "Before major changes"

# Before scraping new data:
./backup_system.sh
```

---

## 🎯 My Recommendation for You

Based on your 42MB of data:

### **SETUP (Do This Now):**

1. **Run first backup:**
   ```bash
   ./backup_system.sh
   ```

2. **Setup cloud sync** (if you have Google Drive/Dropbox):
   ```bash
   ln -s ~/Backups/football_betting ~/GoogleDrive/football_betting_backup
   ```

3. **Verify it worked:**
   ```bash
   ls -lh ~/Backups/football_betting/
   ls -lh ~/GoogleDrive/football_betting_backup/  # If cloud
   ```

### **ONGOING:**

- **Automatic:** Cloud syncs automatically (if setup)
- **Weekly:** Run `./backup_system.sh` before betting session
- **Monthly:** Copy to external drive (extra safety)

### **COST:**

- **Free** if using Google Drive/Dropbox free tier (42MB << 2GB)
- **$0** ongoing (unless you want paid cloud)

---

## 📋 Backup Checklist

Before every betting session:

- [ ] Last backup < 7 days ago?
- [ ] Cloud sync working? (check timestamp)
- [ ] External drive backup this month?
- [ ] Verified backup size looks right?

If any "No" → Run `./backup_system.sh` now!

---

## Bottom Line

**Your 60.7% system + 21,522 games = VALUABLE DATA**

**Don't lose it to HD failure!**

**Setup takes 5 minutes:**
```bash
./backup_system.sh
ln -s ~/Backups/football_betting ~/GoogleDrive/football_betting_backup
```

**Then it's automatic.** 💾✅

---

*Backup script included: `backup_system.sh`*
*Questions? Check the script comments or run with `-h`*
