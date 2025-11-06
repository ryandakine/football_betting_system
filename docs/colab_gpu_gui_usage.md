# Colab GPU Fun Station

The Colab GPU Fun Station keeps the interface bright and simple so a young helper can explore confidently while grown-ups handle the advanced pieces.

## Quick Launch

- Open your desktop menu and click **Colab GPU Fun Station** (a launcher is installed automatically).
- Or run it manually from the project folder with `python3 colab_gpu_gui.py`.
- The helper script `./launch_colab_gui.sh` does the same thing if you prefer the terminal.

If Python says Tkinter is missing, install the system package (for example `sudo apt install python3-tk` on Ubuntu/Debian).

## Tabs at a Glance

- **Home** – Friendly temperature, CPU, and memory cards (“computer feels like…”) plus playful tips powered by `system_performance_analyzer.py`. Tap **Refresh Now** whenever you need an instant update.
- **Magic Notebooks** – One-click notebook creators. Press **Make This Notebook**, then upload the saved file from `colab_notebooks/` into Google Colab.
- **Live Lights** – A gentle “light show” that logs what the computer is doing. Green entries are normal; orange or red entries signal a good time to call a grown-up.
- **Grown-Ups** – Safe configuration switches (temperature units, friendly warning pop-ups, and the optional cloud helper address). The tab reminds everyone to ask for help before changing anything.

## Cloud Helper (Optional)

Adults can connect a remote Ollama/Colab bridge:

1. Open the **Grown-Ups** tab and toggle **Use remote helper when available**.
2. Paste the HTTPS URL that points to your remote Ollama instance (for example, an ngrok tunnel that exposes `/api/tags`).
3. Return to **Magic Notebooks** and press **Check Cloud Helper**. The status log shows a kid-friendly success or “ask for help” message.

Behind the scenes, the app calls `CloudGPUWrapper.initialize()` and sends a short completion request so you know the cloud endpoint is healthy.

## Desktop Launcher

A ready-to-use launcher lives at `~/.local/share/applications/colab_gpu_fun_station.desktop`. If you ever need to recreate it, run:

```bash
./launch_colab_gui.sh --install-launcher
```

That command rewrites the desktop entry with the correct absolute path.

## Troubleshooting

- **No Tkinter** – Install the GUI toolkit (`python3-tk`, `brew install python-tk@3.10`, etc.).
- **No temperature reading** – Some machines don’t expose `psutil.sensors_temperatures`; the Home tab will simply show “N/A”.
- **Cloud helper offline** – Confirm the remote endpoint answers `GET /api/tags` and that any tunnel/firewall rules allow connections.

If a young helper spots red text or a warning pop-up, it is their cue to fetch a grown-up—that’s how the Fun Station is designed to be used. 🙂
