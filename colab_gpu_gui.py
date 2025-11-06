#!/usr/bin/env python3
"""Modern GUI wrapper for the Colab GPU efficiency workflow."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import psutil
import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from cloud_gpu_wrapper import CloudGPUWrapper, CloudConfig
from system_performance_analyzer import SystemPerformanceAnalyzer


CONFIG_DIR = Path.home() / ".colab_gpu_gui"
CONFIG_PATH = CONFIG_DIR / "settings.json"
NOTEBOOK_DIR = Path("colab_notebooks")

DEFAULT_SETTINGS: Dict[str, object] = {
    "temp_unit": "F",
    "alerts_enabled": True,
    "cloud_enabled": False,
    "cloud_url": "",
    "fallback_to_local": True,
    "monitor_interval_ms": 4000,
}

NOTEBOOK_TEMPLATES: Dict[str, Dict[str, object]] = {
    "ml": {
        "title": "Colab GPU - Machine Learning Starter",
        "description": "GPU-accelerated training notebook with data loading and evaluation hooks.",
        "cells": [
            "# %% [markdown]\n# 🧠 Colab GPU - Machine Learning Starter\n"
            "# Configure environment and verify GPU\n",
            "!nvidia-smi",
            "# %%\nimport torch\nprint(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else print('CPU only')\n",
            "# %%\n# TODO: add your dataset loading and training code here\n",
        ],
    },
    "data": {
        "title": "Colab GPU - Data Processing",
        "description": "Optimized ETL notebook with pandas/pyspark bootstrap.",
        "cells": [
            "# %% [markdown]\n# 📊 Colab GPU - Data Processing Pipeline\n",
            "!pip install --quiet pandas pyarrow",
            "import pandas as pd\nimport numpy as np\nprint('Ready for large-scale data wrangling!')\n",
        ],
    },
    "image": {
        "title": "Colab GPU - Computer Vision",
        "description": "Starter notebook for image pipelines and diffusers.",
        "cells": [
            "# %% [markdown]\n# 🖼️ Colab GPU - Computer Vision Starter\n",
            "!pip install --quiet opencv-python pillow",
            "import cv2\nprint('OpenCV ready; add your pipeline below')\n",
        ],
    },
    "general": {
        "title": "Colab GPU - General Purpose Accelerator",
        "description": "Blank GPU-ready notebook scaffold.",
        "cells": [
            "# %% [markdown]\n# ⚡ Colab GPU - General Purpose Notebook\n",
            "!nvidia-smi",
            "# %%\nprint('Notebook scaffold loaded. Add cells as needed.')\n",
        ],
    },
}


class ColabGPUApp:
    """Tkinter application shell."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("Colab GPU Fun Station")
        self.root.geometry("1080x720")
        self.settings = self._load_settings()
        self.monitor_running = False
        self.monitor_job: Optional[str] = None

        self.system_analyzer = SystemPerformanceAnalyzer()
        self.cloud_wrapper: Optional[CloudGPUWrapper] = None
        self.cloud_lock = threading.Lock()

        self._configure_styles()
        self._build_layout()
        self._schedule_metric_refresh()
        self._start_monitoring(auto=True)

    def _configure_styles(self) -> None:
        self.root.configure(bg="#1e1f29")
        self.root.option_add("*Font", "Segoe UI 12")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "TFrame",
            background="#1e1f29",
        )
        style.configure(
            "Card.TFrame",
            background="#2a2c3b",
            relief="flat",
        )
        style.configure(
            "Header.TLabel",
            background="#1e1f29",
            foreground="#fefefe",
            font=("Segoe UI", 22, "bold"),
        )
        style.configure(
            "Title.TLabel",
            background="#1e1f29",
            foreground="#f5f6ff",
            font=("Segoe UI", 18, "bold"),
        )
        style.configure(
            "Body.TLabel",
            background="#2a2c3b",
            foreground="#e2e4f0",
            font=("Segoe UI", 13),
        )
        style.configure(
            "Body.TButton",
            padding=10,
            background="#3d4160",
            foreground="#f5f6ff",
            font=("Segoe UI", 14, "bold"),
        )
        style.map(
            "Body.TButton",
            background=[("active", "#5960a8")],
        )

    def _build_layout(self) -> None:
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=12, pady=12)

        self.dashboard_frame = ttk.Frame(notebook, style="TFrame")
        self.colab_frame = ttk.Frame(notebook, style="TFrame")
        self.monitor_frame = ttk.Frame(notebook, style="TFrame")
        self.settings_frame = ttk.Frame(notebook, style="TFrame")

        notebook.add(self.dashboard_frame, text="Home")
        notebook.add(self.colab_frame, text="Magic Notebooks")
        notebook.add(self.monitor_frame, text="Live Lights")
        notebook.add(self.settings_frame, text="Grown-Ups")

        self._build_dashboard_tab()
        self._build_colab_tab()
        self._build_monitor_tab()
        self._build_settings_tab()

    def _build_dashboard_tab(self) -> None:
        title = ttk.Label(self.dashboard_frame, text="Hi there! Let's see how your computer is feeling.", style="Header.TLabel")
        title.pack(anchor="w", pady=(8, 16))

        grid = ttk.Frame(self.dashboard_frame, style="TFrame")
        grid.pack(fill="x")

        self.temp_value = tk.StringVar(value="Temperature: warming up the thermometer…")
        self.cpu_value = tk.StringVar(value="Brain Power: checking…")
        self.memory_value = tk.StringVar(value="Memory Snacks: counting…")
        self.load_value = tk.StringVar(value="Busy Meter: loading…")

        cards = [
            ("🌡️ Computer Temperature", self.temp_value),
            ("🧠 Brain Power (CPU)", self.cpu_value),
            ("🍪 Memory Snacks (RAM)", self.memory_value),
            ("🎛️ Busy Meter", self.load_value),
        ]

        for idx, (label, var) in enumerate(cards):
            frame = ttk.Frame(grid, style="Card.TFrame")
            frame.grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="nsew")
            grid.columnconfigure(idx % 2, weight=1)
            grid.rowconfigure(idx // 2, weight=1)

            ttk.Label(frame, text=label, style="Body.TLabel", font=("Segoe UI", 16, "bold")).pack(
                anchor="w", padx=16, pady=(14, 8)
            )
            ttk.Label(frame, textvariable=var, style="Body.TLabel", font=("Segoe UI", 14)).pack(
                anchor="w", padx=16, pady=(0, 12)
            )

        refresh_frame = ttk.Frame(self.dashboard_frame, style="TFrame")
        refresh_frame.pack(anchor="w", padx=4, pady=(8, 4))
        ttk.Button(
            refresh_frame,
            text="Refresh Now",
            style="Body.TButton",
            command=self._update_metrics,
        ).pack(side="left", padx=(0, 12))
        ttk.Label(
            refresh_frame,
            text="Need help? Ask an adult if anything turns red.",
            style="Body.TLabel",
        ).pack(side="left")

        self.recommendations_box = ScrolledText(self.dashboard_frame, height=8, bg="#1e1f29", fg="#f5f6ff")
        self.recommendations_box.pack(fill="both", expand=True, pady=(20, 0))
        self.recommendations_box.insert("end", "Friendly tips will appear here soon!")
        self.recommendations_box.configure(state="disabled")

    def _build_colab_tab(self) -> None:
        title = ttk.Label(self.colab_frame, text="Make a magic notebook to use in Google Colab!", style="Header.TLabel")
        title.pack(anchor="w", pady=(8, 16))

        description = ttk.Label(
            self.colab_frame,
            text="Pick a notebook style, press a button, and we'll save it for you automatically.",
            style="Body.TLabel",
        )
        description.pack(anchor="w")

        cards_container = ttk.Frame(self.colab_frame, style="TFrame")
        cards_container.pack(fill="both", expand=True, pady=20)

        for idx, (key, meta) in enumerate(NOTEBOOK_TEMPLATES.items()):
            frame = ttk.Frame(cards_container, style="Card.TFrame")
            frame.grid(row=idx // 2, column=idx % 2, padx=10, pady=10, sticky="nsew")
            cards_container.columnconfigure(idx % 2, weight=1)
            cards_container.rowconfigure(idx // 2, weight=1)

            ttk.Label(frame, text=meta["title"], style="Body.TLabel", font=("Segoe UI", 13, "bold")).pack(
                anchor="w", padx=16, pady=(14, 8)
            )
            ttk.Label(frame, text=meta["description"], style="Body.TLabel", wraplength=380).pack(
                anchor="w", padx=16, pady=(0, 16)
            )
            ttk.Button(
                frame,
                text="Make This Notebook",
                style="Body.TButton",
                command=lambda template=key: self._handle_notebook_generation(template),
            ).pack(anchor="w", padx=16, pady=(0, 10))

        actions_frame = ttk.Frame(self.colab_frame, style="Card.TFrame")
        actions_frame.pack(fill="x", pady=10)

        ttk.Label(
            actions_frame,
            text="Quick Actions",
            style="Body.TLabel",
            font=("Segoe UI", 13, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 8))

        action_buttons = ttk.Frame(actions_frame, style="TFrame")
        action_buttons.pack(anchor="w", padx=16, pady=(0, 10))

        ttk.Button(
            action_buttons,
            text="Open Google Colab",
            style="Body.TButton",
            command=lambda: webbrowser.open("https://colab.research.google.com/"),
        ).grid(row=0, column=0, padx=(0, 12), pady=6)

        ttk.Button(
            action_buttons,
            text="Check Cloud Helper",
            style="Body.TButton",
            command=self._async_check_cloud_health,
        ).grid(row=0, column=1, padx=(0, 12), pady=6)

        ttk.Button(
            action_buttons,
            text="Show My Notebooks",
            style="Body.TButton",
            command=self._open_notebook_dir,
        ).grid(row=0, column=2, padx=(0, 12), pady=6)

        self.colab_status = ScrolledText(actions_frame, height=8, bg="#1e1f29", fg="#f5f6ff")
        self.colab_status.pack(fill="both", expand=True, padx=16, pady=(4, 16))
        self._append_colab_status("Ready! Make a notebook or ask a grown-up to set up the cloud helper.")

    def _build_monitor_tab(self) -> None:
        title = ttk.Label(self.monitor_frame, text="Watch the lights dance while we keep you safe.", style="Header.TLabel")
        title.pack(anchor="w", pady=(8, 16))

        ttk.Label(
            self.monitor_frame,
            text="Green lights mean all good. Orange or red? Tap a grown-up!",
            style="Body.TLabel",
        ).pack(anchor="w", padx=12, pady=(0, 12))

        controls = ttk.Frame(self.monitor_frame, style="TFrame")
        controls.pack(anchor="w", pady=(0, 12))

        ttk.Button(
            controls,
            text="Start Light Show",
            style="Body.TButton",
            command=self._start_monitoring,
        ).grid(row=0, column=0, padx=(0, 12))

        ttk.Button(
            controls,
            text="Pause Lights",
            style="Body.TButton",
            command=self._stop_monitoring,
        ).grid(row=0, column=1, padx=(0, 12))

        ttk.Button(
            controls,
            text="Copy Last Story",
            style="Body.TButton",
            command=self._copy_latest_snapshot,
        ).grid(row=0, column=2, padx=(0, 12))

        self.monitor_log = ScrolledText(self.monitor_frame, height=25, bg="#1e1f29", fg="#f5f6ff")
        self.monitor_log.pack(fill="both", expand=True)
        self.monitor_log.insert("end", "Lights are ready. We started the show for you!\n")
        self.monitor_log.configure(state="disabled")

    def _build_settings_tab(self) -> None:
        title = ttk.Label(self.settings_frame, text="Grown-up controls (please ask for help before changing).", style="Header.TLabel")
        title.pack(anchor="w", pady=(8, 16))

        prefs_frame = ttk.Frame(self.settings_frame, style="Card.TFrame")
        prefs_frame.pack(fill="x", pady=(0, 16))

        ttk.Label(prefs_frame, text="Temperature numbers look like", style="Body.TLabel", font=("Segoe UI", 12, "bold")).grid(
            row=0, column=0, sticky="w", padx=16, pady=(16, 6)
        )

        self.temp_unit_var = tk.StringVar(value=str(self.settings.get("temp_unit", "F")))
        ttk.Radiobutton(
            prefs_frame,
            text="Fahrenheit (°F)",
            value="F",
            variable=self.temp_unit_var,
            command=lambda: self._save_settings(False),
        ).grid(row=1, column=0, sticky="w", padx=24, pady=4)

        ttk.Radiobutton(
            prefs_frame,
            text="Celsius (°C)",
            value="C",
            variable=self.temp_unit_var,
            command=lambda: self._save_settings(False),
        ).grid(row=2, column=0, sticky="w", padx=24, pady=4)

        self.alerts_var = tk.BooleanVar(value=bool(self.settings.get("alerts_enabled", True)))
        ttk.Checkbutton(
            prefs_frame,
            text="Turn on friendly warning pop-ups",
            variable=self.alerts_var,
            command=lambda: self._save_settings(False),
        ).grid(row=3, column=0, sticky="w", padx=16, pady=(16, 6))

        ttk.Label(prefs_frame, text="Cloud helper (grown-ups only)", style="Body.TLabel", font=("Segoe UI", 12, "bold")).grid(
            row=4, column=0, sticky="w", padx=16, pady=(24, 6)
        )

        self.cloud_enabled_var = tk.BooleanVar(value=bool(self.settings.get("cloud_enabled", False)))
        ttk.Checkbutton(
            prefs_frame,
            text="Use remote helper when available",
            variable=self.cloud_enabled_var,
            command=lambda: self._save_settings(False),
        ).grid(row=5, column=0, sticky="w", padx=16, pady=4)

        ttk.Label(prefs_frame, text="Cloud helper address (https://…)", style="Body.TLabel").grid(
            row=6, column=0, sticky="w", padx=16, pady=(12, 4)
        )
        self.cloud_url_var = tk.StringVar(value=str(self.settings.get("cloud_url", "")))
        ttk.Entry(prefs_frame, textvariable=self.cloud_url_var, width=48).grid(
            row=7, column=0, sticky="w", padx=16, pady=(0, 12)
        )

        self.fallback_var = tk.BooleanVar(value=bool(self.settings.get("fallback_to_local", True)))
        ttk.Checkbutton(
            prefs_frame,
            text="If cloud helper is busy, use the local one",
            variable=self.fallback_var,
            command=lambda: self._save_settings(False),
        ).grid(row=8, column=0, sticky="w", padx=16, pady=4)

        ttk.Label(
            prefs_frame,
            text="How often to check (seconds)",
            style="Body.TLabel",
        ).grid(row=9, column=0, sticky="w", padx=16, pady=(18, 4))

        self.interval_var = tk.StringVar(value=str(int(self.settings.get("monitor_interval_ms", 4000) / 1000)))
        ttk.Entry(prefs_frame, textvariable=self.interval_var, width=12).grid(
            row=10, column=0, sticky="w", padx=16, pady=(0, 20)
        )

        ttk.Button(
            self.settings_frame,
            text="Save Grown-up Settings",
            style="Body.TButton",
            command=lambda: self._save_settings(True),
        ).pack(anchor="e", padx=12, pady=(0, 16))

    def _load_settings(self) -> Dict[str, object]:
        if CONFIG_PATH.exists():
            try:
                with CONFIG_PATH.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                merged = DEFAULT_SETTINGS.copy()
                merged.update(data)
                return merged
            except (json.JSONDecodeError, OSError):
                messagebox.showwarning("Settings", "We couldn't read the saved settings, so we're using safe defaults.")
        return DEFAULT_SETTINGS.copy()

    def _save_settings(self, show_message: bool) -> None:
        try:
            interval_seconds = float(self.interval_var.get() or "4")
        except ValueError:
            interval_seconds = 4.0
            self.interval_var.set("4")

        settings = {
            "temp_unit": self.temp_unit_var.get(),
            "alerts_enabled": self.alerts_var.get(),
            "cloud_enabled": self.cloud_enabled_var.get(),
            "cloud_url": self.cloud_url_var.get().strip(),
            "fallback_to_local": self.fallback_var.get(),
            "monitor_interval_ms": max(2000, int(interval_seconds * 1000)),
        }
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf-8") as fh:
            json.dump(settings, fh, indent=2)
        self.settings.update(settings)
        if show_message:
            messagebox.showinfo("Settings", "All saved! Thanks for checking with a grown-up.")

    def _schedule_metric_refresh(self) -> None:
        self._update_metrics()
        self.root.after(4500, self._schedule_metric_refresh)

    def _update_metrics(self) -> None:
        temp_c = self._read_cpu_temp_c()
        temp_unit = self.settings.get("temp_unit", "F")
        temp_display = "N/A"

        if temp_c is not None:
            temp_display = f"{temp_c:.1f}°C"
            if temp_unit == "F":
                temp_f = (temp_c * 9 / 5) + 32
                temp_display = f"{temp_f:.1f}°F"

        cpu_percentage = psutil.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()

        try:
            load1, load5, load15 = os.getloadavg()
            load_display = f"{load1:.2f} / {load5:.2f} / {load15:.2f}"
        except OSError:
            load_display = "Unavailable"

        self.temp_value.set(f"Feels like: {temp_display}")
        self.cpu_value.set(f"Brain power used: {cpu_percentage:.1f}%")
        self.memory_value.set(
            f"Snacks eaten: {mem.percent:.1f}% ({self._format_bytes(mem.used)} of {self._format_bytes(mem.total)})"
        )
        self.load_value.set(f"Busy meter (1/5/15 min): {load_display}")

        self._refresh_recommendations()

    def _refresh_recommendations(self) -> None:
        insights = self.system_analyzer.generate_optimization_recommendations()
        self.recommendations_box.configure(state="normal")
        self.recommendations_box.delete("1.0", "end")
        if not insights:
            self.recommendations_box.insert("end", "Everything looks happy and healthy! 🎉\n")
        else:
            for item in insights:
                line = (
                    f"{item['category']} tip ({item['priority']}): {item['recommendation']}\n"
                    f"Try this: {item['action']}\n\n"
                )
                self.recommendations_box.insert("end", line)
        self.recommendations_box.configure(state="disabled")

    def _handle_notebook_generation(self, template_key: str) -> None:
        meta = NOTEBOOK_TEMPLATES.get(template_key)
        if not meta:
            messagebox.showerror("Notebook", "Uh-oh! We couldn't find that notebook style.")
            return

        NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = NOTEBOOK_DIR / f"{template_key}_colab_{timestamp}.ipynb"
        notebook_dict = self._build_notebook_payload(meta)

        with filename.open("w", encoding="utf-8") as fh:
            json.dump(notebook_dict, fh, indent=2)

        self._append_colab_status(f"Notebook created: {filename}")
        messagebox.showinfo(
            "Notebook Ready",
            f"Your notebook is waiting in {filename}.\nUpload it to Google Colab to play!",
        )

    def _build_notebook_payload(self, meta: Dict[str, object]) -> Dict[str, object]:
        cells = []
        for source in meta["cells"]:
            cell_type = "code"
            if source.strip().startswith("# %% [markdown]"):
                cell_type = "markdown"
                cleaned = source.replace("# %% [markdown]", "").lstrip("# ")
                source_lines = [line + "\n" for line in cleaned.splitlines()]
            else:
                source_lines = [line + ("\n" if not line.endswith("\n") else "") for line in source.splitlines()]

            cells.append(
                {
                    "cell_type": cell_type,
                    "metadata": {},
                    "source": source_lines,
                }
            )
            if cell_type == "code":
                cells[-1]["outputs"] = []
                cells[-1]["execution_count"] = None

        return {
            "cells": cells,
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {
                    "file_extension": ".py",
                    "mimetype": "text/x-python",
                    "name": "python",
                },
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }

    def _append_colab_status(self, message: str) -> None:
        self.colab_status.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.colab_status.insert("end", f"[{timestamp}] {message}\n")
        self.colab_status.see("end")
        self.colab_status.configure(state="disabled")

    def _async_check_cloud_health(self) -> None:
        thread = threading.Thread(target=self._check_cloud_health, daemon=True)
        thread.start()

    def _check_cloud_health(self) -> None:
        if not self.settings.get("cloud_enabled"):
            self._append_colab_status("Cloud integration disabled. Enable it in Settings.")
            return

        cloud_url = str(self.settings.get("cloud_url") or "").strip()
        if not cloud_url:
            self._append_colab_status("Cloud URL missing. Add an ngrok/base URL in Settings.")
            return

        self._append_colab_status("Probing cloud GPU endpoint…")

        with self.cloud_lock:
            if not self.cloud_wrapper:
                config = CloudConfig(
                    enabled=True,
                    ngrok_url=cloud_url,
                    fallback_to_local=bool(self.settings.get("fallback_to_local", True)),
                    timeout_seconds=15,
                    retry_attempts=2,
                )
                self.cloud_wrapper = CloudGPUWrapper(config)

        success = False
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            success = loop.run_until_complete(self.cloud_wrapper.initialize())
            loop.close()
            asyncio.set_event_loop(None)
        except Exception as exc:
            self._append_colab_status(f"Initialization error: {exc}")
            success = False

        if not success:
            self._append_colab_status("Cloud GPU health check failed. See logs for details.")
            return

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response = loop.run_until_complete(
                self.cloud_wrapper.generate_completion(
                    prompt="Respond with a short readiness confirmation.",
                    model=self.cloud_wrapper.config.models[0],
                    max_tokens=40,
                )
            )
            loop.close()
            asyncio.set_event_loop(None)
        except Exception as exc:
            self._append_colab_status(f"Health check failed during inference: {exc}")
            return

        if response.success:
            self._append_colab_status(
                f"Cloud GPU healthy via {response.provider.value} provider ({response.response_time:.1f}s)."
            )
        else:
            self._append_colab_status(f"Cloud GPU responded with error: {response.error or 'unknown'}")

    def _start_monitoring(self, auto: bool = False) -> None:
        if self.monitor_running:
            if not auto:
                messagebox.showinfo("Live Lights", "The light show is already running!")
            return
        self.monitor_running = True
        if auto:
            self._append_monitor("Light show started automatically. Sit back and relax!")
        else:
            self._append_monitor("Light show started. We'll keep watch for you.")
        self._schedule_monitor_tick()

    def _stop_monitoring(self) -> None:
        if not self.monitor_running:
            messagebox.showinfo("Live Lights", "The lights are already paused.")
            return
        self.monitor_running = False
        if self.monitor_job:
            self.root.after_cancel(self.monitor_job)
            self.monitor_job = None
        self._append_monitor("Light show paused. Press start when you're ready again.")

    def _schedule_monitor_tick(self) -> None:
        if not self.monitor_running:
            return
        interval_ms = int(self.settings.get("monitor_interval_ms", 4000))
        self.monitor_job = self.root.after(interval_ms, self._monitor_tick)

    def _monitor_tick(self) -> None:
        snapshot = self.system_analyzer.get_system_resources()
        bottlenecks = self.system_analyzer.identify_bottlenecks()
        message = (
            f"CPU {snapshot['cpu_percent']:.1f}% | MEM {snapshot['memory_percent']:.1f}% | "
            f"Disk {snapshot['disk_usage']:.1f}% | Processes {snapshot['process_count']}"
        )
        if bottlenecks:
            message += f" | Heads-up: {', '.join(bottlenecks)}"
        self._append_monitor(message)
        self._schedule_monitor_tick()

    def _append_monitor(self, message: str) -> None:
        self.monitor_log.configure(state="normal")
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.monitor_log.insert("end", f"[{timestamp}] {message}\n")
        self.monitor_log.see("end")
        self.monitor_log.configure(state="disabled")

    def _copy_latest_snapshot(self) -> None:
        content = self.monitor_log.get("end-4l", "end-1c")
        self.root.clipboard_clear()
        self.root.clipboard_append(content.strip())
        messagebox.showinfo("Live Lights", "Got it! The latest story is ready to paste.")

    def _open_notebook_dir(self) -> None:
        NOTEBOOK_DIR.mkdir(parents=True, exist_ok=True)
        path = NOTEBOOK_DIR.resolve()
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            os.system(f'open "{path}"')
        else:
            os.system(f'xdg-open "{path}"')

    def _read_cpu_temp_c(self) -> Optional[float]:
        try:
            temps = psutil.sensors_temperatures()
        except (AttributeError, NotImplementedError):
            return None
        if not temps:
            return None

        for label in ("coretemp", "cpu-thermal", "acpitz"):
            if label in temps:
                readings = temps[label]
                values = [entry.current for entry in readings if entry.current is not None]
                if values:
                    return float(sum(values) / len(values))
        first_group = next(iter(temps.values()), [])
        values = [entry.current for entry in first_group if entry.current is not None]
        return float(sum(values) / len(values)) if values else None

    @staticmethod
    def _format_bytes(value: float) -> str:
        suffixes = ["B", "KB", "MB", "GB", "TB"]
        idx = 0
        while value >= 1024 and idx < len(suffixes) - 1:
            value /= 1024.0
            idx += 1
        return f"{value:.1f}{suffixes[idx]}"

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    app = ColabGPUApp()
    app.run()


if __name__ == "__main__":
    import asyncio

    main()
