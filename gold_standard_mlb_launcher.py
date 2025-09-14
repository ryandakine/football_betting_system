#!/usr/bin/env python3
"""
GOLD-STANDARD MLB LAUNCHER – crash-safe, sentiment-ready
▹ FastAPI server on :8767
▹ Runs core pipeline once at start-up
▹ /opportunities ← POST flattened odds rows
▹ /sentiment     ← POST daily YouTube sentiment JSON
▹ /run-today     ← manual re-run
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from pydantic_settings import BaseSettings

# ────────────── local imports ────────────── #
from gold_standard_main import GoldStandardMLBSystem

# ╔═════════════════════════════════════════╗
# ║ 1. logging                              ║
# ╚═════════════════════════════════════════╝
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-5s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
LOGGER = logging.getLogger("launcher")


# ╔═════════════════════════════════════════╗
# ║ 2. config                               ║
# ╚═════════════════════════════════════════╝
class Settings(BaseSettings):
    bankroll: float = 500.0
    base_unit_size: float = 5.0
    max_units: int = 5
    confidence_threshold: float = 0.55
    max_opportunities: int = 50
    batch_size: int = 20
    max_concurrent_requests: int = 10
    slack_webhook_url: str = ""

    host: str = "0.0.0.0"
    port: int = 8767

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "allow"


CFG = Settings()

# make sure we have the disk folders we rely on
for d in ("data", "results", "logs"):
    Path(d).mkdir(exist_ok=True)

# ╔═════════════════════════════════════════╗
# ║ 3. instantiate core System              ║
# ╚═════════════════════════════════════════╝
system = GoldStandardMLBSystem(
    bankroll=CFG.bankroll,
    base_unit_size=CFG.base_unit_size,
    max_units=CFG.max_units,
    confidence_threshold=CFG.confidence_threshold,
    max_opportunities=CFG.max_opportunities,
)


# helper so the engine can see today’s sentiment if it exists
def _latest_sentiment() -> dict[str, Any] | None:
    fp = Path("data") / f"sentiment_{datetime.now():%Y-%m-%d}.json"
    return json.loads(fp.read_text()) if fp.exists() else None


# ╔═════════════════════════════════════════╗
# ║ 4. FastAPI app                          ║
# ╚═════════════════════════════════════════╝
app = FastAPI(title="GS-MLB", version="1.1.0", docs_url="/docs", redoc_url=None)


class OddsRow(BaseModel):
    game_id: str
    bookmaker: str
    home_team: str
    away_team: str
    commence_time: str
    home_price: int | None = None
    away_price: int | None = None

    class Config:
        extra = "allow"


# ───────────────────────────────────────────
# internal async helpers
async def _process_rows(rows: list[dict[str, Any]]) -> str:
    """Batch the rows, hand to system, then save picks."""
    batches = [
        rows[i : i + CFG.batch_size] for i in range(0, len(rows), CFG.batch_size)
    ]
    tasks = [system.analyze_opportunities_concurrently(batch) for batch in batches]

    analysed: list[dict[str, Any]] = []
    for coro in asyncio.as_completed(tasks):
        analysed.extend(await coro)

    file_ = system.save_recommendations(analysed)
    LOGGER.info("💾 Picks saved ➜ %s  (%d)", file_, len(analysed))
    return file_


async def _run_full_pipeline() -> None:
    """Run the core system end-to-end with sentiment pre-loaded."""
    try:
        system.public_sentiment = _latest_sentiment()
        if system.public_sentiment:
            LOGGER.info(
                "📊  Market sentiment loaded (%s)",
                system.public_sentiment.get("market_sentiment"),
            )

        if hasattr(system, "run_gold_standard_pipeline"):
            raw_rows = await system.run_gold_standard_pipeline() or []
        else:  # fallback for legacy name
            raw_rows = await system.execute_today() or []

        if raw_rows:
            await _process_rows(raw_rows)
        else:
            LOGGER.info("✓ Core pipeline completed (no rows returned)")

    except Exception:  # noqa: BLE001
        LOGGER.error("‼️  Core pipeline crashed:\n%s", traceback.format_exc())


# ───────────────────────────────────────────
# event hooks & endpoints
@app.on_event("startup")
async def _startup() -> None:
    LOGGER.info("🚀  Bootstrap run …")
    await _run_full_pipeline()


@app.get("/ping")
async def ping() -> dict[str, str]:
    return {"status": "alive", "utc": datetime.utcnow().isoformat(timespec="seconds")}


@app.post("/opportunities")
async def opportunities(rows: list[OddsRow]) -> dict[str, Any]:
    LOGGER.info("📥  /opportunities %d rows", len(rows))
    fname = await _process_rows([r.model_dump() for r in rows])
    return {"status": "saved", "file": fname}


# ────────── NEW: sentiment ingest ────────── #
@app.post("/sentiment")
async def sentiment(data: dict[str, Any]) -> dict[str, str]:
    """Receive the YouTube public-sentiment JSON from n8n."""
    fp = Path("data") / f"sentiment_{datetime.now():%Y-%m-%d}.json"
    fp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("🤝  Sentiment received ➜ %s", fp)
    return {"status": "ok"}


@app.post("/run-today")
async def run_today() -> dict[str, str]:
    LOGGER.info("🔔  Manual /run-today triggered")
    await _run_full_pipeline()
    return {"status": "ok"}


# ╔═════════════════════════════════════════╗
# ║ 5. main                                 ║
# ╚═════════════════════════════════════════╝
if __name__ == "__main__":
    uvicorn.run(
        "gold_standard_mlb_launcher:app", host=CFG.host, port=CFG.port, reload=False
    )
