# Live Trading Runbook

This runbook is for operating the live bot safely. It contains commands but no secrets.

## Local Secrets

Create `.env` on the machine that runs the bot:

```bash
QUANTPHEMES_API_KEY=<your key>
QUANTPHEMES_PORTFOLIO_ID=<portfolio id>
QUANTPHEMES_STRATEGY_ID=<strategy id>
```

Never commit `.env`.

## Pre-Live Checklist

- `artifacts/q_state.pkl` exists and came from a reviewed real-data experiment.
- `experiments/production_2800.yaml` points to the intended symbol.
- Dry-run produces decision logs.
- Quantphemes strategy/portfolio are confirmed in the platform UI.
- The bot can read cash, position, and last price.
- Production force-close is scheduled before the market close, currently `15:55` HKT.
- You know how to stop the service.

## Dry-Run

```bash
python -m apps.bot --config experiments/production_2800.yaml --dry-run
```

Simulated day:

```bash
python -m apps.bot \
  --config experiments/production_2800.yaml \
  --dry-run \
  --simulate-now 2026-05-14T09:25:00+08:00
```

Dry-run should emit structured JSON lines with `dry_run: true`.

## Live Command

```bash
python -m apps.bot --config experiments/production_2800.yaml
```

Omitting `--dry-run` allows PATCH calls to Quantphemes.

## Azure VM Deployment

Recommended runtime layout:

```text
/home/azureuser/Algo-Trading-RL/
  .env
  artifacts/q_state.pkl
  artifacts/logs/
```

Install once:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip build-essential
git clone https://github.com/YelamanKarassay/Algo-Trading-RL.git
cd Algo-Trading-RL
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev]"
```

Copy `.env`, `artifacts/q_state.pkl`, and any required local data files separately. Do not commit those files.

## systemd Service

Service file:

```ini
[Unit]
Description=Quantphemes RL Trading Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/Algo-Trading-RL
Environment=PYTHONUNBUFFERED=1
ExecStart=/home/azureuser/Algo-Trading-RL/.venv/bin/python -m apps.bot --config experiments/production_2800.yaml
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable --now quantphemes-bot
```

Check status:

```bash
sudo systemctl status quantphemes-bot --no-pager
```

Stop:

```bash
sudo systemctl stop quantphemes-bot
```

Restart after code or model update:

```bash
sudo systemctl restart quantphemes-bot
```

## Monitoring

System logs:

```bash
sudo journalctl -u quantphemes-bot -f
```

Bot JSON logs:

```bash
cd ~/Algo-Trading-RL
tail -f artifacts/logs/bot_$(date +%F).jsonl
```

Each decision log should include timestamp, state, action, price, target quantity, current quantity, dry-run flag, and status.

## Rollback

To rollback a bad model:

```bash
sudo systemctl stop quantphemes-bot
cp artifacts/q_state_backup.pkl artifacts/q_state.pkl
sudo systemctl start quantphemes-bot
```

To rollback code:

```bash
cd ~/Algo-Trading-RL
git checkout <known-good-commit>
source .venv/bin/activate
pip install -e ".[dev]"
sudo systemctl restart quantphemes-bot
```
