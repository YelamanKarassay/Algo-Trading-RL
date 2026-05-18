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

The multi-bot paper deployment also expects one strategy/portfolio pair per bot:

```bash
STRATEGY_RL_CROSS_ID=<copy-master strategy id>
PORTFOLIO_RL_CROSS_ID=<portfolio id>
STRATEGY_RL_VOL_ID=<copy-master strategy id>
PORTFOLIO_RL_VOL_ID=<portfolio id>
STRATEGY_RL_FACOV_ID=<copy-master strategy id>
PORTFOLIO_RL_FACOV_ID=<portfolio id>
STRATEGY_RL_BROAD_A_ID=<copy-master strategy id>
PORTFOLIO_RL_BROAD_A_ID=<portfolio id>
STRATEGY_RL_BROAD_B_ID=<copy-master strategy id>
PORTFOLIO_RL_BROAD_B_ID=<portfolio id>
```

## Pre-Live Checklist

- The relevant `artifacts/deploy/<bot>_q_state.pkl` exists and came from a reviewed real-data experiment.
- The relevant `experiments/production_rl_*.yaml` points to the intended symbol and Q-table.
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

For the active paper deployment, `systemd` runs one command per strategy. Example:

```bash
python -m apps.bot \
  --config experiments/production_rl_broad_a.yaml \
  --strategy-id "$STRATEGY_RL_BROAD_A_ID" \
  --portfolio-id "$PORTFOLIO_RL_BROAD_A_ID"
```

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

## systemd Services

The Azure deployment runs five active services:

| Service | Bot | Config | Log directory |
|---|---|---|---|
| `quantphemes-rl-cross` | `RL_CROSS` | `experiments/production_rl_cross.yaml` | `artifacts/logs/rl_cross/` |
| `quantphemes-rl-vol` | `RL_VOL` | `experiments/production_rl_vol.yaml` | `artifacts/logs/rl_vol/` |
| `quantphemes-rl-facov` | `RL_FACOV` | `experiments/production_rl_facov.yaml` | `artifacts/logs/rl_facov/` |
| `quantphemes-rl-broad-a` | `RL_BROAD_A` | `experiments/production_rl_broad_a.yaml` | `artifacts/logs/rl_broad_a/` |
| `quantphemes-rl-broad-b` | `RL_BROAD_B` | `experiments/production_rl_broad_b.yaml` | `artifacts/logs/rl_broad_b/` |

The retired `quantphemes-rl-prime`, `quantphemes-rl-bias`, and `quantphemes-rl-fullcov` services should stay stopped because `7299.HK` is not currently tradable through Quantphemes.

Template service file:

```ini
[Unit]
Description=Quantphemes RL Paper Bot
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=azureuser
WorkingDirectory=/home/azureuser/Algo-Trading-RL
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=/home/azureuser/Algo-Trading-RL/.env
ExecStart=/home/azureuser/Algo-Trading-RL/.venv/bin/python -m apps.bot --config experiments/production_rl_broad_a.yaml --strategy-id ${STRATEGY_RL_BROAD_A_ID} --portfolio-id ${PORTFOLIO_RL_BROAD_A_ID}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Check all active services:

```bash
sudo systemctl status \
  quantphemes-rl-cross \
  quantphemes-rl-vol \
  quantphemes-rl-facov \
  quantphemes-rl-broad-a \
  quantphemes-rl-broad-b \
  --no-pager
```

Stop the full active set:

```bash
sudo systemctl stop \
  quantphemes-rl-cross \
  quantphemes-rl-vol \
  quantphemes-rl-facov \
  quantphemes-rl-broad-a \
  quantphemes-rl-broad-b
```

Restart after code or model update:

```bash
sudo systemctl restart \
  quantphemes-rl-cross \
  quantphemes-rl-vol \
  quantphemes-rl-facov \
  quantphemes-rl-broad-a \
  quantphemes-rl-broad-b
```

## Monitoring

System logs:

```bash
sudo journalctl -u quantphemes-rl-broad-a -f
```

Bot JSON logs:

```bash
cd ~/Algo-Trading-RL
tail -f artifacts/logs/rl_broad_a/bot_$(TZ=Asia/Hong_Kong date +%F).jsonl
```

Each decision log should include timestamp, state, action, price, target quantity, current quantity, dry-run flag, and status.

## Rollback

To rollback a bad model:

```bash
sudo systemctl stop quantphemes-rl-broad-a
cp artifacts/deploy/rl_broad_a_q_state_backup.pkl artifacts/deploy/rl_broad_a_q_state.pkl
sudo systemctl start quantphemes-rl-broad-a
```

To rollback code:

```bash
cd ~/Algo-Trading-RL
git checkout <known-good-commit>
source .venv/bin/activate
pip install -e ".[dev]"
sudo systemctl restart quantphemes-bot
```

For multi-bot code rollback, restart the five active `quantphemes-rl-*` services instead of the legacy `quantphemes-bot` service.
