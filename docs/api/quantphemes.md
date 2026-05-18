# Quantphemes API Integration

The code separates documented endpoints from undocumented stock-data endpoints. This keeps the stable broker wrapper clean and makes fragile dependencies visible.

## Documented Client

`quantphemes_rl.api_client.quantphemes.QuantphemesClient` covers:

| Method | Endpoint |
|---|---|
| `list_portfolios()` | `GET /api/v1/portfolio` |
| `get_portfolio_strategies(portfolio_id)` | `GET /api/v1/portfolio/{portfolioId}/strategy` |
| `get_holdings(strategy_id)` | `GET /api/v1/strategy/{strategyId}/holding` |
| `get_orders(strategy_id)` | `GET /api/v1/strategy/{strategyId}/order` |
| `get_strategy_quantities(strategy_id)` | `GET /api/v1/strategy/{strategyId}/holding/strategy-quantity` |
| `create_holding(strategy_id, payload)` | `POST /api/v1/strategy/{strategyId}/holding` |
| `update_holding(strategy_id, payload)` | `PATCH /api/v1/strategy/{strategyId}/holding` |

Non-2xx responses raise `QuantphemesAPIError` with the response body.

## Undocumented Price Reads

Stock price reads live in `_undocumented.py` because they are not part of the public documented client and may change.

Observed symbol behavior:

| Use | Symbol |
|---|---|
| Last-price endpoint | usually accepts the numeric code, for example `2800` |
| Trading holding create/PATCH | requires the trading symbol, for example `2800.HK` |

Price availability does not guarantee tradeability. Quantphemes accepted price reads for `7299`, but rejected holding creation for both `7299` and `7299.HK` with “Invalid symbols (not in tradable stock list).” The `7299` paper deployments are therefore stopped until Quantphemes whitelists that instrument or confirms a different trading symbol.

## Holding PATCH Shape

The working target-position payload is:

```json
{
  "holdings": [
    {
      "effective_datetime": "2026-05-14T09:30:00+00:00",
      "stocks": [
        {
          "symbol": "2800.HK",
          "quantity": 500
        }
      ]
    }
  ]
}
```

The live bot computes target quantity from cash, price, lot size, and fee buffer, then calls `patch_target`.

## Debugging No Visible Trade

If PATCH succeeds but no trade appears:

1. Confirm the target quantity is different from current quantity.
2. Confirm trading symbol includes `.HK`.
3. Confirm the master strategy already has a holding/copy-worker strategy. New masters need one zero-quantity `POST /holding` bootstrap before the first `PATCH`.
4. Confirm the symbol is in the platform tradable list, not merely available from the price endpoint.
5. Wait a few seconds; paper fills may not be instant.
6. Query orders on the master strategy id.
7. Confirm account/free-plan limitations with Quantphemes if orders still do not appear.

## Current Tradability Notes

| Symbol | Observed Status |
|---|---|
| `2800.HK` | Holding creation and PATCH succeeded |
| `2828.HK` | Holding creation and PATCH succeeded |
| `7226.HK` | Holding creation and PATCH succeeded |
| `7299.HK` | Holding creation rejected as not tradable |

## Free-Plan Notes

The platform plan may limit automation features even when paper trading and read APIs work. Keep a short evidence trail when testing:

- Request path and method.
- Sanitized payload shape.
- Response status and body.
- Follow-up order query result.
- Platform UI screenshot if needed.
