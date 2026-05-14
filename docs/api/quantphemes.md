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
| Last-price endpoint | `2800` |
| Trading holding PATCH | `2800.HK` |

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
3. Wait a few seconds; paper fills may not be instant.
4. Query orders on the master strategy id.
5. Check whether the platform created or references a child holding strategy.
6. Confirm account/free-plan limitations with Quantphemes if orders still do not appear.

## Free-Plan Notes

The platform plan may limit automation features even when paper trading and read APIs work. Keep a short evidence trail when testing:

- Request path and method.
- Sanitized payload shape.
- Response status and body.
- Follow-up order query result.
- Platform UI screenshot if needed.
