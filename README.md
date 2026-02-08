# Ztockly Scalping Scanner v7

v7 adds:
- **Fib‑anchored partial take profit targets** (TP1 at recent swing high/low, TP2 at fib extension)
- **Liquidity‑weighted scoring** (premarket/afterhours discounted vs RTH)
- **ATR‑normalized score calibration per ticker** (keeps scores comparable across very different vol tickers)
- **Optional higher‑TF bias overlay** (15m/30m)

Still included from v6/v5:
- Session VWAP (ET reset) + cumulative VWAP
- VWAP logic selector for scoring + dual VWAP chart toggle
- Pro mode: liquidity sweeps + order blocks + breaker blocks + FVG + EMA context
- Fib retracement confluence scoring (adds points when price is near key retracement levels)
- In‑app alerts with cooldown

## Run
```bash
pip install -r requirements.txt
export ALPHAVANTAGE_API_KEY="YOUR_KEY"
streamlit run app.py
```

## Notes
- Watchlists > ~10 tickers at 1min can hit Alpha Vantage limits. Keep it tight.


v7.1 fixes:
- Session VWAP now resets at **09:30 ET** by default (RTH VWAP). Optional toggle to include Premarket (starts 04:00).
- ATR normalization slider upgraded; **Auto (per ticker)** baseline added.


v7.2:
- Table + per-ticker diagnostics now show **Current ATR% / Baseline ATR% / Score scale** so you can sanity-check normalization instantly.


v7.3:
- Fixes indentation bug in the ranked table.
- Colors **Score scale** (red < 0.90, green > 1.10).
- Adds tooltip on **Score scale** explaining ATR normalization for that ticker.


v7.3-fixed:
- Rebuilt scanner results block to eliminate indentation mismatches on Streamlit Cloud.


v7.3.1:
- Fix Streamlit Cloud Styler tooltip crash by removing Styler tooltips.
- Adds **Scale note** column explaining the normalization per ticker.


v7.3.2:
- Fix NameError by defining scale-note function before applying it.


v7.3.3:
- Removes pandas Styler entirely (Streamlit Cloud safe). Adds Scale flag + Scale note columns instead.


v7.3.4:
- Fix crash when ATR normalization is Auto (target_atr_pct=None).
