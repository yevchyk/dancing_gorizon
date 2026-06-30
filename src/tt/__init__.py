"""Танцюючий Тарас (ТТ) — new training paradigm.

Regress the forward price CURVE (cumulative log-return, vol-normalized) as a
multi-output target; horizon is the OUTPUT axis, not an input feature. See
DANCING_TARAS.md (authoritative) and CLAUDE.md §13. Binance-only. Does NOT touch
the hc/v2/v3/v4/v5 pipeline — it only REUSES their candle-prep machinery.
"""
