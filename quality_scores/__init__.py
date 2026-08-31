"""
quality_scores — per-company fundamental quality scoring, one module per
methodology ("school"). Each school takes the same underlying
point-in-time fundamentals (via line_items.get_annual_snapshots) and
returns its own score + breakdown, so different frameworks (Piotroski,
QMJ, Magic Formula, Altman Z, ...) can be run, compared, and backtested
independently of each other.

Currently implemented:
    - piotroski.piotroski_f_score
"""
