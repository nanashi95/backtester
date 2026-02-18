"""Strategy registry — add new strategies here."""

from strategies.sma_crossover import SmaCrossover
from strategies.donchian_breakout import DonchianBreakout
from strategies.donchian_reversal import DonchianReversal
from strategies.donchian_reversal_100 import DonchianReversal100
from strategies.sma_multi_tf import SmaMultiTF
from strategies.donchian_trend import DonchianTrend

STRATEGIES = [
    DonchianReversal(),
]
