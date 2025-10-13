Going to move into testing multi regieme strats like the one that surpassed all the other SMA ones.  Going to try different lengths and exit rules next.

Two regimes, two rule sets:

Regime A, SMA20 at or above SMA200
Use standard 200 line logic only. The 20 line is ignored.

Entry, need 3 consecutive closes above SMA200.
You enter at the next session open after the third close.

Exit, need 2 consecutive closes at or below SMA200.
You exit at the next session open after the second qualifying close.

Regime B, SMA20 below SMA200
Use an early reentry via the 20 line, plus a momentum gate, and a quick exit.

Entry, need 3 consecutive days where, first, close is above SMA20, second, SMA20 is rising compared to the prior day, the slope gate. Both must be true for 3 days in a row.
You enter at the next session open after the third qualifying day.

Exit, need 1 close at or below SMA20, configurable as exit_days_short, default 1.
You exit at the next session open after that close.