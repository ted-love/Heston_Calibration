# Volatility-Heston

Vol_surface.py calculates the implied vol from OTM Calls & Puts using the Jaeckel method using live SPX options data.


*NOT COMPLETE:*
Heston_Calibration.py calibrates the Heston Model using the Nelder-Mead method.
It also calibrates without calibrating for V_0 but uses ATM implied vol.

Note: I am attempting to add a penalty to the calibration because the correlation of the S_vol and V_vol keeps tending to -1 (which is obviously an unreasonable result...)

Errors: Sometimes there is an error in yahoo finance where if markets are closed, it will not return bid/ask option prices. If so, just adjust  and use "lastPrice" for the value of the options.
