# Volatility-Heston

Vol_surface.py calculates the implied vol from OTM Calls & Puts using the Jaeckel method on live SPX options data.

Heston_COS_METHOD.py calculates European options for the heston model using cosine expansion. Is a robust method for fast Heston calculation and takes the cosine expansion of the whole integral of the fourier transform. It is 300x faster than QuantLib's Python implementation FFT of the Heston, because we can truncate and use far less terms to converge (Typically only 64-100 terms vs 1,000 terms) 

*INCOMPLETE*:
Levenberg_Marquardt.py is my own implementation of the Levenbrg-Marquardt Algorithm for calibrating the Heston model. It can currently calibrate 1 option reasonably well, but is not able to calibrate multiple options at once. (Because I do not have a robust method to choose a damping factor and how to adjust the damping factor after each iteration). 

Heston_Calibration.py calibrates the Heston Model using the Nelder-Mead method with the Carr-Madan calculation of the Heston. 
It also calibrates without calibrating for V_0 but uses ATM implied vol.

Errors:
- Sometimes there is an error in yahoo finance where if markets are closed, it will not return bid/ask option prices. If so, use "lastPrice" for the value of the options.
- Sometimes yahoo finance will not return the spot price. Then use spot price from other sources
