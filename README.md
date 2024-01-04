# Heston_Calibation

Heston_Calibration.py calibrates the Heston Model using the Levenberg-Marquardt (LM) algoithm with COS-expansion calculation for the Heston model. Running the file will use pre-loaded historial data. The prices are transformed into implied volatilities and the heston is calibrated against the market implied vols. Dividend rates calculated from creating an implied forward-rate curve, and historical data is use for initial guesses.

tools/Heston_COS_METHOD.py is a vectorised method of calculating European options for the heston model using cosine expansion. It is a robust method for fast Heston calculation and takes the cosine expansion of the whole integral of the fourier transform. It is vectorised along the different options AND vectorised along the summation. It is 300x faster than QuantLib's Python implementation FFT of the Heston, because we can truncate and use far less terms to converge (Typically only 64-100 terms vs 1,000 terms) 

tools/Levenberg_Marquardt.py is my own implementation of a box-constrained Levenbrg-Marquardt Algorithm for calibrating the Heston model, and can calibrate all 5 parameters of the heston model. It converts the prices to implied volatilities and calibrated to implied volatilities. The damping factor is reduced based on the gain factor. Thus, the damping factor can still increase even if the step is accepted if the gain factor is too small. Using the gain factor to change in the damping facto is better than the tradition multiply/divide by 10 method as it results in far fewer rejected steps. To constrain the parameters, if the new parameter would exceed a bound, the new parameter value becomes the mid point of the old parameter and boundary value.  An acceleration parameter is used once the damping factor < 1e-5, which occurs after many accepted iterations. It compounds after every accepted iteration and resets once the damping factor. This reduces this iterations to find the local minimum, and the evidence for this is featured in accelerator_justification.png

Dependencies
```
pip install py_vollib_vectorized
```
