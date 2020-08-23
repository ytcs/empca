# empca

Python implementation of the algorithm presented in https://arxiv.org/abs/1910.14261. Please cite the paper if you use this library for research.

## Example Usage

``` python
from empca import *

X=ti_rfft(traces)         # perform phase-unwrapping FFT for time-shift invariance
W=np.diag(1/(psd))        # for non-stationary noise use inverse of full CSD matrix as weights

_empca=EMPCA(n_comp=3)    # optimal number of components needs to be tuned
chi2s= _empca.fit(X,W)     
recon=ti_irfft(_empca.coeff@_empca.eigvec) # reconstructed pulses from fitted amplitudes and templates
```
