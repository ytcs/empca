# empca

Python implementation of the algorithm presented in https://arxiv.org/abs/1910.14261. Please cite the paper if you use this library for research.

## Example Usage

``` python
from empca import *

W=np.diag(1/(psd))        # for non-stationary noise use full CSD matrix as weight
empca=EMPCA(n_comp=3)     # optimal number of components need to be tuned
X=ti_rfft(traces)         # Phase-unwrapping FFT for time-shift invariance
chi2s= empca.fit(X,W)     
recon=empca.coeff@empca.eigvec # reconstructed pulses from fitted amplitudes and templates
```
