
from matplotlib import pylab as plt
import dive_model
import no_ba_model
import asocial_model
import pymc
from pymc import MCMC, MAP
from pymc.Matplot import plot as mcplot


MASOC = MAP(asocial_model)
MASOC.fit(method ='fmin', iterlim=100000, tol=.000001)
print(MASOC.AIC)
print(MASOC.BIC)

MNOBA = MAP(no_ba_model)
MNOBA.fit(method ='fmin', iterlim=100000, tol=.000001)
print(MNOBA.AIC)
print(MNOBA.BIC)
print(MNOBA.dist.value)
print(MNOBA.lag.value)
print(MNOBA.intrinsic_rate.value)
print(MNOBA.social_rate.value)

MFULL = MAP(dive_model)
MFULL.fit(method ='fmin', iterlim=100000, tol=.000001)
print(MFULL.AIC)
print(MFULL.BIC)
print(MFULL.dist.value)
print(MFULL.lag.value)
print(MFULL.blind_angle.value)
print(MFULL.intrinsic_rate.value)
print(MFULL.social_rate.value)
