# Reproduce raanes'2016 ("EnRTS and EnKS")

from common import *

from mods.Lorenz95.sak08 import HMM

HMM.t = Chronology(0.01,dkObs=15,T=4**5,BurnIn=20)

# from mods.Lorenz95.raanes2016 import HMM
# cfgs += EnKS ('Sqrt',N=25,infl=1.08,rot=False,Lag=12)
# cfgs += EnRTS('Sqrt',N=25,infl=1.08,rot=False,cntr=0.99)
# ...
# print_averages(cfgs,avrgs,[],['rmse_u'])
