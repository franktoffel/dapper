# Illustrate how to use DAPPER
# to benchmark a DA method using a "twin experiment".

# Load DAPPER (assumes current directory is <path-to-dapper>)
from common import *

# Load "twin experiment" setup: a hidden Markov Model (HMM)
from mods.Lorenz63.sak12 import HMM
HMM.t.T = 30 # shorten experiment

# Specify a DA method configuration
config = EnKF('Sqrt', N=10, infl=1.02, rot=True)
# config = Var3D(infl=0.9)
# config = PartFilt(N=100,reg=2.4,NER=0.3)

# These attributes affect live and replay plots
config.liveplotting=True
config.store_u=True

# Simulate synthetic truth (xx) and noisy obs (yy)
xx,yy = simulate(HMM)

# Assimilate yy, knowing the HMM; xx is used for assessment.
stats = config.assimilate(HMM,xx,yy)

# Average stats time series
avrgs = stats.average_in_time()

# Print averages
print_averages(config,avrgs,[],['rmse_a','rmv_a'])

# Replay liveplotters -- can adjust speed, time-window, etc.
replay(stats)

# Explore objects individually:
# print(HMM)
# print(config)
# print(stats)
# print(avrgs)

# Excercise: Try using
# - Optimal interpolation
# - The (extended) Kalman filter
# - The iterative EnKS
# Hint: suggested DA configs are listed in the HMM file.

# Excercise: Run an experiment for each of the models:
# - LotkaVolterra
# - Lorenz95
# - LA
# - QG


