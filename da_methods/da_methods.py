from common import *

@DA_Config
def EnKF(upd_a,N,infl=1.0,rot=False,**kwargs):
  """
  The EnKF.

  Ref: Evensen, Geir. (2009):
  "The ensemble Kalman filter for combined state and parameter estimation."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    # Loop
    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      # Analysis update
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        E = EnKF_analysis(E,Obs(E,t),Obs.noise,yy[kObs],upd_a,stats,kObs)
        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator


def EnKF_analysis(E,Eo,hnoise,y,upd_a,stats,kObs):
    """
    The EnKF analysis update, in many flavours and forms,
    specified via 'upd_a'.

    Main references:
    Sakov and Oke (2008). "Implications of the Form of the ... EnSRF..."
    Sakov and Oke (2008). "A deterministic formulation of the EnKF..."
    """
    R = hnoise.C    # Obs noise cov
    N,Nx = E.shape  # Dimensionality
    N1  = N-1       # Ens size - 1

    mu = mean(E,0)  # Ens mean
    A  = E - mu     # Ens anomalies

    xo = mean(Eo,0) # Obs ens mean 
    Y  = Eo-xo      # Obs ens anomalies
    dy = y - xo     # Mean "innovation"

    if 'PertObs' in upd_a:
        # Uses classic, perturbed observations (Burgers'98)
        C  = Y.T @ Y + R.full*N1
        D  = mean0(hnoise.sample(N))
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        dE = (KG @ ( y - D - Eo ).T).T
        E  = E + dE

    elif 'Sqrt' in upd_a:
        # Uses a symmetric square root (ETKF)
        # to deterministically transform the ensemble.
        
        # The various versions below differ only numerically.
        # EVD is default, but for large N use SVD version.
        if upd_a == 'Sqrt' and N>Nx:
          upd_a = 'Sqrt svd'
        
        if 'explicit' in upd_a:
          # Not recommended due to numerical costs and instability.
          # Implementation using inv (in ens space)
          Pw = inv(Y @ R.inv @ Y.T + N1*eye(N))
          T  = sqrtm(Pw) * sqrt(N1)
          HK = R.inv @ Y.T @ Pw @ Y
          #KG = R.inv @ Y.T @ Pw @ A
        elif 'svd' in upd_a:
          # Implementation using svd of Y R^{-1/2}.
          V,s,_ = svd0(Y @ R.sym_sqrt_inv.T)
          d     = pad0(s**2,N) + N1
          Pw    = ( V * d**(-1.0) ) @ V.T
          T     = ( V * d**(-0.5) ) @ V.T * sqrt(N1) 
          trHK  = np.sum( (s**2+N1)**(-1.0) * s**2 ) # data/doc_snippets/trHK.jpg
        elif 'sS' in upd_a:
          # Same as 'svd', but with slightly different notation
          # (sometimes used by Sakov) using the normalization sqrt(N1).
          S     = Y @ R.sym_sqrt_inv.T / sqrt(N1)
          V,s,_ = svd0(S)
          d     = pad0(s**2,N) + 1
          Pw    = ( V * d**(-1.0) )@V.T / N1 # = G/(N1)
          T     = ( V * d**(-0.5) )@V.T
          trHK  = np.sum(  (s**2 + 1)**(-1.0)*s**2 ) # data/doc_snippets/trHK.jpg
        else: # 'eig' in upd_a:
          # Implementation using eig. val. decomp.
          d,V   = eigh(Y @ R.inv @ Y.T + N1*eye(N))
          T     = V@diag(d**(-0.5))@V.T * sqrt(N1)
          Pw    = V@diag(d**(-1.0))@V.T
          HK    = R.inv @ Y.T @ (V@ diag(d**(-1)) @V.T) @ Y
        w = dy @ R.inv @ Y.T @ Pw
        E = mu + w@A + T@A

    elif 'Serial' in upd_a:
        # Observations assimilated one-at-a-time:
        inds = serial_inds(upd_a, y, R, A)
        #  Requires de-correlation:
        dy   = dy @ R.sym_sqrt_inv.T
        Y    = Y  @ R.sym_sqrt_inv.T
        # Enhancement in the nonlinear case: re-compute Y each scalar obs assim.
        # But: little benefit, model costly (?), updates cannot be accumulated on S and T.

        if any(x in upd_a for x in ['Stoch','ESOPS','Var1']):
            # More details: Misc/Serial_ESOPS.py.
            # Settings for reproducing literature benchmarks may be found in
            # mods/Lorenz95/hot15.py
            for i,j in enumerate(inds):

              # Perturbation creation
              if 'ESOPS' in upd_a:
                # "2nd-O exact perturbation sampling"
                if i==0:
                  # Init -- increase nullspace by 1 
                  V,s,UT = tsvd(A,N1)
                  s[N-2:] = 0
                  A = reconst(V,s,UT)
                  v = V[:,N-2]
                else:
                  # Orthogonalize v wrt. the new A
                  # v   = Zj - Yj            # This formula (from paper) relies on Y=HX.
                  mult  = (v@A) / (Yj@A)     # Should be constant*ones(Nx), meaning that:
                  v     = v - mult[0]*Yj     # we can project v into ker(A) such that v@A is null.
                  v    /= sqrt(v@v)          # Normalize
                Zj  = v*sqrt(N1)             # Standardized perturbation along v
                Zj *= np.sign(rand()-0.5)    # Random sign
              else:
                # The usual stochastic perturbations. 
                Zj = mean0(randn(N)) # Un-coloured noise
                if 'Var1' in upd_a:
                  Zj *= sqrt(N/(Zj@Zj))

              # Select j-th obs
              Yj  = Y[:,j]       # [j] obs anomalies
              dyj = dy [j]       # [j] innov mean
              DYj = Zj - Yj      # [j] innov anomalies
              DYj = DYj[:,None]  # Make 2d vertical

              # Kalman gain computation
              C     = Yj@Yj + N1 # Total obs cov
              KGx   = Yj @ A / C # KG to update state
              KGy   = Yj @ Y / C # KG to update obs

              # Updates
              A    += DYj * KGx
              mu   += dyj * KGx
              Y    += DYj * KGy
              dy   -= dyj * KGy
            E = mu + A
        else:
            # "Potter scheme", "EnSRF"
            # - EAKF's two-stage "update-regress" form yields the same *ensemble* as this.
            # - The form below may be derived as "serial ETKF", but does not yield the same
            #   ensemble as 'Sqrt' (which processes obs as a batch) -- only the same mean/cov.
            T = eye(N)
            for j in inds:
              Yj = Y[:,j]
              C  = Yj@Yj + N1
              Tj = np.outer(Yj, Yj /  (C + sqrt(N1*C)))
              T -= Tj @ T
              Y -= Tj @ Y
            w = dy@Y.T@T/N1
            E = mu + w@A + T@A

    elif 'DEnKF' is upd_a:
        # Uses "Deterministic EnKF" (sakov'08)
        C  = Y.T @ Y + R.full*N1
        YC = mrdiv(Y, C)
        KG = A.T @ YC
        HK = Y.T @ YC
        E  = E + KG@dy - 0.5*(KG@Y.T).T

    else:
      raise KeyError("No analysis update method found: '" + upd_a + "'.") 

    # Diagnostic: relative influence of observations
    if 'trHK' in locals(): stats.trHK[kObs] = trHK     /hnoise.M
    elif 'HK' in locals(): stats.trHK[kObs] = trace(HK)/hnoise.M

    return E



def post_process(E,infl,rot):
  """
  Inflate, Rotate.

  To avoid recomputing/recombining anomalies, this should be inside EnKF_analysis().
  But for readability it is nicer to keep it as a separate function,
  also since it avoids inflating/rotationg smoothed states (for the EnKS).
  """
  do_infl = infl!=1.0 and infl!='-N'

  if do_infl or rot:
    A, mu = center(E)
    N,Nx  = E.shape
    T     = eye(N)

    if do_infl:
      T = infl * T

    if rot:
      T = genOG_1(N,rot) @ T

    E = mu + T@A
  return E



def add_noise(E, dt, noise, config):
  """
  Treatment of additive noise for ensembles.
  Settings for reproducing literature benchmarks may be found in
  mods/LA/raanes2015.py

  Ref: Raanes, Patrick Nima, Alberto Carrassi, and Laurent Bertino (2015):
  "Extending the square root method to account for additive forecast noise in ensemble methods."
  """
  method = config.get('fnoise_treatm','Stoch')

  if noise.C is 0: return E

  N,Nx = E.shape
  A,mu = center(E)
  Q12  = noise.C.Left
  Q    = noise.C.full

  def sqrt_core():
    T    = np.nan # cause error if used
    Qa12 = np.nan # cause error if used
    A2   = A.copy() # Instead of using (the implicitly nonlocal) A,
    # which changes A outside as well. NB: This is a bug in Datum!
    if N<=Nx:
      Ainv = tinv(A2.T)
      Qa12 = Ainv@Q12
      T    = funm_psd(eye(N) + dt*(N-1)*(Qa12@Qa12.T), sqrt)
      A2   = T@A2
    else: # "Left-multiplying" form
      P = A2.T @ A2 /(N-1)
      L = funm_psd(eye(Nx) + dt*mrdiv(Q,P), sqrt)
      A2= A2 @ L.T
    E = mu + A2
    return E, T, Qa12

  if method == 'Stoch':
    # In-place addition works (also) for empty [] noise sample.
    E += sqrt(dt)*noise.sample(N)
  elif method == 'none':
    pass
  elif method == 'Mult-1':
    varE   = np.var(E,axis=0,ddof=1).sum()
    ratio  = (varE + dt*diag(Q).sum())/varE
    E      = mu + sqrt(ratio)*A
    E      = reconst(*tsvd(E,0.999)) # Explained in Datum
  elif method == 'Mult-M':
    varE   = np.var(E,axis=0)
    ratios = sqrt( (varE + dt*diag(Q))/varE )
    E      = mu + A*ratios
    E      = reconst(*tsvd(E,0.999)) # Explained in Datum
  elif method == 'Sqrt-Core':
    E = sqrt_core()[0]
  elif method == 'Sqrt-Mult-1':
    varE0 = np.var(E,axis=0,ddof=1).sum()
    varE2 = (varE0 + dt*diag(Q).sum())
    E, _, Qa12 = sqrt_core()
    if N<=Nx:
      A,mu   = center(E)
      varE1  = np.var(E,axis=0,ddof=1).sum()
      ratio  = varE2/varE1
      E      = mu + sqrt(ratio)*A
      E      = reconst(*tsvd(E,0.999)) # Explained in Datum
  elif method == 'Sqrt-Add-Z':
    E, _, Qa12 = sqrt_core()
    if N<=Nx:
      Z  = Q12 - A.T@Qa12
      E += sqrt(dt)*(Z@randn((Z.shape[1],N))).T
  elif method == 'Sqrt-Dep':
    E, T, Qa12 = sqrt_core()
    if N<=Nx:
      # Q_hat12: reuse svd for both inversion and projection.
      Q_hat12      = A.T @ Qa12
      U,s,VT       = tsvd(Q_hat12,0.99)
      Q_hat12_inv  = (VT.T * s**(-1.0)) @ U.T
      Q_hat12_proj = VT.T@VT
      rQ = Q12.shape[1]
      # Calc D_til
      Z      = Q12 - Q_hat12
      D_hat  = A.T@(T-eye(N))
      Xi_hat = Q_hat12_inv @ D_hat
      Xi_til = (eye(rQ) - Q_hat12_proj)@randn((rQ,N))
      D_til  = Z@(Xi_hat + sqrt(dt)*Xi_til)
      E     += D_til.T
  else:
    raise KeyError('No such method')
  return E



# Reshapings used in smoothers to go to/from
# 3D arrays, where the 0th axis is the Lag index.
def reshape_to(E):
  K,N,Nx = E.shape
  return E.transpose([1,0,2]).reshape((N,K*Nx))
def reshape_fr(E,Nx):
  N,Km = E.shape
  K    = Km//Nx
  return E.reshape((N,K,Nx)).transpose([1,0,2])


@DA_Config
def EnKS(upd_a,N,Lag,infl=1.0,rot=False,**kwargs):
  """
  EnKS (ensemble Kalman smoother)

  Ref: Evensen, Geir. (2009):
  "The ensemble Kalman filter for combined state and parameter estimation."

  The only difference to the EnKF is the management of the lag and the reshapings.

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/raanes2016.py
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    E    = zeros((chrono.K+1,N,Dyn.M))
    E[0] = X0.sample(N)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E[k] = Dyn(E[k-1],t-dt,dt)
      E[k] = add_noise(E[k], dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E[k])

        kkLag    = range(k-Lag*chrono.dkObs, k+1)
        ELag     = E[kkLag]

        Eo       = Obs(E[k],t)
        y        = yy[kObs]

        ELag     = reshape_to(ELag)
        ELag     = EnKF_analysis(ELag,Eo,Obs.noise,y,upd_a,stats,kObs)
        E[kkLag] = reshape_fr(ELag,Dyn.M)
        E[k]     = post_process(E[k],infl,rot)
        stats.assess(k,kObs,'a',E=E[k])

    for k in progbar(range(chrono.K+1),desc='Assessing'):
      stats.assess(k,None,'u',E=E[k])
  return assimilator


@DA_Config
def EnRTS(upd_a,N,cntr,infl=1.0,rot=False,**kwargs):
  """
  EnRTS (Rauch-Tung-Striebel) smoother.

  Ref: Raanes, Patrick Nima. (2016):
  "On the ensemble Rauch‐Tung‐Striebel smoother..."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/raanes2016.py
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    E    = zeros((chrono.K+1,N,Dyn.M))
    Ef   = E.copy()
    E[0] = X0.sample(N)

    # Forward pass
    for k,kObs,t,dt in progbar(chrono.ticker):
      E[k]  = Dyn(E[k-1],t-dt,dt)
      E[k]  = add_noise(E[k], dt, Dyn.noise, kwargs)
      Ef[k] = E[k]

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E[k])
        Eo   = Obs(E[k],t)
        y    = yy[kObs]
        E[k] = EnKF_analysis(E[k],Eo,Obs.noise,y,upd_a,stats,kObs)
        E[k] = post_process(E[k],infl,rot)
        stats.assess(k,kObs,'a',E=E[k])

    # Backward pass
    for k in progbar(range(chrono.K)[::-1]):
      A  = center(E[k])[0]
      Af = center(Ef[k+1])[0]

      J = tinv(Af) @ A
      J *= cntr
      
      E[k] += ( E[k+1] - Ef[k+1] ) @ J

    for k in progbar(range(chrono.K+1),desc='Assessing'):
      stats.assess(k,E=E[k])
  return assimilator




def serial_inds(upd_a, y, cvR, A):
  if 'mono' in upd_a:
    # Not robust?
    inds = arange(len(y))
  elif 'sorted' in upd_a:
    dC = cvR.diag
    if np.all(dC == dC[0]):
      # Sort y by P
      dC = np.sum(A*A,0)/(N-1)
    inds = np.argsort(dC)
  else: # Default: random ordering
    inds = np.random.permutation(len(y))
  return inds

@DA_Config
def SL_EAKF(N,loc_rad,taper='GC',ordr='rand',infl=1.0,rot=False,**kwargs):
  """
  Serial, covariance-localized EAKF.

  Ref: Karspeck, Alicia R., and Jeffrey L. Anderson. (2007):
  "Experimental implementation of an ensemble adjustment filter..."

  Used without localization, this should be equivalent
  (full ensemble equality) to the EnKF 'Serial'.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    N1   = N-1
    R    = Obs.noise
    Rm12 = Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        y    = yy[kObs]
        inds = serial_inds(ordr, y, R, center(E)[0])
            
        state_taperer = Obs.localizer(loc_rad, 'y2x', t, taper)
        for j in inds:
          # Prep:
          # ------------------------------------------------------
          Eo = Obs(E,t)
          xo = mean(Eo,0)
          Y  = Eo - xo
          mu = mean(E ,0)
          A  = E-mu
          # Update j-th component of observed ensemble:
          # ------------------------------------------------------
          Y_j    = Rm12[j,:] @ Y.T
          dy_j   = Rm12[j,:] @ (y - xo)
          # Prior var * N1:
          sig2_j = Y_j@Y_j                
          if sig2_j<1e-9: continue
          # Update (below, we drop the locality subscript: _j)
          sig2_u = 1/( 1/sig2_j + 1/N1 )   # KG * N1
          alpha  = (N1/(N1+sig2_j))**(0.5) # Update contraction factor
          dy2    = sig2_u * dy_j/N1        # Mean update
          Y2     = alpha*Y_j               # Anomaly update
          # Update state (regress update from obs space, using localization)
          # ------------------------------------------------------
          ii, tapering = state_taperer(j)
          if len(ii) == 0: continue
          Regression = (A[:,ii]*tapering).T @ Y_j/np.sum(Y_j**2)
          mu[ ii]   += Regression*dy2
          A[:,ii]   += np.outer(Y2 - Y_j, Regression)
          # Without localization:
          # Regression = A.T @ Y_j/np.sum(Y_j**2)
          # mu        += Regression*dy2
          # A         += np.outer(Y2 - Y_j, Regression)

          E = mu + A

        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator



def effective_N(YR,dyR,xN,g):
  """
  Effective ensemble size N,
  as measured by the finite-size EnKF-N
  """
  N,Ny = YR.shape
  N1   = N-1

  V,s,UT = svd0(YR)
  du     = UT @ dyR

  eN, cL = hyperprior_coeffs(s,N,xN,g)

  pad_rk = lambda arr: pad0( arr, min(N,Ny) )
  dgn_rk = lambda l: pad_rk((l*s)**2) + N1

  # Make dual cost function (in terms of l1)
  J      = lambda l:          np.sum(du**2/dgn_rk(l)) \
           +    eN/l**2 \
           +    cL*log(l**2)
  # Derivatives (not required with minimize_scalar):
  Jp     = lambda l: -2*l   * np.sum(pad_rk(s**2) * du**2/dgn_rk(l)**2) \
           + -2*eN/l**3 \
           +  2*cL/l
  Jpp    = lambda l: 8*l**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l)**3) \
           +  6*eN/l**4 \
           + -2*cL/l**2
  # Find inflation factor (optimize)
  l1 = Newton_m(Jp,Jpp,1.0)
  #l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
  #l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2), tol=1e-4).x

  za = N1/l1**2
  return za


@DA_Config
def LETKF(N,loc_rad,taper='GC',infl=1.0,rot=False,mp=False,**kwargs):
  """
  Same as EnKF (sqrt), but with localization.

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py

  Ref: Hunt, Brian R., Eric J. Kostelich, and Istvan Szunyogh. (2007):
  "Efficient data assimilation for spatiotemporal chaos..."
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0,R,N1 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, N-1

    _map = multiproc_map if mp else map

    # Warning: if len(ii) is small, analysis may be slowed-down with '-N' infl
    xN = kwargs.get('xN',1.0)
    g  = kwargs.get('g',0)

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)

        # Decompose ensmeble
        mu = mean(E,0)
        A  = E - mu
        # Obs space variables
        y    = yy[kObs]
        Y,xo = center(Obs(E,t))
        # Transform obs space
        Y  = Y        @ R.sym_sqrt_inv.T
        dy = (y - xo) @ R.sym_sqrt_inv.T

        state_batches, obs_taperer = Obs.localizer(loc_rad, 'x2y', t, taper)
        # for ii in state_batches:
        def local_analysis(ii):

          # Locate local obs
          jj, tapering = obs_taperer(ii)
          if len(jj) == 0: return E[:,ii], N1 # no update
          Y_jj   = Y[:,jj]
          dy_jj  = dy [jj]

          # Adaptive inflation
          za = effective_N(Y_jj,dy_jj,xN,g) if infl=='-N' else N1

          # Taper
          Y_jj  *= sqrt(tapering)
          dy_jj *= sqrt(tapering)

          # Compute ETKF update
          if len(jj) < N:
            # SVD version
            V,sd,_ = svd0(Y_jj)
            d      = pad0(sd**2,N) + za
            Pw     = (V * d**(-1.0)) @ V.T
            T      = (V * d**(-0.5)) @ V.T * sqrt(za)
          else:
            # EVD version
            d,V   = eigh(Y_jj@Y_jj.T + za*eye(N))
            T     = V@diag(d**(-0.5))@V.T * sqrt(za)
            Pw    = V@diag(d**(-1.0))@V.T
          AT  = T @ A[:,ii]
          dmu = dy_jj @ Y_jj.T @ Pw @ A[:,ii]
          Eii = mu[ii] + dmu + AT
          return Eii, za

        # Run local analyses
        zz = []
        result = _map(local_analysis, state_batches )
        for ii, (Eii, za) in zip(state_batches, result):
          zz += [za]
          E[:,ii] = Eii

        # Global post-processing
        E = post_process(E,infl,rot)

        stats.infl[kObs] = sqrt(N1/mean(zz))

      stats.assess(k,kObs,E=E)
  return assimilator





# Notes on optimizers for the 'dual' EnKF-N:
# ----------------------------------------
#  Using minimize_scalar:
#  - Doesn't take dJdx. Advantage: only need J
#  - method='bounded' not necessary and slower than 'brent'.
#  - bracket not necessary either...
#  Using multivariate minimization: fmin_cg, fmin_bfgs, fmin_ncg
#  - these also accept dJdx. But only fmin_bfgs approaches
#    the speed of the scalar minimizers.
#  Using scalar root-finders:
#  - brenth(dJ1, LowB, 1e2,     xtol=1e-6) # Same speed as minimization
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6) # No improvement
#  - newton(dJ1,1.0, fprime=dJ2, tol=1e-6, fprime2=dJ3) # No improvement
#  - Newton_m(dJ1,dJ2, 1.0) # Significantly faster. Also slightly better CV?
# => Despite inconvienience of defining analytic derivatives,
#    Newton_m seems like the best option.
#  - In extreme (or just non-linear Obs.mod) cases,
#    the EnKF-N cost function may have multiple minima.
#    Then: should use more robust optimizer!
#
# For 'primal'
# ----------------------------------------
# Similarly, Newton_m seems like the best option,
# although alternatives are provided (commented out).
#
def Newton_m(fun,deriv,x0,is_inverted=False,
    conf=1.0,xtol=1e-4,ytol=1e-7,itermax=10**2):
  "Simple (and fast) implementation of Newton root-finding"
  itr, dx, Jx = 0, np.inf, fun(x0)
  norm = lambda x: sqrt(np.sum(x**2))
  while ytol<norm(Jx) and xtol<norm(dx) and itr<itermax:
    Dx  = deriv(x0)
    if is_inverted:
      dx  = Dx @ Jx
    elif isinstance(Dx,float):
      dx  = Jx/Dx
    else:
      dx  = mldiv(Dx,Jx)
    dx *= conf
    x0 -= dx
    Jx  = fun(x0)
  return x0


def hyperprior_coeffs(s,N,xN=1,g=0):
    """
    EnKF-N inflation prior may be specified by the constants:
     - eN: Effect of unknown mean
     - cL: Coeff in front of log term

    These are trivial constants in the original EnKF-N,
    but could be further adjusted (corrected and tuned),
    as described below.
     
    Reason 1: mode correction.
    As a func of I-KH ("prior's weight"), adjust l1's mode towards 1.
    As noted in Boc15, mode correction becomes necessary when R-->infty,
    because then there should be no ensemble update (and also no inflation!).
    Why leave the prior mode below 1 at all?
    Because it sets up "tension" (negative feedback) in the inflation cycle:
    the prior pulls downwards, while the likelihood tends to pull upwards.
     
    Reason 2: Boosting the inflation prior's certainty from N to xN*N.
    The aim is to take advantage of the fact that the ensemble may not
    have quite as much sampling error as a fully stochastic sample,
    as illustrated in section 2.1 of Raanes2018adaptive.
    The tuning is controlled by:
     - xN=1: is fully agnostic, i.e. assumes the ensemble is generated
             from a highly chaotic or stochastic model.
     - xN>1: increases the certainty of the hyper-prior,
             which is appropriate for more linear and deterministic systems.
     - xN<1: yields a more (than 'fully') agnostic hyper-prior,
             as if N were smaller than it truly is.
     - xN<=0 is not meaningful.
    This parameter has not yet been explicitly described in the litteture,
    although if effectively constitutes a bridging of
    the Jeffreys (xN=1) and Dirac (xN=Inf) hyperpriors for the prior covariance, B,
    which are discussed in Boc15.
    Its effect of damping has also been employed in Anderson's inflation work.

    Reference:
    Raanes2018adaptive: Patrick N. Raanes, Marc Bocquet, Alberto Carrassi (2018):
    "Adaptive covariance inflation in the EnKF Gaussian scale mixtures"
    """
    N1 = N-1

    eN = (N+1)/N
    cL = (N+g)/N1

    # Mode correction (almost) as in eqn 36 of Boc15
    prior_mode = eN/cL                     # Mode of l1 (before correction)
    diagonal   = pad0( s**2, N ) + N1      # diag of Y@R.inv@Y + N1*I [Hessian of J(w)]
    I_KH       = mean( diagonal**(-1) )*N1 # ≈ 1/(1 + HBH/R)
    #I_KH      = 1/(1 + (s**2).sum()/N1)   # Scalar alternative: use tr(HBH/R).
    mc         = sqrt(prior_mode**I_KH)    # Correction coeff

    # Apply correction
    eN /= mc
    cL *= mc

    # Apply xN 
    eN *= xN
    cL *= xN

    return eN, cL

def zeta_a(eN,cL,w):
  """
  EnKF-N inflation estimation via w.
  Returns zeta_a = (N-1)/pre-inflation^2.

  Using this inside an iterative minimization as in the iEnKS
  effectively blends the distinction between the primal and dual EnKF-N.
  """
  N  = len(w)
  N1 = N-1
  za = N1*cL/(eN + w@w)
  return za


@DA_Config
def EnKF_N(N,dual=True,Hess=False,g=0,xN=1.0,infl=1.0,rot=False,**kwargs):
  """
  Finite-size EnKF (EnKF-N).

  Reference:
  Boc15: Bocquet, Marc, Patrick N. Raanes, and Alexis Hannart. (2015):
  "Expanding the validity of the ensemble Kalman filter..."

  This implementation is pedagogical, prioritizing the "dual" form.
  In consequence, the efficiency of the "primal" form suffers a bit.
  The primal form is there mainly to demonstrate equivalence
  (to the dual, provided the same minimum is found by the optimizers),
  and to allow comparison with iEnKS(), which uses the full, non-lin Obs.mod.

  'infl' should be unnecessary (assuming no model error, or that Q is correct).

  'Hess': use non-approx Hessian for ensemble transform matrix?

  'g' is the nullity of A (state anomalies's), ie. g=max(1,N-Nx),
  compensating for the redundancy in the space of w.
  But we have made it an input argument instead, with default 0,
  because mode-finding (of p(x) via the dual) completely ignores this redundancy,
  and the mode gets (undesireably) modified by g.

  'xN' allows tuning the hyper-prior for the inflation.
  Usually, I just try setting it to 1 (default), or 2.
  Further description in hyperprior_coeffs().

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz63/sak12.py
  """
  def assimilator(stats,HMM,xx,yy):
    # Unpack
    Dyn,Obs,chrono,X0,R,N1 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, N-1

    # Init
    E = X0.sample(N)
    stats.assess(0,E=E)

    # Loop
    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      # Analysis
      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        Eo = Obs(E,t)
        y  = yy[kObs]

        mu = mean(E,0)
        A  = E - mu

        xo = mean(Eo,0)
        Y  = Eo-xo
        dy = y - xo

        V,s,UT = svd0  (Y @ R.sym_sqrt_inv.T)
        du     = UT @ (dy @ R.sym_sqrt_inv.T)
        dgn_N  = lambda l: pad0( (l*s)**2, N ) + N1

        # Adjust hyper-prior
        #xN_ = noise_level(xN,stats,chrono,N1,kObs,A,locals().get('A_old',None))
        eN, cL = hyperprior_coeffs(s,N,xN,g)

        if dual:
            # Make dual cost function (in terms of l1)
            pad_rk = lambda arr: pad0( arr, min(N,Obs.M) )
            dgn_rk = lambda l: pad_rk((l*s)**2) + N1
            J      = lambda l:          np.sum(du**2/dgn_rk(l)) \
                     +    eN/l**2 \
                     +    cL*log(l**2)
            # Derivatives (not required with minimize_scalar):
            Jp     = lambda l: -2*l   * np.sum(pad_rk(s**2) * du**2/dgn_rk(l)**2) \
                     + -2*eN/l**3 \
                     +  2*cL/l
            Jpp    = lambda l: 8*l**2 * np.sum(pad_rk(s**4) * du**2/dgn_rk(l)**3) \
                     +  6*eN/l**4 \
                     + -2*cL/l**2
            # Find inflation factor (optimize)
            l1 = Newton_m(Jp,Jpp,1.0)
            #l1 = fmin_bfgs(J, x0=[1], gtol=1e-4, disp=0)
            #l1 = minimize_scalar(J, bracket=(sqrt(prior_mode), 1e2), tol=1e-4).x
        else:
            # Primal form, in a fully linearized version.
            za     = lambda w: zeta_a(eN,cL,w)
            J      = lambda w: .5*np.sum(( (dy-w@Y)@R.sym_sqrt_inv.T)**2 ) + \
                               .5*N1*cL*log(eN + w@w)
            # Derivatives (not required with fmin_bfgs):
            Jp     = lambda w: -Y@R.inv@(dy-w@Y) + w*za(w)
            #Jpp   = lambda w:  Y@R.inv@Y.T + za(w)*(eye(N) - 2*np.outer(w,w)/(eN + w@w))
            #Jpp   = lambda w:  Y@R.inv@Y.T + za(w)*eye(N) # approx: no radial-angular cross-deriv
            nvrs   = lambda w: (V * (pad0(s**2,N) + za(w))**-1.0) @ V.T # inverse of Jpp-approx
            # Find w (optimize)
            wa     = Newton_m(Jp,nvrs,zeros(N),is_inverted=True)
            #wa    = Newton_m(Jp,Jpp ,zeros(N))
            #wa    = fmin_bfgs(J,zeros(N),Jp,disp=0)
            l1     = sqrt(N1/za(wa))

        # Uncomment to revert to ETKF
        #l1 = 1.0

        # Explicitly inflate prior => formulae look different from Boc15.
        A *= l1
        Y *= l1

        # Compute sqrt update
        Pw      = (V * dgn_N(l1)**(-1.0)) @ V.T
        w       = dy@R.inv@Y.T@Pw
        # For the anomalies:
        if not Hess:
          # Regular ETKF (i.e. sym sqrt) update (with inflation)
          T     = (V * dgn_N(l1)**(-0.5)) @ V.T * sqrt(N1)
          #     = (Y@R.inv@Y.T/N1 + eye(N))**(-0.5)
        else:
          # Also include angular-radial co-dependence.
          # Note: denominator not squared coz unlike Boc15 we have inflated Y.
          Hw    = Y@R.inv@Y.T/N1 + eye(N) - 2*np.outer(w,w)/(eN + w@w)
          T     = funm_psd(Hw, lambda x: x**-.5) # is there a sqrtm Woodbury?
          
        E = mu + w@A + T@A
        E = post_process(E,infl,rot)

        stats.infl[kObs] = l1
        stats.trHK[kObs] = (((l1*s)**2 + N1)**(-1.0)*s**2).sum()/HMM.Ny

      stats.assess(k,kObs,E=E)
  return assimilator



@DA_Config
def iEnKS(upd_a,N,Lag=1,nIter=10,wTol=0,MDA=False,step=False,bundle=False,xN=None,infl=1.0,rot=False,**kwargs):
  """
  Iterative EnKS.

  As in Boc14, optimization uses Gauss-Newton. See Boc12 for Levenberg-Marquardt.
  If MDA=True, then there's not really any optimization, but rather Gaussian annealing.

  - upd_a : Analysis update form (flavour). One of:
    * 'Sqrt'   : as in ETKF  , using a deterministic matrix square root transform.
    * 'PertObs': as in EnRML , using stochastic, perturbed-observations.
    * 'Order1' : as in DEnKF of Sakov (2008).
  - Lag   : Length of the DA window (DAW), in multiples of dkObs (i.e. cycles).
            Lag=1 (default) => iterative "filter" iEnKF (Sakov et al 2012).
            Lag=0           => maximum-likelihood filter (Zupanski 2005).
  - Shift : How far (in cycles) to slide the DAW. Fixed at 1 for code simplicity.
  - nIter : Maximal num. of iterations used (>=1).
            Supporting nIter==0 requires more code than it's worth.
  - wTol  : Rel. tolerance defining convergence. Default: 0 => always do nIter iterations.
  - MDA   : Use iterations of the "multiple data assimlation" type.
  - bundle: Use finite-diff. linearization instead of of least-squares regression.
  - xN    : If set, use EnKF_N() pre-inflation. See further documentation there.

  Total number of model simulations (of duration dtObs): N * (nIter*Lag + 1).
  (due to boundary cases: only asymptotically valid)
  
  References:
  - Boc12: M. Bocquet, and P. Sakov. (2012): "Combining inflation-free and iterative EnKFs ..."
  - Boc14: M. Bocquet, and P. Sakov. (2014): "An iterative ensemble Kalman smoother."

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/sak08.py
  mods/Lorenz63/sak12.py
  mods/Lorenz63/boc12.py
  """

  # NB It's very difficult to know what should happen to all of the time indices in all cases of
  # nIter and Lag. => Any changes to this function must be unit-tested via scripts/test_iEnKS.py.

  # TODO:
  # - Implement MDA (quasi-static optimization). Boc notes:
  #   * The 'balancing step' is complicated.
  #   * Trouble playing nice with '-N' inflation estimation.
  # - Modularize upd_a (and localization?)

  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0,R,KObs, N1 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, HMM.t.KObs, N-1
    Rm12 = Obs.noise.C.sym_sqrt_inv

    assert Dyn.noise.C is 0, "Q>0 not yet supported. TODO: implement Sakov et al 2017"

    # Makes the iEnKS very much alike the iExtKS. 
    if bundle: EPS = 1e-4 # Sakov/Boc use T=1e-4*eye(N), but I prefer using T*=1e-4,
    else     : EPS = 1.0  # yielding a conditional shape for the bundle.

    def make_CVar(E,scaling=1.0):
      "Define change-of-var's from ens space to ens subspace (of state space)."
      X,x = center(E)
      w2x = lambda w: x + (scaling*w)@X
      return w2x

    # Initial ensemble
    E = X0.sample(N)

    # Loop for sliding the DA window (DAW).
    for DAW_right in progbar(arange(-1,KObs+Lag+1)):
        DAW_left = DAW_right-Lag+1 # => len(DAW)==Lag

        # Crop to to obs chronology
        DAW = range( max(0,DAW_left), min(KObs,DAW_right) + 1)

        if 0<=DAW_right<=KObs: # if ∃ "not-fully-assimlated" obs (todo: quasi-static).

            # Init iterations.
            w2x_left = make_CVar(E) # CVar for x[DAW_left-1] based on ens. prior to yy[DAW_right] assim.
            w        = zeros(N)     # Control vector for the mean state.
            T        = eye(N)       # Anomalies transform matrix.
            Tinv     = eye(N)       # Rather than on-demand tinv(T), an explicit Tinv variable allows for
                                    # merging MDA code with iEnKS/EnRML, and sometimes for flop savings.

            for iteration in arange(nIter):

                # Forecast from [DAW_left-1] to [DAW_right]
                E = w2x_left(w + T*EPS)                         # Reconstruct smoothed ensemble.
                for kObs in DAW:                                # Loop len(DAW) cycles.
                  for k,t,dt in chrono.cycle_to_obs(kObs):      # Loop dkObs steps
                    E = Dyn(E,t-dt,dt)                          # Forecast 1 dt step
                if iteration==0: w2x_right = make_CVar(E,1/EPS) # CVar for x[DAW_right], used for f/a stats.

                # Prepare analysis of current obs: yy[kObs], where kObs==DAW_right if len(DAW)>0
                Eo     = Obs(E,t)          # Observe ensemble.
                y      = yy[DAW_right]     # Select current obs/data.
                Y,xo   = center(Eo)        # Get obs {anomalies, mean}.
                dy     = (y - xo) @ Rm12.T # Transform obs space.
                Y      = Y        @ Rm12.T # Transform obs space.
                Y0     = (Tinv/EPS) @ Y    # "De-condition" the obs anomalies.
                V,s,UT = svd0(Y0)          # Decompose Y0.

                # Set the prior cov normalizt. factor ("effective ensemble size").
                # => set the pre-inflation, implicitly via: za = (N-1)/infl^2.
                if xN is None: za  = N1
                else         : za  = zeta_a(*hyperprior_coeffs(s,N,xN), w)
                if MDA       : za *= nIter # inflation (factor: nIter) of the ObsErrCov

                # TODO
                # if step and MDA==False:
                  # if iteration==0 and kObs==0:
                    # warnings.warn("Using step length")
                  # step_length = sin((iteration+1)/(nIter+1)*pi) # smaller => shorter steps
                  # max_length  = max(sin((arange(nIter)+1)/(nIter+1)*pi))
                  # step_length /= max_length # normalize => always one occurance of 1

                # Post. cov (approx) of w, estimated at current iteration, raised to power.
                Pw_pwr = lambda expo: (V * (pad0(s**2,N) + za)**-expo) @ V.T
                Pw     = Pw_pwr(1.0)

                if MDA: # Frame update using annealing (progressive assimilation).
                    Pw = Pw @ T # apply previous update
                    w += dy @ Y.T @ Pw
                    if 'PertObs' in upd_a:   #== "ES-MDA". By Emerick/Reynolds.
                      D     = mean0(randn(Y.shape)) * sqrt(nIter)
                      T    -= (Y + D) @ Y.T @ Pw
                    elif 'Sqrt' in upd_a:    #== "ETKF-ish". By Raanes.
                      T     = Pw_pwr(0.5) * sqrt(za) @ T
                    elif 'Order1' in upd_a:  #== "DEnKF-ish". By Emerick.
                      T    -= 0.5 * Y @ Y.T @ Pw
                    # Tinv = eye(N) [as initialized] coz MDA does not de-condition.

                else: # Frame update as Gauss-Newton optimzt. of log-posterior.
                    grad  = Y0@dy - w*za # Cost function gradient
                    w     = w + grad@Pw  # Gauss-Newton step
                    if 'Sqrt' in upd_a:      #== "ETKF-ish". By Bocquet/Sakov.
                      T     = Pw_pwr(0.5) * sqrt(N1) # Sqrt-transforms
                      Tinv  = Pw_pwr(-.5) / sqrt(N1) # Saves time [vs tinv(T)] when Nx<N
                    elif 'PertObs' in upd_a: #== "EnRML". By Oliver/Chen/Raanes/Evensen/Stordal.
                      D     = mean0(randn(Y.shape)) if iteration==0 else D
                      gradT = -(Y+D)@Y0.T + N1*(eye(N) - T)
                      T     = T + gradT@Pw
                      Tinv  = tinv(T, threshold=N1)
                    elif 'Order1' in upd_a:  #== "DEnKF-ish". By Raanes.
                      # Included for completeness; a non-MDA Order1 version does not make much sense.
                      gradT = -0.5*Y@Y0.T + N1*(eye(N) - T)
                      T     = T + gradT@Pw
                      Tinv  = tinv(T, threshold=N1)
                # END if MDA

                # Check for convergence
                if wTol and np.linalg.norm(w) < N*1e-4: break

            # END loop iteration

            # Forecast ('f') and analysis ('a') stats for E[DAW_right]. The 'a' stats might also be recorded
            # at the end of the last iteration or during first iteration (after the shift), which would yield
            # different results in the non-linear case. The definition used here has the
            # advantage of being simple and not breaking in the special cases of L=0 and/or nIter=1.
            stats.assess(k,   DAW_right, 'f', E = w2x_right(eye(N)) )
            stats.assess(k,   DAW_right, 'a', E = w2x_right(w+T) )
            stats.iters      [DAW_right] = iteration+1
            if xN: stats.infl[DAW_right] = sqrt(N1/za)
            stats.trHK       [DAW_right] = trace(Y0.T @ Pw @ Y0)/HMM.Ny # TODO mda

            # Final estimate of E at [DAW_left-1]
            E = w2x_left(w+T)
            E = post_process(E,infl,rot)
        # END if 0<=DAW_right<=KObs

        # Shift/Slide DAW by propagating smoothed ('u') ensemble [DAW_left-1]
        if 0 <= DAW_left <= KObs:
          for k,t,dt in chrono.cycle_to_obs(DAW_left):
            stats.assess(k-1,None,'u',E=E)
            E = Dyn(E,t-dt,dt)
    # END loop DAW_right
    stats.assess(k,None,'u',E=E)
  return assimilator




@DA_Config
def iLEnKS(upd_a,N,loc_rad,taper='GC',Lag=1,nIter=10,xN=1.0,infl=1.0,rot=False,**kwargs):
  """
  Iterative, Localized EnKS-N.

  Based on iEnKS() and LETKF() codes,
  which describes the other input arguments.

  - upd_a : - 'Sqrt' (i.e. ETKF)
            - '-N' (i.e. EnKF-N)
  
  References:
  Raanes (2018) : "Stochastic, iterative ensemble smoothers"
  Bocquet (2015): "Localization and the iterative ensemble Kalman smoother"

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/boc15loc.py
  """

  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0,R,KObs, N1 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0, HMM.Obs.noise.C, HMM.t.KObs, N-1
    assert Dyn.noise.C is 0, "Q>0 not yet supported. See Sakov et al 2017: 'An iEnKF with mod. error'"

    # Init DA cycles
    E = X0.sample(N)

    # Loop DA cycles
    for kObs in progbar(arange(KObs+1)):
        # Set (shifting) DA Window. Shift is 1.
        DAW_0  = kObs-Lag+1
        DAW    = arange(max(0,DAW_0),DAW_0+Lag)
        DAW_dt = chrono.ttObs[DAW[-1]] - chrono.ttObs[DAW[0]] + chrono.dtObs

        # Get localization setup (at time t)
        state_batches, obs_taperer = Obs.localizer(loc_rad, 'x2y', chrono.ttObs[kObs], taper)
        nBatch = len(state_batches)

        # Store 0th (iteration) estimate as (x0,A0)
        A0,x0  = center(E)
        # Init iterations
        w      = np.tile( zeros(N) , (nBatch,1) )
        Tinv   = np.tile( eye(N)   , (nBatch,1,1) )
        T      = np.tile( eye(N)   , (nBatch,1,1) )

        # Loop iterations
        for iteration in arange(nIter):

            # Assemble current estimate of E[kObs-Lag]
            for ib, ii in enumerate(state_batches):
              E[:,ii] = x0[ii] + w[ib]@A0[:,ii]+ T[ib]@A0[:,ii]

            # Forecast
            for kDAW in DAW:                           # Loop Lag cycles
              for k,t,dt in chrono.cycle_to_obs(kDAW): # Loop dkObs steps (1 cycle)
                E = Dyn(E,t-dt,dt)                     # Forecast 1 dt step (1 dkObs)

            if iteration==0:
              stats.assess(k,kObs,'f',E=E)

            # Analysis of y[kObs] (already assim'd [:kObs])
            y    = yy[kObs]
            Y,xo = center(Obs(E,t))
            # Transform obs space
            Y  = Y        @ R.sym_sqrt_inv.T
            dy = (y - xo) @ R.sym_sqrt_inv.T

            # Inflation estimation.
            # Set "effective ensemble size", za = (N-1)/pre-inflation^2.
            if upd_a == 'Sqrt':
                za = N1 # no inflation
            elif upd_a == '-N':
              # Careful not to overwrite w,T,Tinv !
              V,s,UT = svd0(Y)
              grad   = Y@dy
              Pw     = (V * (pad0(s**2,N) + N1)**-1.0) @ V.T
              w_glob = Pw@grad 
              za     = zeta_a(*hyperprior_coeffs(s,N,xN), w_glob)
            else: raise KeyError("upd_a: '" + upd_a + "' not matched.") 

            for ib, ii in enumerate(state_batches):
              # Shift indices (to adjust for time difference)
              ii_kObs = Obs.loc_shift(ii, DAW_dt)
              # Localize
              jj, tapering = obs_taperer(ii_kObs)
              if len(jj) == 0: continue
              Y_jj   = Y[:,jj] * sqrt(tapering)
              dy_jj  = dy[jj]  * sqrt(tapering)

              # "Uncondition" the observation anomalies
              # (and yet this linearization of Obs.mod improves with iterations)
              Y_jj     = Tinv[ib] @ Y_jj
              # Prepare analysis: do SVD
              V,s,UT   = svd0(Y_jj)
              # Gauss-Newton ingredients
              grad     = -Y_jj@dy_jj + w[ib]*za
              Pw       = (V * (pad0(s**2,N) + za)**-1.0) @ V.T
              # Conditioning for anomalies (discrete linearlizations)
              T[ib]    = (V * (pad0(s**2,N) + za)**-0.5) @ V.T * sqrt(N1)
              Tinv[ib] = (V * (pad0(s**2,N) + za)**+0.5) @ V.T / sqrt(N1)
              # Gauss-Newton step
              dw       = Pw@grad
              w[ib]   -= dw

            # Stopping condition # TODO
            #if np.linalg.norm(dw) < N*1e-4:
              #break

        # Analysis 'a' stats for E[kObs].
        stats.assess(k,kObs,'a',E=E)
        stats.trHK [kObs] = trace(Y.T @ Pw @ Y)/HMM.Ny
        stats.infl [kObs] = sqrt(N1/za)
        stats.iters[kObs] = iteration+1

        # Final (smoothed) estimate of E[kObs-Lag]
        for ib, ii in enumerate(state_batches):
          E[:,ii] = x0[ii] + w[ib]@A0[:,ii]+ T[ib]@A0[:,ii]

        E = post_process(E,infl,rot)

        # Forecast smoothed ensemble by shift (1*dkObs)
        if DAW_0 >= 0:
          for k,t,dt in chrono.cycle_to_obs(DAW_0):
            stats.assess(k-1,None,'u',E=E)
            E = Dyn(E,t-dt,dt)

    # Assess the last (Lag-1) obs ranges
    for kDAW in arange(DAW[0]+1,KObs+1):
      for k,t,dt in chrono.cycle_to_obs(kDAW):
        stats.assess(k-1,None,'u',E=E)
        E = Dyn(E,t-dt,dt)
    stats.assess(chrono.K,None,'u',E=E)

  return assimilator



@DA_Config
def PartFilt(N,NER=1.0,resampl='Sys',reg=0,nuj=True,qroot=1.0,wroot=1.0,**kwargs):
  """
  Particle filter ≡ Sequential importance (re)sampling SIS (SIR).
  This is the bootstrap version: the proposal density is just
  q(x_0:t|y_1:t) = p(x_0:t) = p(x_t|x_{t-1}) p(x_0:{t-1}).

  Ref:
  [1]: Christopher K. Wikle, L. Mark Berliner, 2006:
    "A Bayesian tutorial for data assimilation"
  [2]: Van Leeuwen, 2009: "Particle Filtering in Geophysical Systems"
  [3]: Zhe Chen, 2003:
    "Bayesian Filtering: From Kalman Filters to Particle Filters, and Beyond"

  Tuning settings:
   - NER: Trigger resampling whenever N_eff <= N*NER.
       If resampling with some variant of 'Multinomial',
       no systematic bias is introduced.
   - qroot: "Inflate" (anneal) the proposal noise kernels
       by this root to increase diversity.
       The weights are updated to maintain un-biased-ness.
       Ref: [3], section VI-M.2


  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/boc10.py and mods/Lorenz95/boc10_m40.py.
  Other interesting settings include: mods/Lorenz63/sak12.py
  """
  # TODO:
  #if miN < 1:
    #miN = N*miN

  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx, Rm12 = Dyn.M, Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    w = 1/N*ones(N)

    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      if Dyn.noise.C is not 0:
        D  = randn((N,Nx))
        E += sqrt(dt*qroot)*(D@Dyn.noise.C.Right)

        if qroot != 1.0:
          # Evaluate p/q (for each col of D) when q:=p**(1/qroot).
          w *= exp(-0.5*np.sum(D**2, axis=1) * (1 - 1/qroot))
          w /= w.sum()

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)

        innovs = (yy[kObs] - Obs(E,t)) @ Rm12.T
        w      = reweight(w,innovs=innovs)

        if trigger_resampling(w, NER, [stats,E,k,kObs]):
          C12    = reg*bandw(N,Nx)*raw_C12(E,w)
          #C12  *= sqrt(rroot) # Re-include?
          idx,w  = resample(w, resampl, wroot=wroot)
          E,chi2 = regularize(C12,E,idx,nuj)
          #if rroot != 1.0:
            # Compensate for rroot
            #w *= exp(-0.5*chi2*(1 - 1/rroot))
            #w /= w.sum()
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator



@DA_Config
def OptPF(N,Qs,NER=1.0,resampl='Sys',reg=0,nuj=True,wroot=1.0,**kwargs):
  """
  "Optimal proposal" particle filter.
  OR
  Implicit particle filter.

  Note: Regularization (Qs) is here added BEFORE Bayes' rule.
  If Qs==0: OptPF should be equal to the bootsrap filter: PartFilt().

  Ref: Bocquet et al. (2010):
    "Beyond Gaussian statistical modeling in geophysical data assimilation"

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/boc10.py and mods/Lorenz95/boc10_m40.py.
  Other interesting settings include: mods/Lorenz63/sak12.py
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx, R = Dyn.M, Obs.noise.C.full

    E = X0.sample(N)
    w = 1/N*ones(N)

    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      if Dyn.noise.C is not 0:
        E += sqrt(dt)*(randn((N,Nx))@Dyn.noise.C.Right)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)
        y = yy[kObs]

        Eo = Obs(E,t)
        innovs = y - Eo

        # EnKF-ish update
        s   = Qs*bandw(N,Nx)
        As  = s*raw_C12(E,w)
        Ys  = s*raw_C12(Eo,w)
        C   = Ys.T@Ys + R
        KG  = As.T@mrdiv(Ys,C)
        E  += sample_quickly_with(As)[0]
        D   = Obs.noise.sample(N)
        dE  = KG @ (y-Obs(E,t)-D).T
        E   = E + dE.T

        # Importance weighting
        chi2   = innovs*mldiv(C,innovs.T).T
        logL   = -0.5 * np.sum(chi2, axis=1)
        w      = reweight(w,logL=logL)
        
        # Resampling
        if trigger_resampling(w, NER, [stats,E,k,kObs]):
          C12    = reg*bandw(N,Nx)*raw_C12(E,w)
          idx,w  = resample(w, resampl, wroot=wroot)
          E,_    = regularize(C12,E,idx,nuj)

      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator



@DA_Config
def PFa(N,alpha,NER=1.0,resampl='Sys',reg=0,nuj=True,qroot=1.0,**kwargs):
  """
  PF with weight adjustment withOUT compensating for the bias it introduces.
  'alpha' sets wroot before resampling such that N_effective becomes >alpha*N.
  Using alpha≈NER usually works well.

  Explanation:
  Recall that the bootstrap particle filter has "no" bias,
  but significant variance (which is reflected in the weights).
  The EnKF is quite the opposite.
  Similarly, by adjusting the weights we play on the bias-variance spectrum.
  NB: This does not mean that we make a PF-EnKF hybrid
      -- we're only playing on the weights.

  Hybridization with xN did not show much promise.
  """

  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx, Rm12 = Dyn.M, Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    w = 1/N*ones(N)

    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      if Dyn.noise.C is not 0:
        D  = randn((N,Nx))
        E += sqrt(dt*qroot)*(D@Dyn.noise.C.Right)

        if qroot != 1.0:
          # Evaluate p/q (for each col of D) when q:=p**(1/qroot).
          w *= exp(-0.5*np.sum(D**2, axis=1) * (1 - 1/qroot))
          w /= w.sum()

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)

        innovs = (yy[kObs] - Obs(E,t)) @ Rm12.T
        w      = reweight(w,innovs=innovs)

        if trigger_resampling(w, NER, [stats,E,k,kObs]):
          C12    = reg*bandw(N,Nx)*raw_C12(E,w)
          #C12  *= sqrt(rroot) # Re-include?

          wroot = 1.0
          while True:
            s   = ( w**(1/wroot - 1) ).clip(max=1e100)
            s  /= (s*w).sum()
            sw  = s*w
            if 1/(sw@sw) < N*alpha:
              wroot += 0.2
            else:
              stats.wroot[kObs] = wroot
              break
          idx,w  = resample(sw, resampl, wroot=1)


          E,chi2 = regularize(C12,E,idx,nuj)
          #if rroot != 1.0:
            # Compensate for rroot
            #w *= exp(-0.5*chi2*(1 - 1/rroot))
            #w /= w.sum()
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator




@DA_Config
def PFxN_EnKF(N,Qs,xN,re_use=True,NER=1.0,resampl='Sys',wroot_max=5,**kwargs):
  """
  Particle filter with EnKF-based proposal, q.
  Also employs xN duplication, as in PFxN.

  Recall that the proposals:
  Opt.: q_n(x) = c_n·N(x|x_n,Q     )·N(y|Hx,R)  (1)
  EnKF: q_n(x) = c_n·N(x|x_n,bar{B})·N(y|Hx,R)  (2)
  with c_n = p(y|x^{k-1}_n) being the composite proposal-analysis weight,
  and with Q possibly from regularization (rather than actual model noise).

  Here, we will use the posterior mean of (2) and cov of (1).
  Or maybe we should use x_a^n distributed according to a sqrt update?
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx, Rm12, Ri = Dyn.M, Obs.noise.C.sym_sqrt_inv, Obs.noise.C.inv

    E = X0.sample(N)
    w = 1/N*ones(N)

    DD = None
    
    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      if Dyn.noise.C is not 0:
        E += sqrt(dt)*(randn((N,Nx))@Dyn.noise.C.Right)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)
        y  = yy[kObs]
        Eo = Obs(E,t)
        wD = w.copy()

        # Importance weighting
        innovs = (y - Eo) @ Rm12.T
        w      = reweight(w,innovs=innovs)
        
        # Resampling
        if trigger_resampling(w, NER, [stats,E,k,kObs]):
          # Weighted covariance factors
          Aw = raw_C12(E,wD)
          Yw = raw_C12(Eo,wD)

          # EnKF-without-pertubations update
          if N>Nx:
            C       = Yw.T @ Yw + Obs.noise.C.full
            KG      = mrdiv(Aw.T@Yw,C)
            cntrs   = E + (y-Eo)@KG.T
            Pa      = Aw.T@Aw - KG@Yw.T@Aw
            P_cholU = funm_psd(Pa, sqrt)
            if DD is None or not re_use:
              DD    = randn((N*xN,Nx))
              chi2  = np.sum(DD**2, axis=1) * Nx/N
              log_q = -0.5 * chi2
          else:
            V,sig,UT = svd0( Yw @ Rm12.T )
            dgn      = pad0( sig**2, N ) + 1
            Pw       = (V * dgn**(-1.0)) @ V.T
            cntrs    = E + (y-Eo)@Ri@Yw.T@Pw@Aw
            P_cholU  = (V*dgn**(-0.5)).T @ Aw
            # Generate N·xN random numbers from NormDist(0,1), and compute
            # log(q(x))
            if DD is None or not re_use:
              rnk   = min(Nx,N-1)
              DD    = randn((N*xN,N))
              chi2  = np.sum(DD**2, axis=1) * rnk/N
              log_q = -0.5 * chi2
            #NB: the DoF_linalg/DoF_stoch correction is only correct "on average".
            # It is inexact "in proportion" to V@V.T-Id, where V,s,UT = tsvd(Aw).
            # Anyways, we're computing the tsvd of Aw below, so might as well
            # compute q(x) instead of q(xi).

          # Duplicate
          ED  = cntrs.repeat(xN,0)
          wD  = wD.repeat(xN) / xN

          # Sample q
          AD = DD@P_cholU
          ED = ED + AD

          # log(prior_kernel(x))
          s         = Qs*bandw(N,Nx)
          innovs_pf = AD @ tinv(s*Aw)
          # NB: Correct: innovs_pf = (ED-E_orig) @ tinv(s*Aw)
          #     But it seems to make no difference on well-tuned performance !
          log_pf    = -0.5 * np.sum(innovs_pf**2, axis=1)

          # log(likelihood(x))
          innovs = (y - Obs(ED,t)) @ Rm12.T
          log_L  = -0.5 * np.sum(innovs**2, axis=1)

          # Update weights
          log_tot = log_L + log_pf - log_q
          wD      = reweight(wD,logL=log_tot)

          # Resample and reduce
          wroot = 1.0
          while wroot < wroot_max:
            idx,w  = resample(wD, resampl, wroot=wroot, N=N)
            dups   = sum(mask_unique_of_sorted(idx))
            if dups == 0:
              E = ED[idx]
              break
            else:
              wroot += 0.1
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator




@DA_Config
def PFxN(N,Qs,xN,re_use=True,NER=1.0,resampl='Sys',wroot_max=5,**kwargs):
  """
  Idea: sample xN duplicates from each of the N kernels.
  Let resampling reduce it to N.
  Additional idea: employ w-adjustment to obtain N unique particles, without jittering.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx, Rm12 = Dyn.M, Obs.noise.C.sym_sqrt_inv

    DD = None
    E  = X0.sample(N)
    w  = 1/N*ones(N)

    stats.assess(0,E=E,w=1/N)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      if Dyn.noise.C is not 0:
        E += sqrt(dt)*(randn((N,Nx))@Dyn.noise.C.Right)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E,w=w)
        y  = yy[kObs]
        wD = w.copy()

        innovs = (y - Obs(E,t)) @ Rm12.T
        w      = reweight(w,innovs=innovs)

        if trigger_resampling(w, NER, [stats,E,k,kObs]):
          # Compute kernel colouring matrix
          cholR = Qs*bandw(N,Nx)*raw_C12(E,wD)
          cholR = chol_reduce(cholR)

          # Generate N·xN random numbers from NormDist(0,1)
          if DD is None or not re_use:
            DD = randn((N*xN,Nx))

          # Duplicate and jitter
          ED  = E.repeat(xN,0)
          wD  = wD.repeat(xN) / xN
          ED += DD[:,:len(cholR)]@cholR

          # Update weights
          innovs = (y - Obs(ED,t)) @ Rm12.T
          wD     = reweight(wD,innovs=innovs)

          # Resample and reduce
          wroot = 1.0
          while wroot < wroot_max:
            idx,w = resample(wD, resampl, wroot=wroot, N=N)
            dups  = sum(mask_unique_of_sorted(idx))
            if dups == 0:
              E = ED[idx]
              break
            else:
              wroot += 0.1
      stats.assess(k,kObs,'u',E=E,w=w)
  return assimilator



def trigger_resampling(w,NER,stat_args):
  "Return boolean: N_effective <= threshold. Also write stats."

  N_eff       = 1/(w@w)
  do_resample = N_eff <= len(w)*NER

  # Unpack stat args
  stats, E, k, kObs = stat_args

  stats.N_eff [kObs] = N_eff
  stats.resmpl[kObs] = 1 if do_resample else 0

  # Why have we put stats.assess() here?
  # Because we need to write stats.N_eff and stats.resmpl before calling assess()
  # so that the liveplotting does not eliminate these curves (as inactive).
  stats.assess(k,kObs,'a',E=E,w=w)

  return do_resample



def reweight(w,lklhd=None,logL=None,innovs=None):
  """
  Do Bayes' rule (for the empirical distribution of an importance sample).
  Do computations in log-space, for at least 2 reasons:
  - Normalization: will fail if sum==0 (if all innov's are large).
  - Num. precision: lklhd*w should have better precision in log space.
  Output is non-log, for the purpose of assessment and resampling.

  If input is 'innovs': likelihood := NormDist(innovs|0,Id).
  """
  assert all_but_1_is_None(lklhd,logL,innovs), \
      "Input error. Only specify one of lklhd, logL, innovs"

  # Get log-values.
  # Use context manager 'errstate' to not warn for log(0) = -inf.
  # Note: the case when all(w==0) will cause nan's,
  #       which should cause errors outside.
  with np.errstate(divide='ignore'):
    logw = log(w)        
    if lklhd is not None:
      logL = log(lklhd)
    elif innovs is not None:
      chi2 = np.sum(innovs**2, axis=1)
      logL = -0.5 * chi2

  logw   = logw + logL   # Bayes' rule in log-space
  logw  -= logw.max()    # Avoid numerical error
  w      = exp(logw)     # non-log
  w     /= w.sum()       # normalize
  return w

def raw_C12(E,w):
  """
  Compute the 'raw' matrix-square-root of the ensemble' covariance.
  The weights are used both for the mean and anomalies (raw sqrt).

  Note: anomalies (and thus cov) are weighted,
  and also computed based on a weighted mean.
  """
  # If weights are degenerate: use unweighted covariance to avoid C=0.
  if weight_degeneracy(w):
    w = ones(len(w))/len(w)
    # PS: 'avoid_pathological' already treated here.

  mu  = w@E
  A   = E - mu
  ub  = unbias_var(w, avoid_pathological=False)
  C12 = sqrt(ub*w[:,None]) * A
  return C12

def mask_unique_of_sorted(idx):
  "NB: returns a mask which is True at [i] iff idx[i] is NOT unique."
  duplicates  = idx==np.roll(idx,1)
  duplicates |= idx==np.roll(idx,-1)
  return duplicates

def bandw(N,M):
  """"
  Optimal bandwidth (not bandwidth^2), as per Scott's rule-of-thumb.
  Refs: [1] section 12.2.2, and [2] #Rule_of_thumb
  [1]: Doucet, de Freitas, Gordon, 2001:
    "Sequential Monte Carlo Methods in Practice"
  [2] wikipedia.org/wiki/Multivariate_kernel_density_estimation
  """
  return N**(-1/(M+4))


def regularize(C12,E,idx,no_uniq_jitter):
  """
  After resampling some of the particles will be identical.
  Therefore, if noise.is_deterministic: some noise must be added.
  This is adjusted by the regularization 'reg' factor
  (so-named because Dirac-deltas are approximated  Gaussian kernels),
  which controls the strength of the jitter.
  This causes a bias. But, as N-->∞, the reg. bandwidth-->0, i.e. bias-->0.
  Ref: [1], section 12.2.2.

  [1]: Doucet, de Freitas, Gordon, 2001:
    "Sequential Monte Carlo Methods in Practice"
  """
  # Select
  E = E[idx]

  # Jitter
  if no_uniq_jitter:
    dups         = mask_unique_of_sorted(idx)
    sample, chi2 = sample_quickly_with(C12, N=sum(dups))
    E[dups]     += sample
  else:
    sample, chi2 = sample_quickly_with(C12, N=len(E))
    E           += sample

  return E, chi2


def resample(w,kind='Systematic',N=None,wroot=1.0):
  """
  Multinomial resampling.

  - kind: 'Systematic', 'Residual' or 'Stochastic'.
    'Stochastic' corresponds to np.random.choice() or np.random.multinomial().
    'Systematic' and 'Residual' are more systematic (less stochastic)
    varaitions of 'Stochastic' sampling.
    Among the three, 'Systematic' is fastest, introduces the least noise,
    and brings continuity benefits for localized particle filters,
    and is therefore generally prefered.
    Example: see data/doc_snippets/ex_resample.py.

  - N can be different from len(w)
    (e.g. in case some particles have been elimintated).

  - wroot: Adjust weights before resampling by this root to
    promote particle diversity and mitigate thinning.
    The outcomes of the resampling are then weighted to maintain un-biased-ness.
    Ref: [3], section 3.1

  Note: (a) resampling methods are beneficial because they discard
  low-weight ("doomed") particles and reduce the variance of the weights.
  However, (b) even unbiased/rigorous resampling methods introduce noise;
  (increases the var of any empirical estimator, see [1], section 3.4).
  How to unify the seemingly contrary statements of (a) and (b) ?
  By recognizing that we're in the *sequential/dynamical* setting,
  and that *future* variance may be expected to be lower by focusing
  on the high-weight particles which we anticipate will 
  have more informative (and less variable) future likelihoods.

  [1]: Doucet, Johansen, 2009, v1.1:
    "A Tutorial on Particle Filtering and Smoothing: Fifteen years later." 
  [2]: Van Leeuwen, 2009: "Particle Filtering in Geophysical Systems"
  [3]: Liu, Chen Longvinenko, 2001:
    "A theoretical framework for sequential importance sampling with resampling"
  """

  assert(abs(w.sum()-1) < 1e-5)

  # Input parsing
  N_o = len(w)  # N _original
  if N is None: # N to sample
    N = N_o

  # Compute factors s such that s*w := w**(1/wroot). 
  if wroot!=1.0:
    s   = ( w**(1/wroot - 1) ).clip(max=1e100)
    s  /= (s*w).sum()
    sw  = s*w
  else:
    s   = ones(N_o)
    sw  = w

  # Do the actual resampling
  idx = _resample(sw,kind,N_o,N)

  w  = 1/s[idx] # compensate for above scaling by s
  w /= w.sum()  # normalize

  return idx, w


def _resample(w,kind,N_o,N):
  "Core functionality for resample(). See its docstring."
  if kind in ['Stochastic','Stoch']:
    # van Leeuwen [2] also calls this "probabilistic" resampling
    idx = np.random.choice(N_o,N,replace=True,p=w)
    # np.random.multinomial is faster (slightly different usage) ?
  elif kind in ['Residual','Res']:
    # Doucet [1] also calls this "stratified" resampling.
    w_N   = w*N             # upscale
    w_I   = w_N.astype(int) # integer part
    w_D   = w_N-w_I         # decimal part
    # Create duplicate indices for integer parts
    idx_I = [i*ones(wi,dtype=int) for i,wi in enumerate(w_I)]
    idx_I = np.concatenate(idx_I)
    # Multinomial sampling of decimal parts
    N_I   = w_I.sum() # == len(idx_I)
    N_D   = N - N_I
    idx_D = np.random.choice(N_o,N_D,replace=True,p=w_D/w_D.sum())
    # Concatenate
    idx   = np.hstack((idx_I,idx_D))
  elif kind in ['Systematic','Sys']:
    # van Leeuwen [2] also calls this "stochastic universal" resampling
    U     = rand(1) / N
    CDF_a = U + arange(N)/N
    CDF_o = np.cumsum(w)
    #idx  = CDF_a <= CDF_o[:,None]
    #idx  = np.argmax(idx,axis=0) # Finds 1st. SO/a/16244044/
    idx   = np.searchsorted(CDF_o,CDF_a)
  else:
    raise KeyError
  return idx

def sample_quickly_with(C12,N=None):
  """
  Gaussian sampling in the quickest fashion,
  which depends on the size of the colouring matrix 'C12'.
  """
  (N_,M) = C12.shape
  if N is None: N = N_
  if N_ > 2*M:
    cholR  = chol_reduce(C12)
    D      = randn((N,cholR.shape[0]))
    chi2   = np.sum(D**2, axis=1)
    sample = D@cholR
  else:
    chi2_compensate_for_rank = min(M/N_,1.0)
    D      = randn((N,N_))
    chi2   = np.sum(D**2, axis=1) * chi2_compensate_for_rank
    sample = D@C12
  return sample, chi2  




@DA_Config
def EnCheat(**kwargs):
  """A baseline/reference method.
  Should be implemented as part of Stats instead."""
  def assimilator(stats,HMM,xx,yy): pass
  return assimilator


@DA_Config
def Climatology(**kwargs):
  """
  A baseline/reference method.
  Note that the "climatology" is computed from truth, which might be
  (unfairly) advantageous if the simulation is too short (vs mixing time).
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    muC = mean(xx,0)
    AC  = xx - muC
    PC  = CovMat(AC,'A')

    stats.assess(0,mu=muC,Cov=PC)
    stats.trHK[:] = 0

    for k,kObs,_,_ in progbar(chrono.ticker):
      fau = 'u' if kObs is None else 'fau'
      stats.assess(k,kObs,fau,mu=muC,Cov=PC)
  return assimilator


@DA_Config
def OptInterp(**kwargs):
  """
  Optimal Interpolation -- a baseline/reference method.
  Uses the Kalman filter equations,
  but with a prior from the Climatology.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    # Get H.
    msg  = "For speed, only time-independent H is supported."
    H    = Obs.jacob(np.nan, np.nan)
    if not np.all(np.isfinite(H)): raise AssimFailedError(msg)

    # Compute "climatological" Kalman gain
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC) / (xx.shape[0] - 1)
    KG  = mrdiv(PC@H.T, H@PC@H.T + Obs.noise.C.full)

    # Setup scalar "time-series" covariance dynamics.
    # ONLY USED FOR DIAGNOSTICS, not to change the Kalman gain.
    Pa    = (eye(Dyn.M) - KG@H) @ PC
    CorrL = estimate_corr_length(AC.ravel(order='F'))
    WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

    # Init
    mu = muC
    stats.assess(0,mu=mu,Cov=PC)

    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      mu = Dyn(mu,t-dt,dt)
      if kObs is not None:
        stats.assess(k,kObs,'f',mu=muC,Cov=PC)
        # Analysis
        mu = muC + KG@(yy[kObs] - Obs(muC,t))
      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator


@DA_Config
def Var3D(infl=1.0,**kwargs):
  """
  3D-Var -- a baseline/reference method.
  Uses the Kalman filter equations,
  but with a prior covariance estimated from the Climatology
  and a scalar time-series approximation to the dynamics
  (that does NOT use the innovation to estimate the background covariance).
  """
  # TODO: The wave-crest yields good results for sak08, but not for boc10 
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    # Compute "climatology"
    muC = mean(xx,0)
    AC  = xx - muC
    PC  = (AC.T @ AC)/(xx.shape[0] - 1)

    # Setup scalar "time-series" covariance dynamics
    CorrL = estimate_corr_length(AC.ravel(order='F'))
    WaveC = wave_crest(0.5,CorrL) # Nevermind careless W0 init

    # Init
    mu = muC
    P  = PC
    stats.assess(0,mu=mu,Cov=P)

    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      mu = Dyn(mu,t-dt,dt)
      P  = 2*PC*WaveC(k)

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        # Analysis
        P *= infl
        H  = Obs.jacob(mu,t)
        KG = mrdiv(P@H.T, H@P@H.T + Obs.noise.C.full)
        KH = KG@H
        mu = mu + KG@(yy[kObs] - Obs(mu,t))

        # Re-calibrate wave_crest with new W0 = Pa/(2*PC).
        # Note: obs innovations are not used to estimate P!
        Pa    = (eye(Dyn.M) - KH) @ P
        WaveC = wave_crest(trace(Pa)/trace(2*PC),CorrL)

      stats.assess(k,kObs,mu=mu,Cov=2*PC*WaveC(k,kObs))
  return assimilator


@DA_Config
def Var3D_Lag(infl=1.0,**kwargs):
  """
  Background covariance based on lagged time series.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    # Get H.
    msg  = "For speed, only time-independent H is supported."
    H    = Obs.jacob(np.nan, np.nan)
    if not np.all(np.isfinite(H)): raise AssimFailedError(msg)


    # Compute "lag" Kalman gain
    muC = mean(xx,0)
    L   = chrono.dkObs
    AC  = infl * (xx[L:] - xx[:-L])
    PC  = (AC.T @ AC) / (xx.shape[0] - 1)
    KG  = mrdiv(PC@H.T, H@PC@H.T + Obs.noise.C.full)

    # Init
    mu = muC
    stats.assess(0,mu=mu,Cov=PC)

    for k,kObs,t,dt in progbar(chrono.ticker):
      # Forecast
      mu = Dyn(mu,t-dt,dt)
      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=PC)
        # Analysis
        mu = mu + KG@(yy[kObs] - Obs(mu,t))
      stats.assess(k,kObs,mu=mu,Cov=PC)
  return assimilator


def wave_crest(W0,L):
  """
  Return a sigmoid [function W(k)] that may be used
  to provide scalar approximations to covariance dynamics. 

  We use the logistic function for the sigmoid.
  This has theoretical benefits: it's the solution of the
  "population growth" ODE: dE/dt = a*E*(1-E/E(∞)).
  PS: It might be better to use the "error growth ODE" of Lorenz/Dalcher/Kalnay,
  but this has a significantly more complicated closed-form solution,
  and reduces to the above ODE when there's no model error (ODE source term).

  As "any sigmoid", W is symmetric around 0 and W(-∞)=0 and W(∞)=1.
  It is further fitted such that
   - W has a scale (slope at 0) equal to that of f(t) = 1-exp(-t/L),
     where L is suppsed to be the system's decorrelation length (L),
     as detailed in doc/wave_crest.jpg.
   - W has a shift (translation) such that it takes the value W0 at k=0.
     Typically, W0 = 2*PC/Pa, where
     2*PC = 2*Var(free_run) = Var(mu-truth), and Pa = Var(analysis).

  The best way to illustrate W(k) and test it is to:
   - set dkObs very large, to see a long evolution;
   - set store_u = True, to store intermediate stats;
   - plot_time_series (var vs err).
  """
  sigmoid = lambda t: 1/(1+exp(-t))
  inv_sig = lambda s: log(s/(1-s))
  shift   = inv_sig(W0) # "reset" point
  scale   = 1/(2*L)     # derivative of exp(-t/L) at t=1/2

  def W(k,reset=False):
    # Manage intra-DAW counter, dk.
    if reset is None:
      dk = k - W.prev_obs
    elif reset > 0:
      W.prev_obs = k
      dk = k - W.prev_obs # = 0
    else:
      dk = k - W.prev_obs
    # Compute
    return sigmoid(shift + scale*dk)

  # Initialize W.prev_obs: provides persistent ref [for W(k)] to compute dk.
  W.prev_obs = (shift-inv_sig(0.5))/scale 

  return W


# TODO: Clean up
@DA_Config
def ExtRTS(infl=1.0,**kwargs):
  """
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Nx = Dyn.M

    R  = Obs.noise.C.full
    Q  = 0 if Dyn.noise.C==0 else Dyn.noise.C.full

    mu    = zeros((chrono.K+1,Nx))
    P     = zeros((chrono.K+1,Nx,Nx))

    # Forecasted values
    muf   = zeros((chrono.K+1,Nx))
    Pf    = zeros((chrono.K+1,Nx,Nx))
    Ff    = zeros((chrono.K+1,Nx,Nx))

    mu[0] = X0.mu
    P [0] = X0.C.full

    stats.assess(0,mu=mu[0],Cov=P[0])

    # Forward pass
    for k,kObs,t,dt in progbar(chrono.ticker, 'ExtRTS->'):
      mu[k]  = Dyn(mu[k-1],t-dt,dt)
      F      = Dyn.jacob(mu[k-1],t-dt,dt) 
      P [k]  = infl**(dt)*(F@P[k-1]@F.T) + dt*Q

      # Store forecast and Jacobian
      muf[k] = mu[k]
      Pf [k] = P [k]
      Ff [k] = F

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu[k],Cov=P[k])
        H     = Obs.jacob(mu[k],t)
        KG    = mrdiv(P[k] @ H.T, H@P[k]@H.T + R)
        y     = yy[kObs]
        mu[k] = mu[k] + KG@(y - Obs(mu[k],t))
        KH    = KG@H
        P[k]  = (eye(Nx) - KH) @ P[k]
        stats.assess(k,kObs,'a',mu=mu[k],Cov=P[k])

    # Backward pass
    for k in progbar(range(chrono.K)[::-1],'ExtRTS<-'):
      J     = mrdiv(P[k]@Ff[k+1].T, Pf[k+1])
      mu[k] = mu[k]  + J @ (mu[k+1]  - muf[k+1])
      P[k]  = P[k] + J @ (P[k+1] - Pf[k+1]) @ J.T
    for k in progbar(range(chrono.K+1),desc='Assess'):
      stats.assess(k,mu=mu[k],Cov=P[k])

  return assimilator


@DA_Config
def ExtKF(infl=1.0,**kwargs):
  """
  The extended Kalman filter.
  A baseline/reference method.

  If everything is linear-Gaussian, this provides the exact solution
  to the Bayesian filtering equations.

  - infl (inflation) may be specified.
    Default: 1.0 (i.e. none), as is optimal in the lin-Gauss case.
    Gets applied at each dt, with infl_per_dt := inlf**(dt), so that 
    infl_per_unit_time == infl.
    Specifying it this way (per unit time) means less tuning.
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    R  = Obs.noise.C.full
    Q  = 0 if Dyn.noise.C==0 else Dyn.noise.C.full

    mu = X0.mu
    P  = X0.C.full

    stats.assess(0,mu=mu,Cov=P)

    for k,kObs,t,dt in progbar(chrono.ticker):
      
      mu = Dyn(mu,t-dt,dt)
      F  = Dyn.jacob(mu,t-dt,dt) 
      P  = infl**(dt)*(F@P@F.T) + dt*Q

      # Of academic interest? Higher-order linearization:
      # mu_i += 0.5 * (Hessian[f_i] * P).sum()

      if kObs is not None:
        stats.assess(k,kObs,'f',mu=mu,Cov=P)
        H  = Obs.jacob(mu,t)
        KG = mrdiv(P @ H.T, H@P@H.T + R)
        y  = yy[kObs]
        mu = mu + KG@(y - Obs(mu,t))
        KH = KG@H
        P  = (eye(Dyn.M) - KH) @ P

        stats.trHK[kObs] = trace(KH)/Dyn.M

      stats.assess(k,kObs,mu=mu,Cov=P)
  return assimilator



@DA_Config
def RHF(N,ordr='rand',infl=1.0,rot=False,**kwargs):
  """
  Rank histogram filter. 

  Quick & dirty implementation without attention to (de)tails.

  Ref:
  Anderson'2010: "A Non-Gaussian Ensemble Filter Update for DA"

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz63/anderson2010non.py 
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0

    N1         = N-1
    step       = 1/N
    cdf_grid   = linspace(step/2, 1-step/2, N)

    R    = Obs.noise
    Rm12 = Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        y    = yy[kObs]
        inds = serial_inds(ordr, y, R, center(E)[0])
            
        for i,j in enumerate(inds):
          Eo = Obs(E,t)
          xo = mean(Eo,0)
          Y  = Eo - xo
          mu = mean(E ,0)
          A  = E-mu

          # Update j-th component of observed ensemble
          dYf    = Rm12[j,:] @ (y - Eo).T # NB: does Rm12 make sense?
          Yj     = Rm12[j,:] @ Y.T
          Regr   = A.T@Yj/np.sum(Yj**2)

          Sorted = np.argsort(dYf)
          Revert = np.argsort(Sorted)
          dYf    = dYf[Sorted]
          w      = reweight(ones(N),innovs=dYf[:,None]) # Lklhd
          w      = w.clip(1e-10) # Avoid zeros in interp1
          cw     = w.cumsum()
          cw    /= cw[-1]
          cw    *= N1/N
          cdfs   = np.minimum(np.maximum(cw[0],cdf_grid),cw[-1])
          dhE    = -dYf + np.interp(cdfs, cw, dYf)
          dhE    = dhE[Revert]
          # Update state by regression
          E     += np.outer(-dhE, Regr)

        E = post_process(E,infl,rot)

      stats.assess(k,kObs,E=E)
  return assimilator




@DA_Config
def LNETF(N,loc_rad,taper='GC',infl=1.0,Rs=1.0,rot=False,**kwargs):
  """
  The Nonlinear-Ensemble-Transform-Filter (localized).

  Ref: Julian Tödter and Bodo Ahrens (2014):
  "A Second-Order Exact Ensemble Square Root Filter for Nonlinear Data Assimilation"

  It is (supposedly) a deterministic upgrade of the NLEAF of Lei and Bickel (2011).

  Settings for reproducing literature benchmarks may be found in
  mods/Lorenz95/tod15.py
  mods/Lorenz95/wiljes2017.py
  """
  def assimilator(stats,HMM,xx,yy):
    Dyn,Obs,chrono,X0 = HMM.Dyn, HMM.Obs, HMM.t, HMM.X0
    Rm12 = Obs.noise.C.sym_sqrt_inv

    E = X0.sample(N)
    stats.assess(0,E=E)

    for k,kObs,t,dt in progbar(chrono.ticker):
      E = Dyn(E,t-dt,dt)
      E = add_noise(E, dt, Dyn.noise, kwargs)

      if kObs is not None:
        stats.assess(k,kObs,'f',E=E)
        mu = mean(E,0)
        A  = E - mu

        Eo = Obs(E,t)
        xo = mean(Eo,0)
        YR = (Eo-xo)  @ Rm12.T
        yR = (yy[kObs] - xo) @ Rm12.T

        state_batches, obs_taperer = Obs.localizer(loc_rad, 'x2y', t, taper)
        for ii in state_batches:
          # Localize obs
          jj, tapering = obs_taperer(ii)
          if len(jj) == 0: return

          Y_jj  = YR[:,jj] * sqrt(tapering)
          dy_jj = yR[jj]   * sqrt(tapering)

          # NETF:
          # This "paragraph" is the only difference to the LETKF.
          innovs = (dy_jj-Y_jj)/Rs
          if 'laplace' in str(type(Obs.noise)).lower():
            w    = laplace_lklhd(innovs)
          else: # assume Gaussian
            w    = reweight(ones(N),innovs=innovs)
          dmu    = w@A[:,ii]
          AT     = sqrt(N)*funm_psd(diag(w) - np.outer(w,w), sqrt)@A[:,ii]

          E[:,ii] = mu[ii] + dmu + AT

        E = post_process(E,infl,rot)
      stats.assess(k,kObs,E=E)
  return assimilator

def laplace_lklhd(xx):
  """
  Compute likelihood of xx wrt. the sampling distribution
  LaplaceParallelRV(C=I), i.e., for x in xx:
  p(x) = exp(-sqrt(2)*|x|_1) / sqrt(2).
  """
  logw   = -sqrt(2)*np.sum(np.abs(xx), axis=1)
  logw  -= logw.max()    # Avoid numerical error
  w      = exp(logw)     # non-log
  w     /= w.sum()       # normalize
  return w



