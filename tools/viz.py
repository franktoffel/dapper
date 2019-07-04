from common import *

#from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import juggle_axes

from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from matplotlib import colors
from matplotlib.ticker import MaxNLocator


def setup_wrapping(M,periodic=True):
  """
  Make periodic indices and a corresponding function
  (that works for ensemble input).
  """

  if periodic:
    ii = np.hstack([-0.5, arange(M), M-0.5])
    def wrap(E):
      midpoint = (E[[0],...] + E[[-1],...])/2
      return ccat(midpoint,E,midpoint)

  else:
    ii = arange(M)
    wrap = lambda x: x

  return ii, wrap
  
def adjust_position(ax,adjust_extent=False,**kwargs):
  """
  Adjust values (add) to get_position().
  kwarg must be one of 'x0','y0','width','height'.
  """
  # Load get_position into d
  pos = ax.get_position()
  d   = OrderedDict()
  for key in ['x0','y0','width','height']:
    d[key] = getattr(pos,key)
  # Make adjustments
  for key,item in kwargs.items():
    d[key] += item
    if adjust_extent:
      if key=='x0': d['width']  -= item
      if key=='y0': d['height'] -= item
  # Set
  ax.set_position(d.values())

def span(xx,axis=None):
  a = xx.min(axis)
  b = xx.max(axis)
  return a, b

def stretch(a,b,factor=1,int=False):
  """
  Stretch distance a-b by factor.
  Return a,b.
  If int: floor(a) and ceil(b)
  """
  c = (a+b)/2
  a = c + factor*(a-c) 
  b = c + factor*(b-c) 
  if int:
    a = floor(a)
    b = ceil(b)
  return a, b


def set_ilim(ax,i,Min=None,Max=None):
  """Set bounds on axis i.""" 
  if i is 0: ax.set_xlim(Min,Max)
  if i is 1: ax.set_ylim(Min,Max)
  if i is 2: ax.set_zlim(Min,Max)

# Examples:
# K_lag = estimate_good_plot_length(stats.xx,chrono,mult = 80)
def estimate_good_plot_length(xx,chrono=None,mult=100):
  """
  Estimate good length for plotting stuff
  from the time scale of the system.
  Provide sensible fall-backs (better if chrono is supplied).
  """
  if xx.ndim == 2:
    # If mult-dim, then average over dims (by ravel)....
    # But for inhomogeneous variables, it is important
    # to subtract the mean first!
    xx = xx - mean(xx,axis=0)
    xx = xx.ravel(order='F')

  try:
    K = mult * estimate_corr_length(xx)
  except ValueError:
    K = 0

  if chrono != None:
    t = chrono
    K = int(min(max(K, t.dkObs), t.K))
    T = round2sigfig(t.tt[K],2) # Could return T; T>tt[-1]
    K = find_1st_ind(t.tt >= T)
    if K: return K
    else: return t.K
  else:
    K = int(min(max(K, 1), len(xx)))
    T = round2sigfig(K,2)
    return K

def get_plot_inds(xx,chrono,K=None,T=None,**kwargs):
  """
  Def subset of kk for plotting, from one of
   - K
   - T
   - mult * auto-correlation length of xx
  """
  t = chrono
  if K is None:
    if T: K = find_1st_ind(t.tt >= min(T,t.T))
    else: K = estimate_good_plot_length(xx,chrono=t,**kwargs)
  plot_kk    = t.kk[:K+1]
  plot_kkObs = t.kkObs[t.kkObs<=K]
  return plot_kk, plot_kkObs


def plot_hovmoller(xx,chrono=None,**kwargs):
  """
  Plot Hovmöller diagram.
  kwargs forwarded to get_plot_inds().
  """
  #cm = mpl.colors.ListedColormap(sns.color_palette("BrBG", 256)) # RdBu_r
  #cm = plt.get_cmap('BrBG')
  fig, ax = plt.subplots(num=16,figsize=(4,3.5))
  set_figpos('3311 mac')

  Nx = xx.shape[1]

  if chrono!=None:
    kk,_ = get_plot_inds(xx,chrono,mult=40,**kwargs)
    tt   = chrono.tt[kk]
    ax.set_ylabel('Time (t)')
  else:
    K    = estimate_good_plot_length(xx,mult=40)
    tt   = arange(K)
    ax.set_ylabel('Time indices (k)')

  plt.contourf(arange(Nx),tt,xx[kk],25)
  plt.colorbar()
  ax.set_position([0.125, 0.20, 0.62, 0.70])
  ax.set_title("Hovmoller diagram (of 'Truth')")
  ax.set_xlabel('Dimension index (i)')
  add_endpoint_xtick(ax)


def add_endpoint_xtick(ax):
  """Useful when xlim(right) is e.g. 39 (instead of 40)."""
  xF = ax.get_xlim()[1]
  ticks = ax.get_xticks()
  if ticks[-1] > xF:
    ticks = ticks[:-1]
  ticks = np.append(ticks, xF)
  ax.set_xticks(ticks)


def integer_hist(E,N,centrd=False,weights=None,**kwargs):
  """Histogram for integers."""
  ax = plt.gca()
  rnge = (-0.5,N+0.5) if centrd else (0,N+1)
  ax.hist(E,bins=N+1,range=rnge,normed=1,weights=weights,**kwargs)
  ax.set_xlim(rnge)


def not_available_text(ax,txt=None,fs=20):
  if txt is None: txt = '[Not available]'
  else:           txt = '[' + txt + ']'
  ax.text(0.5,0.5,txt,
      fontsize=fs,
      transform=ax.transAxes,
      va='center',ha='center',
      wrap=True)

def plot_err_components(stats):
  """
  Plot components of the error.
  Note: it was chosen to plot(ii, mean_in_time(abs(err_i))),
        and thus the corresponding spread measure is MAD.
        If one chose instead: plot(ii, std_in_time(err_i)),
        then the corresponding measure of spread would have been std.
        This choice was made in part because (wrt. subplot 2)
        the singular values (svals) correspond to rotated MADs,
        and because rms(umisf) seems to convoluted for interpretation.
  """
  fgE = plt.figure(15,figsize=(6,6)).clf()
  set_figpos('1312 mac')

  chrono = stats.HMM.t
  Nx     = stats.xx.shape[1]

  err   = mean( abs(stats.err  .a) ,0)
  sprd  = mean(     stats.mad  .a  ,0)
  umsft = mean( abs(stats.umisf.a) ,0)
  usprd = mean(     stats.svals.a  ,0)

  ax_r = plt.subplot(311)
  ax_r.plot(          arange(Nx),               err,'k',lw=2, label='Error')
  if Nx<10**3:
    ax_r.fill_between(arange(Nx),[0]*len(sprd),sprd,alpha=0.7,label='Spread')
  else:
    ax_r.plot(        arange(Nx),              sprd,alpha=0.7,label='Spread')
  #ax_r.set_yscale('log')
  ax_r.set_title('Element-wise error comparison')
  ax_r.set_xlabel('Dimension index (i)')
  ax_r.set_ylabel('Time-average (_a) magnitude')
  ax_r.set_ylim(bottom=mean(sprd)/10)
  ax_r.set_xlim(right=Nx-1); add_endpoint_xtick(ax_r)
  ax_r.get_xaxis().set_major_locator(MaxNLocator(integer=True))
  plt.subplots_adjust(hspace=0.55) # OR: [0.125,0.6, 0.78, 0.34]
  ax_r.legend()

  ax_s = plt.subplot(312)
  ax_s.set_xlabel('Principal component index')
  ax_s.set_ylabel('Time-average (_a) magnitude')
  ax_s.set_title('Spectral error comparison')
  has_been_computed = np.any(np.isfinite(umsft))
  if has_been_computed:
    L = len(umsft)
    ax_s.plot(        arange(L),      umsft,'k',lw=2, label='Error')
    ax_s.fill_between(arange(L),[0]*L,usprd,alpha=0.7,label='Spread')
    ax_s.set_yscale('log')
    ax_s.set_ylim(bottom=1e-4*usprd.sum())
    ax_s.set_xlim(right=Nx-1); add_endpoint_xtick(ax_s)
    ax_s.get_xaxis().set_major_locator(MaxNLocator(integer=True))
    ax_s.legend()
  else:
    not_available_text(ax_s)

  rmse = stats.rmse.a[chrono.maskObs_BI]
  ax_R = plt.subplot(313)
  ax_R.hist(rmse,bins=30,normed=0)
  ax_R.set_ylabel('Num. of occurence (_a)')
  ax_R.set_xlabel('RMSE')
  ax_R.set_title('Histogram of RMSE values')


def plot_rank_histogram(stats):
  chrono = stats.HMM.t

  has_been_computed = \
      hasattr(stats,'rh') and \
      not all(stats.rh.a[-1]==array(np.nan).astype(int))

  fig, ax = freshfig(24, (6,3))
  set_figpos('3331 mac')
  ax.set_title('(Mean of marginal) rank histogram (_a)')
  ax.set_ylabel('Freq. of occurence\n (of truth in interval n)')
  ax.set_xlabel('ensemble member index (n)')
  adjust_position(ax, y0=0.05, x0=0.05, adjust_extent=True)

  if has_been_computed:
    ranks = stats.rh.a[chrono.maskObs_BI]
    Nx    = ranks.shape[1]
    N     = stats.config.N
    if not hasattr(stats,'w'):
      # Ensemble rank histogram
      integer_hist(ranks.ravel(),N)
    else:
      # Experimental: weighted rank histogram.
      # Weight ranks by inverse of particle weight. Why? Coz, with correct
      # importance weights, the "expected value" histogram is then flat.
      # Potential improvement: interpolate weights between particles.
      w  = stats.w.a[chrono.maskObs_BI]
      K  = len(w)
      w  = np.hstack([w, ones((K,1))/N]) # define weights for rank N+1
      w  = array([ w[arange(K),ranks[arange(K),i]] for i in range(Nx)])
      w  = w.T.ravel()
      w  = np.maximum(w, 1/N/100) # Artificial cap. Reduces variance, but introduces bias.
      w  = 1/w
      integer_hist(ranks.ravel(),N,weights=w)
  else:
    not_available_text(ax)
  

def adjustable_box_or_forced():
  "For set_aspect(), adjustable='box-forced' replaced by 'box' since mpl 2.2.0."
  from pkg_resources import parse_version as pv
  return 'box-forced' if pv(mpl.__version__) < pv("2.2.0") else 'box'


def freshfig(num,figsize=None,*args,**kwargs):
  """Create/clear figure.
  - If the figure does not exist: create figure it.
    This allows for figure sizing -- even on Macs.
  - Otherwise: clear figure (we avoid closing/opening so as
    to keep (potentially manually set) figure pos and size.
  - The rest is the same as:
    >>> fig, ax = suplots()
  """
  fig = plt.figure(num=num,figsize=figsize)
  fig.clf()
  _, ax = plt.subplots(num=fig.number,*args,**kwargs)
  return fig, ax

def show_figs(fignums=None):
  """Move all fig windows to top"""
  if fignums == None:
    fignums = plt.get_fignums()
  try:
    fignums = list(fignums)
  except:
    fignums = [fignums]
  for f in fignums:
    plt.figure(f)
    fmw = plt.get_current_fig_manager().window
    fmw.attributes('-topmost',1) # Bring to front, but
    fmw.attributes('-topmost',0) # don't keep in front

def set_figpos(loc):
  """
  Place figure on screen, where 'loc' can be either
    NW, E, ...
  or
    4 digits (as str or int) to define grid M,N,i,j.
  """

  #Only works with both:
   #- Patrick's monitor setup (Dell with Mac central-below)
   #- TkAgg backend. (Previously: Qt4Agg)
  if not user_is_patrick or mpl.get_backend() != 'TkAgg':
    return
  fmw = plt.get_current_fig_manager().window

  loc = str(loc)

  # Qt4Agg only:
  #  # Current values 
  #  w_now = fmw.width()
  #  h_now = fmw.height()
  #  x_now = fmw.x()
  #  y_now = fmw.y()
  #  # Constants 
  #  Dell_w = 2560
  #  Dell_h = 1440
  #  Mac_w  = 2560
  #  Mac_h  = 1600
  #  # Why is Mac monitor scaled by 1/2 ?
  #  Mac_w  /= 2
  #  Mac_h  /= 2
  # Append the string 'mac' to place on mac monitor.
  #  if 'mac' in loc:
  #    x0 = Dell_w/4
  #    y0 = Dell_h+44
  #    w0 = Mac_w
  #    h0 = Mac_h-44
  #  else:
  #    x0 = 0
  #    y0 = 0
  #    w0 = Dell_w
  #    h0 = Dell_h

  # TkAgg
  x0 = 0
  y0 = 0
  w0 = 1280
  h0 = 752
  
  # Def place function with offsets
  def place(x,y,w,h):
    #fmw.setGeometry(x0+x,y0+y,w,h) # For Qt4Agg
    geo = str(int(w)) + 'x' + str(int(h)) + \
        '+' + str(int(x)) + '+' + str(int(y))
    fmw.geometry(newGeometry=geo) # For TkAgg

  if not loc[:4].isnumeric():
    if   loc.startswith('NW'): loc = '2211'
    elif loc.startswith('SW'): loc = '2221'
    elif loc.startswith('NE'): loc = '2212'
    elif loc.startswith('SE'): loc = '2222'
    elif loc.startswith('W' ): loc = '1211'
    elif loc.startswith('E' ): loc = '1212'
    elif loc.startswith('S' ): loc = '2121'
    elif loc.startswith('N' ): loc = '2111'

  # Place
  M,N,i,j = [int(x) for x in loc[:4]]
  assert M>=i>0 and N>=j>0
  h0   -= (M-1)*25
  yoff  = 25*(i-1)
  if i>1:
    yoff += 25
  place((j-1)*w0/N, yoff + (i-1)*h0/M, w0/N, h0/M)


# stackoverflow.com/a/7396313
from matplotlib import transforms as mtransforms
def autoscale_based_on(ax, line_handles):
  "Autoscale axis based (only) on line_handles."
  ax.dataLim = mtransforms.Bbox.unit()
  for iL,lh in enumerate(line_handles):
    xy = np.vstack(lh.get_data()).T
    ax.dataLim.update_from_data_xy(xy, ignore=(iL==0))
  ax.autoscale_view()


from matplotlib.widgets import CheckButtons
import textwrap
def toggle_lines(ax=None,autoscl=True,numbering=False,txtwidth=15,txtsize=None,state=None):
  """
  Make checkbuttons to toggle visibility of each line in current plot.
  autoscl  : Rescale axis limits as required by currently visible lines.
  numbering: Add numbering to labels.
  txtwidth : Wrap labels to this length.

  State of checkboxes can be inquired by 
  OnOff = [lh.get_visible() for lh in ax.findobj(lambda x: isinstance(x,mpl.lines.Line2D))[::2]]
  """

  if ax is None: ax = plt.gca()
  if txtsize is None: txtsize = mpl.rcParams['font.size']

  # Get lines and their properties
  lines = {'handle': list(ax.get_lines())}
  for prop in ['label','color','visible']:
    lines[prop] = [plt.getp(x,prop) for x in lines['handle']]
  # Put into pandas for some reason
  lines = pd.DataFrame(lines)
  # Rm those that start with _
  lines = lines[~lines.label.str.startswith('_')]

  # Adjust labels
  if numbering: lines['label'] = [str(i)+': '+lbl for i,lbl in enumerate(lines['label'])]
  if txtwidth:  lines['label'] = [textwrap.fill(lbl,width=txtwidth) for lbl in lines['label']]

  # Set state. BUGGY? sometimes causes MPL complaints after clicking boxes
  if state is not None:
    state = array(state).astype(bool)
    lines.visible = state
    for i,x in enumerate(state):
      lines['handle'][i].set_visible(x)

  # Setup buttons
  # When there's many, the box-sizing is awful, but difficult to fix.
  W       = 0.23 * txtwidth/15 * txtsize/10
  N       = len(lines)
  nBreaks = sum(lbl.count('\n') for lbl in lines['label']) # count linebreaks
  H       = min(1,0.05*(N+nBreaks))
  plt.subplots_adjust(left=W+0.12,right=0.97)
  rax = plt.axes([0.05, 0.5-H/2, W, H])
  check = CheckButtons(rax, lines.label, lines.visible)

  # Adjust button style
  for i in range(N):
    check.rectangles[i].set(lw=0,facecolor=lines.color[i])
    check.labels[i].set(color=lines.color[i])
    if txtsize: check.labels[i].set(size=txtsize)

  # Callback
  def toggle_visible(label):
    ind = lines.label==label
    handle = lines[ind].handle.item()
    vs = not lines[ind].visible.item()
    handle.set_visible( vs )
    lines.loc[ind,'visible'] = vs
    if autoscl:
      autoscale_based_on(ax,lines[lines.visible].handle)
    plt.draw()
  check.on_clicked(toggle_visible)

  # Return focus
  plt.sca(ax)

  # Must return (and be received) so as not to expire.
  return check


def toggle_viz(*handles,prompt=False,legend=False,pause=True):
  """Toggle visibility of the graphics with handle handles."""

  are_viz = []
  for h in handles:

    # Core functionality: turn on/off
    is_viz = not h.get_visible()
    h.set_visible(is_viz)
    are_viz += [is_viz]

    # Legend updating. Basic version: works by
    #  - setting line's label to actual_label/'_nolegend_' if is_viz/not
    #  - re-calling legend()
    if legend:
        if is_viz:
          try:
            h.set_label(h.actual_label)
          except AttributeError:
            pass
        else:
          h.actual_label = h.get_label()
          h.set_label('_nolegend_')
        # Legend refresh
        ax = h.axes
        with warnings.catch_warnings():
          warnings.simplefilter("error",category=UserWarning)
          try:
            ax.legend()
          except UserWarning:
            # If all labels are '_nolabel_' then ax.legend() throws warning,
            # and quits before refreshing. => Refresh by creating/rm another legend.
            ax.legend('TMP').remove()

  if prompt: input("Press <Enter> to continue...")
  if pause:  plt.pause(0.02)

  return are_viz


def savefig_n(f=None, ext='.pdf'):
  """
  Simplify the exporting of a figure, especially when it's part of a series.
  """
  assert savefig_n.index>=0, "Initalize using savefig_n.index = 1 in your script"
  if f is None:
    f = inspect.getfile(inspect.stack()[1][0])   # Get __file__ of caller
    f = save_dir(f)                              # Prep save dir
  f = f + str(savefig_n.index) + ext             # Compose name
  print("Saving fig to:",f)                      # Print
  plt.savefig(f)                                 # Save
  savefig_n.index += 1                           # Increment index
  plt.pause(0.1)                                 # For safety?
savefig_n.index = -1



def nrowcol(nTotal,AR=1):
  "Return integer nrows and ncols such that nTotal ≈ nrows*ncols."
  nrows = int(floor(sqrt(nTotal)/AR))
  ncols = int(ceil(nTotal/nrows))
  return nrows, ncols


from matplotlib.gridspec import GridSpec
def axes_with_marginals(n_joint, n_marg,**kwargs):
  """
  Create a joint axis along with two marginal axes.

  Example:
  >>> ax_s, ax_x, ax_y = axes_with_marginals(4, 1)
  >>> x, y = np.random.randn(2,500)
  >>> ax_s.scatter(x,y)
  >>> ax_x.hist(x)
  >>> ax_y.hist(y,orientation="horizontal")
  """

  N = n_joint + n_marg

  # Method 1
  #fig, ((ax_s, ax_y), (ax_x, _)) = plt.subplots(2,2,num=plt.gcf().number,
      #sharex='col',sharey='row',gridspec_kw={
        #'height_ratios':[n_joint,n_marg],
        #'width_ratios' :[n_joint,n_marg]})
  #_.set_visible(False) # Actually removing would bug the axis ticks etc.
  
  # Method 2
  gs   = GridSpec(N,N,**kwargs)
  fig  = plt.gcf()
  ax_s = fig.add_subplot(gs[n_marg:N     ,0      :n_joint])
  ax_x = fig.add_subplot(gs[0     :n_marg,0      :n_joint],sharex=ax_s)
  ax_y = fig.add_subplot(gs[n_marg:N     ,n_joint:N      ],sharey=ax_s)
  # Cannot delete ticks coz axis are shared
  plt.setp(ax_x.get_xticklabels(), visible=False)
  plt.setp(ax_y.get_yticklabels(), visible=False)

  return ax_s, ax_x, ax_y

from matplotlib.patches import Ellipse
def cov_ellipse(ax, mu, sigma, **kwargs):
    """
    Draw ellipse corresponding to (Gaussian) 1-sigma countour of cov matrix.

    Inspired by stackoverflow.com/q/17952171

    Example:
    >>> ellipse = cov_ellipse(ax, y, R,
    >>>           facecolor='none', edgecolor='y',lw=4,label='$1\\sigma$')
    """

    # Cov --> Width, Height, Theta
    vals, vecs = np.linalg.eigh(sigma)
    x, y       = vecs[:, -1] # x-y components of largest (last) eigenvector
    theta      = np.degrees(np.arctan2(y, x))
    theta      = theta % 180

    h, w       = 2 * np.sqrt(vals.clip(0))

    # Get artist
    e = Ellipse(mu, w, h, theta, **kwargs)

    ax.add_patch(e)
    e.set_clip_box(ax.bbox) # why is this necessary?

    # Return artist
    return e
    



