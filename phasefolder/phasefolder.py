"""Main module."""
from scipy.signal import medfilt
import math

def find_period(lc):
    """Calculates a period estimate for the given light curve
    
    Description:
        This method uses the maximum of the Lomb-Scargle periodogram
        as an initial guess for the period, then tests values between
        70% to 130% of that guess. 
        calculating the residual for each
        and determining the best guess based on the minimum standard
        deviation.
        
        Finally, this best guess is doubled to check if the best guess
        could be off by a factor of two.
    
    Args:
        lc (lightkurve.LightCurve): a lightkurve.LightCurve object
    Returns:
        best_guess (float): the best period centered around the lomb-sca
    """
    lc = lc[lc.quality == 0]

    pg = lc.normalize(unit='ppm').to_periodogram(minimum_period = 0.042,
                                                 method="BoxLeastSquares")
    period = pg.period_at_max_power  # first guess
    # pg1 = lc.normalize(unit='ppm').to_periodogram(maximum_period = 2.1*period.value,
    # oversample_factor=100)

    best_guess = redef(lc, period)
    return best_guess


def redef(lc, period):
    '''
    Iterates within the range mini to maxi to find the best period
    
        Parameters:
            mini (float): current minimum period for binary search
            maxi (float): current maximum period for binary search
            midpt (float): current assumed period value
            lc (lightkurve.LightCurve): a lightkurve.LightCurve object
        
        Returns: 
            midpt (double): the best period to phase fold on
    '''
    #tests if a multiple of the midpt is better than the current one
    midpt = period.value
    mini = max(.7*period.value, 0.042)
    maxi = min(1.3*period.value, 15)
    #global mini, maxi, midpt, lc
    while(mini + 0.0001 < maxi):
        # kind of an optimization strategy,
        # check if it's better lower or higher
        # then go that way until the difference is nominal.
        
        min_rsd   = calc_residual_stdev(lc, (mini+midpt)/2)
        midpt_rsd = calc_residual_stdev(lc, midpt)
        max_rsd   = calc_residual_stdev(lc, (maxi+midpt)/2)
        
        if min(midpt_rsd, min_rsd, max_rsd) == midpt_rsd:
            break
            
        elif (min_rsd) < (max_rsd):
            maxi  = midpt
            midpt = (mini + midpt)/2
            
        else:
            mini  = midpt
            midpt = (midpt + maxi)/2
            
    if calc_residual_stdev(lc, 2*midpt) < midpt_rsd:
        midpt = 2*midpt
        
    elif calc_residual_stdev(lc, midpt/2) < midpt_rsd:
        midpt = midpt/2
    
    return midpt


def calc_residual_stdev(lc, per):
    """calculates the std dev of residuals of period given by argument (num)
    """

    cleanlightcurve = lc[lc.quality == 0].fold(per)
    compflux        = medfilt(cleanlightcurve.flux, kernel_size=13)
    residual        = compflux - cleanlightcurve.flux
    ressqr          = sum(residual**2)

    rsd = math.sqrt((ressqr)/(len(cleanlightcurve)-2))
    return rsd
