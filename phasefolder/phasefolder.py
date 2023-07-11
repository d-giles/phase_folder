"""Main module."""
from ipywidgets import interactive, widgets
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import math
import pandas as pd
import os
import lightkurve as lk

def find_period(lc, method="BoxLeastSquares"):
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
    if method=="BoxLeastSquares":
        pg = lc.normalize(unit='ppm').to_periodogram(method="BoxLeastSquares")
    else:
        pg = lc.normalize(unit='ppm').to_periodogram(
            minimum_period = 0.042,
            oversample_factor=300
        )
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


def csv_to_lc(file):
    lcdata = pd.read_csv(file)
    lc = lk.LightCurve(time=lcdata.time, 
                       flux=lcdata.flux,
                       flux_err=lcdata.flux_err,
                       quality=lcdata.quality,
                       label=file.split("/")[-1].split(".csv")[0])
    return lc


def example_lcc():
    return [csv_to_lc("lightcurves/"+file) for file in os.listdir("lightcurves")]

def notebook_interact(lcc, method="BoxLeastSquares"):
    # elements
    ## index in lcc
    a = widgets.BoundedIntText(
        value=0,
        min=0,
        max=len(lcc)-1,
        step=1,
        description='Index:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    ## period to fold on
    b = widgets.BoundedFloatText(
        value=5,
        min=0.1,
        step=0.005,
        description='Period:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    ## halve the period
    half_button = widgets.Button(
        description=f"period /2",
        disabled=False,
        button_style='',
        tooltip="period/2",
    )
    half_button.factor = 0.5  # giving these buttons a new attribute 'factor'
    ## double the period
    double_button = widgets.Button(
        description=f"period x2",
        disabled=False,
        button_style='',
        tooltip="2*period",
    )
    double_button.factor = 2
    ## triple the period
    triple_button = widgets.Button(
        description=f"period x3",
        disabled=False,
        button_style='',
        tooltip="3*period",
    )
    triple_button.factor = 3
    ## change the center of the phase folded plot
    time_min = lcc[0].time[lcc[0].flux==lcc[0].flux.min()].value[0]
    c = widgets.BoundedFloatText(
        value=time_min,
        min=min(lcc[a.value].time.value),
        max=max(lcc[a.value].time.value),
        step=1,
        description='Epoch Time:',
        disabled=False,
        layout=widgets.Layout(width='50%')
    )
    ## save the plot and the folded light curve data
    save = widgets.Button(
        description='Save',
        disabled=False,
        button_style='', # 'success', 'info', 'warning', 'danger' or ''
        tooltip='Save',
    )

    # behaviors
    def f(index, period, center):
        """Updates plot on index, period, or center point changes
        """
        lc = lcc[index]
        folded_lc = lc.fold(period, normalize_phase=False, epoch_time=center)

        fig, ax = plt.subplots(2, figsize=(10,4), tight_layout=True)
        lc.scatter(ax=ax[0])
        folded_lc.scatter(ax=ax[1], label=f"Period: {period:.3f} d");
        plt.title(lc.label)
        plt.show()
        return fig


    def update_best_guess(*args):
        """Updates the period when the index changes

        Must be seperate from f() because f changes when the period
        does, so b would constantly get changed back
        """
        lc = lcc[a.value]
        best_period = find_period(lc, method=method)
        b.value = best_period


    def factor_button(button):
        """Multiplies the period by the factor of the button
        """
        b.value = float(button.factor*b.value)


    def on_save_clicked(button):
        """Saves the plot and the folded light curve data
        """
        label = lcc[a.value].label
        fig = f(a.value, b.value, c.value)
        fig.savefig(f"{label}.png")
        folded_lc = lc.fold(b.value, normalize_phase=False)
        folded_lc.to_csv(f"{label}.csv", overwrite=True);

    # linking
    a.observe(update_best_guess, 'value')
    half_button.on_click(factor_button)
    double_button.on_click(factor_button)
    triple_button.on_click(factor_button)
    save.on_click(on_save_clicked)

    # layout
    factors = widgets.HBox(
        [half_button, double_button, triple_button],
        layout=widgets.Layout(width='50%')
    )
    out = widgets.interactive_output(f, {'index': a, 'period': b, 'center': c})
    # initialize and display
    update_best_guess()
    return display(a, b, factors, c, out, save)

