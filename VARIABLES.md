<style TYPE="text/css">
code.has-jax {font: inherit; font-size: 100%; background: inherit; border: inherit;}
</style>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
    tex2jax: {
        inlineMath: [['$','$'], ['\\(','\\)']],
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'] // removed 'code' entry
    }
});
MathJax.Hub.Queue(function() {
    var all = MathJax.Hub.getAllJax(), i;
    for(i = 0; i < all.length; i += 1) {
        all[i].SourceElement().parentNode.className += ' has-jax';
    }
});
</script>
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>



# VARIABLES IN THE PICKLES

So far the luminosity follow-up automatic scripts generate pickles with dictionaries at each step of the cycle. This is kept in this way so that previously written scripts can also be ran without any conflicts on the new files.

**IMPORTANT:** This is due to change during LS2!

The files created by the script are:
  - `fill_xxxx.pkl.gz` : Data at STABLE BEAMS
  - `fill_xxxx_cycle.pkl.gz` : Data from Injection to Flat Top
  - `fill_xxxx_cycle_model.pkl.gz` : Model data from Injection to Flat Top
  - `fill_xxxx_cycle_model_Inj2SB.pkl.gz` : Model data from Injection to SB (still under test)
  - `fill_xxx_fits.pkl.gz` : Parameters of fits performed on SB data
  - `fill_xxx_lifetime.pkl.gz` : Data on lifetime and Losses
  - `fill_xxx_lumi_calc.pkl.gz` : Data on calculated Luminosity
  - `fill_xxx_lumi_meas.pkl.gz` : Data on measured Luminosity
  - `fill_xxx_sbmodel_caseX.pkl.gz` : Lumi model for case X.



## `fill_xxxx.pkl.gz` : SB DATA

The file includes a Python dictionary for the SB data for the two beams:
The variables included are :

-`eh_interp_raw_noncoll` : Raw horizontal emittance of non-colliding bunches
-`ev_interp_raw_noncoll` : Raw vertical emittance of non-colliding bunches
-`eh_interp_raw_coll` : Raw horizontal emittance of colliding bunches
-`ev_interp_raw_coll` : Raw vertical emittance of colliding bunches

-`eh_interp_noncoll` : Smoothed horizontal emittance of non-colliding bunches
-`ev_interp_noncoll` : Smoothed vertical emittance of non-colliding bunches
-`eh_interp_coll` : Smoothed horizontal emittance of colliding bunches
-`ev_interp_coll` : Smoothed vertical emittance of colliding bunches


-`init_emit_h_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the non-colliding bunches
-`init_emit_v_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the vertical emittance of the non-colliding bunches
-`init_emit_h_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the colliding bunches
-`init_emit_v_coll`  : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the vertical emittance of the colliding bunches
-`init_emit_h_noncoll_full` : 1st Parameter of the exponential fit for the full SB duration on the horizontal emittance of the non-colliding bunches
-`init_emit_v_noncoll_full` : 1st Parameter of the exponential fit for the full SB duration on the vertical emittance of the non-colliding bunches
-`init_emit_h_coll_full` : 1st Parameter of the exponential fit for the full SB duration on the horizontal emittance of the colliding bunches
-`init_emit_v_coll_full` : 1st Parameter of the exponential fit for the full SB duration on the vertical emittance of the colliding bunches


-`tau_emit_h_noncoll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the non-colliding bunches
-`tau_emit_v_noncoll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the vertical emittance of the non-colliding bunches
-`tau_emit_h_coll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the colliding bunches
-`tau_emit_v_coll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the volliding bunches
-`tau_emit_h_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the horizontal emittance of the non-colliding bunches
-`tau_emit_v_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the vertical emittance of the non-colliding bunches
-`tau_emit_h_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the horizontal emittance of the colliding bunches
-`tau_emit_v_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the vertical emittance of the colliding bunches


-`b_inten_interp_noncoll` : Bunch intensity of non-colliding bunches
-`b_inten_interp_coll` : Bunch intensity of colliding-bunches

-`init_inten_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the non-colliding bunches
-`init_inten_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the colliding bunches
-`init_inten_noncoll_full` : 1st Parameter of the exponential fit for the full SB duration on the bunch intensity of the non-colliding bunches
-`init_inten_coll_full` : 1st Parameter of the exponential fit for the full SB duration on the bunch intensity of the colliding bunches

-`tau_inten_noncoll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the non-colliding bunches
-`tau_inten_coll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the colliding bunches
-`tau_inten_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch intensity of the non-colliding bunches
-`tau_inten_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch intensity of the colliding bunches

-`intensity_lifetime` : Intensity lifetime

-`bl_interp_m_noncoll` : Bunch length of non-colliding bunches
-`bl_interp_m_coll` : Bunch length of non-colliding bunches

-`init_bl_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch length of the non-colliding bunches
-`init_bl_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch length of the colliding bunches
-`init_bl_noncoll_full`: 1st Parameter of the exponential fit for the full SB duration on the bunch length of the non-colliding bunches
-`init_bl_coll_full`: 1st Parameter of the exponential fit for the full SB duration on the bunch length of the colliding bunches

-`tau_bl_noncol`: Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch length of the non-colliding bunches
-`tau_bl_coll`: Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch length of the colliding bunches
-`tau_bl_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch length of the non-colliding bunches
-`tau_bl_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch length of the colliding bunches


-`time_range` : Time in UNIX time of measurements

-`slots_filled_noncoll` : Bunch slots of the non-colliding bunches
-`slots_filled_coll`: Bunch slots of the colliding bunches 
-`xing_angle` : Crossing angle in IP1/IP5

All variables have sub-keys of [1] and [2], meaning beam 1 and 2 respectively. The only exceptions are: 
    - `time_range` : this is a flat array
    - `xing_angle` : the sub-keys are [1] and [5], for IP1 and IP5 respectively.

For all variables with subkeys [1] and [2], the content is an array of arrays. The axis-0 of the array corresponds to time instances, while axis-1 corresponds to bunches.

**Example**: *I need the time evolution of the average bbb horizontal smoothed emittance of beam 1 for the colliding bunches*:

`np.nanmean(sb_dict['eh_interp_coll'][1], axis=1)` -> returns a flat array, each element is the average (nanmean) of all the horizontal smoothed emittance of all bunches of beam 1 for that time instance.


---

## `fill_xxxx_cycle.pkl.gz` : CYCLE DATA

This includes a dictionary that has directly the keys : 
- `'beam_1'`
- `'beam_2'`

To define the beam for which you are interested. Under these, there is the separation in two different energy steps:
- `'Injection'` : This refers to flat bottom
- `'he_before_SB'` : This refers to flat top (before SB)

In each case of `Injection` and `he_before_SB` the variables that are available at this level are:
 - `t_start`: Unix time in which the injection started
 - `t_end`: Unix time in which the injection ended (i.e. the Start of Ramp)
 - `filled_slots`: Bunch slots that are filled

Additionally, there are two more sub-keys:
- `'at_start'` : This holds all the variables at the start of this cycle step (if `Injection`, then at the Injection, if `he_before_SB` then at the End of Ramp)
- `'at_end'`  : This holds all the variables at the end of this cycle step (if `Injection`, then at the Start of Ramp, if `he_before_SB` then at the Start of Stable Beams)

For each of the `at_start`, `at_end`, we get the following variables:
 - `emith` : Horizontal Emittance for each bunch 
 - `emitv` : Vertical Emittance for each bunch 
 - `intensity` : Intensityfor each bunch 
 - `blength` : Bunch Length for each bunch 
 - `brightness` : Brightness for each bunch 
 - `time_meas`: Measured time for each bunch

To sum up the cycle steps:
Injection             = `cycle_dict['beam_1']['Injection']['at_start']`
Start of Ramp         = `cycle_dict['beam_1']['Injection']['at_end']` 
End of Ramp           = `cycle_dict['beam_1']['he_before_SB']['at_start']` 
Start of Stable Beams = `cycle_dict['beam_1']['he_before_SB']['at_end']` 
 

**Example**: *I need the horizontal emittances at end of RAMP of all the bunches of Beam 2*:

`cycle_dict['beam_1']['he_before_SB']['at_start']['emith']` 

---
## `fill_xxxx_cycle_model.pkl.gz` : CYCLE MODEL DATA

The cycle model data pickle follows the same naming patterns as found in the cycle data pickle.

The first keys are `['beam_1']` and `['beam_2']` to select your beam. Then the sub-keys follow the same scheme : `['Injection']` and `['he_before_SB']`, followed by `['at_start']`, `['at_end']`.

The `t_start` and `t_end` variables at each  `['Injection']` and `['he_before_SB']` step are not available here (not needed). The `filled_slots` variable is still available.

The variables under the `['at_start']`, `['at_end']` include:
- `emith`
- `emitv`
- `blength`
- `time_meas`

Therefore, intensity and brightness are not calculated with the model.

---
## `fill_xxxx_cycle_model_Inj2SB.pkl.gz` : CYCLE MODEL DATA FROM INJECTION TO STABLE BEAMS

This is a data pickle under test. The idea is to calculate growths from injection up to stable beams. The values at the start of injection are taken from data. The value at the start of ramp is propagated at the end of ramp (no blow-up during the ramp is expected).

The variable scheme is similar to the rest cycle data pickles.

---
## `fill_xxx_fits.pkl.gz` : FITS ON STABLE BEAMS

This pickle holds data from fits performed on the evolution of machine and beam parameters during the Stable Beams. Some of the variables in this pickle have been propagated to the SB data pickle.

The first level of keys are :
- `['beam_1']`
- `['beam_2']`
- 
- `['ATLAS']` 
- `['CMS']`


The `['ATLAS']` and `['CMS']` keys, have the following variables as sub-keys:


 - `init_lumi_calc_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the calculated luminosity of the non-colliding bunches (are you suprised that it is empty?)
 - `init_lumi_meas_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the measured luminosity of the non-colliding bunches (are you suprised that it is empty?)
 - `init_lumi_calc_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the calculated luminosity of the colliding bunches
 - `init_lumi_meas_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the calculated luminosity of the colliding bunches
 - `tau_lumi_calc_noncoll_full` : Calculated luminosity lifetime from exponential fit for the full SB duration of the non-colliding bunches
 - `tau_lumi_meas_noncoll_full` : Measured luminosity lifetime from exponential fit for the full SB duration of the non-colliding bunches
 - `tau_lumi_calc_coll_full` : Calculated luminosity lifetime from exponential fit for the full SB duration of the colliding bunches
 - `tau_lumi_meas_coll_full` : Calculated luminosity lifetime from exponential fit for the full SB duration of the colliding bunches
 - `tau_lumi_calc_coll` : Calculated luminosity lifetime from exponential fit for the first `t_fit_length` hours of SB of the colliding bunches
 - `tau_lumi_meas_coll` : Measured luminosity lifetime from exponential fit for the first `t_fit_length` hours of SB of the colliding bunches
 - `tau_lumi_calc_noncoll` : Calculated luminosity lifetime from exponential fit for the first `t_fit_length` hours of SB of the non-colliding bunches
 - `tau_lumi_meas_noncoll` : Measured luminosity lifetime from exponential fit for the first `t_fit_length` hours of SB of the non-colliding bunches
 
The `['beam_1']` and `['beam_2']` keys, have the following variables as sub-keys (as seen already in SB file) :

-`init_emit_h_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the non-colliding bunches
-`init_emit_v_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the vertical emittance of the non-colliding bunches
-`init_emit_h_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the colliding bunches
-`init_emit_v_coll`  : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the vertical emittance of the colliding bunches
-`init_emit_h_noncoll_full` : 1st Parameter of the exponential fit for the full SB duration on the horizontal emittance of the non-colliding bunches
-`init_emit_v_noncoll_full` : 1st Parameter of the exponential fit for the full SB duration on the vertical emittance of the non-colliding bunches
-`init_emit_h_coll_full` : 1st Parameter of the exponential fit for the full SB duration on the horizontal emittance of the colliding bunches
-`init_emit_v_coll_full` : 1st Parameter of the exponential fit for the full SB duration on the vertical emittance of the colliding bunches
-`tau_emit_h_noncoll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the non-colliding bunches
-`tau_emit_v_noncoll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the vertical emittance of the non-colliding bunches
-`tau_emit_h_coll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the colliding bunches
-`tau_emit_v_coll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the horizontal emittance of the volliding bunches
-`tau_emit_h_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the horizontal emittance of the non-colliding bunches
-`tau_emit_v_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the vertical emittance of the non-colliding bunches
-`tau_emit_h_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the horizontal emittance of the colliding bunches
-`tau_emit_v_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the vertical emittance of the colliding bunches
-`init_inten_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the non-colliding bunches
-`init_inten_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the colliding bunches
-`init_inten_noncoll_full` : 1st Parameter of the exponential fit for the full SB duration on the bunch intensity of the non-colliding bunches
-`init_inten_coll_full` : 1st Parameter of the exponential fit for the full SB duration on the bunch intensity of the colliding bunches
-`tau_inten_noncoll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the non-colliding bunches
-`tau_inten_coll` : Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch intensity of the colliding bunches
-`tau_inten_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch intensity of the non-colliding bunches
-`tau_inten_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch intensity of the colliding bunches
-`init_bl_noncoll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch length of the non-colliding bunches
-`init_bl_coll` : 1st Parameter of the exponential fit for the first `t_fit_length` hours of SB on the bunch length of the colliding bunches
-`init_bl_noncoll_full`: 1st Parameter of the exponential fit for the full SB duration on the bunch length of the non-colliding bunches
-`init_bl_coll_full`: 1st Parameter of the exponential fit for the full SB duration on the bunch length of the colliding bunches
-`tau_bl_noncol`: Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch length of the non-colliding bunches
-`tau_bl_coll`: Lifetime in [h] from exponential fit for the first `t_fit_length` hours of SB on the bunch length of the colliding bunches
-`tau_bl_noncoll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch length of the non-colliding bunches
-`tau_bl_coll_full` : Lifetime in [h] from exponential fit for the full SB duration on the bunch length of the colliding bunches


---
## `fill_xxx_lifetime.pkl.gz` : LIFETIME & LOSSES

This pickle holds data for the intensity lifetime and the normalized losses (effective cross-section) of the two beams.

The variable `time_range` which gives the UNIX time of the measurements is included in the top level of this dictionary. Then there is a sub-key for each beam, labelled `[1]` and `[2]`.

For each of these keys the variables under it are:

- `dndt_bbb` : The evolution $\frac{dN}{dt}$ for each beam
- `tau_Np_bbb` : $\frac{-1}{\frac{dN}{dt}N_{0}}$
- `life_time_bbb` : $\frac{1}{(\frac{1}{tau_Np_bbb})}$
- `life_time_Boff_bbb`$\frac{1}{(\frac{1}{tau_Np_bbb}) - (\frac{1}{tau_BOff_bbb}) }$, where `tau_BOff_bbb` is the burn-off expected tau.
- `losses_dndtL_bbb` : This is the $\frac{dN}{dt}$ normalized with the total Luminosity
- `slots`: Filled bunch slots for the respective beam


The variables
- `life_time_Boff_tot`
- `life_time_tot`
- `tau_Np_tot`

are meant to do the same thing, but instead of bunch-by-bunch, to use the total (i.e. sum) intensity and luminosity.

---
## `fill_xxx_lumi_calc.pkl.gz` : CALCULATED LUMINOSITY

This pickle holds the data from the calculation of luminosity from machine parameters.

At the first level there are available the copies of the `xing_angle` and `time_range` variables, and then the dictionary splits into `ATLAS` and `CMS` keys. Each of these keys, holds the variable `bunch_lumi`, which represents the bunch by bunch calculated luminosity in Hz/$μ$b


---
## `fill_xxx_lumi_meas.pkl.gz` : MEASURED LUMINOSITY

This pickle holds the data from the measurement of luminosity from the two experiments (Massi files).

The dictionary holds the keys `ATLAS` and `CMS`, each holding one variable called `bunch_lumi`, which represents the bunch by bunch calculated luminosity in Hz/$μ$b.

---
## `fill_xxx_sbmodel_caseX.pkl.gz` : LUMINOSITY MODEL FOR CASE X

This file holds the data from the luminosity model ran for the case defined as case X (number) in the `config.py` file.

The dictionary splits into sub-keys, defined by the model name:
- `EmpiricalBlowupLosses`: intensities and emittances are taken from data
- `EmpiricalBlowupBOff` : emittances taken from data, intensities from model
- `IBSBOff` : both emittances and intensities are taken from the model
- `IBSLosses` : intensities from the data emittances from the model

Each holds these variables, which split into the sub-keys `[1]` and `[2]` for each beam respectively :


- `eh_interp_noncoll` : Horizontal emittance of the non-colliding bunches
- `ev_interp_noncoll` : Vertical emittance of the non-colliding bunches
- `eh_interp_coll` : Horizontal emittance of the colliding bunches
- `ev_interp_coll` : Vertical emittance of the non-colliding bunches


- `eh_interp_noncoll_IBScorr` : Horizontal emittance of the non-colliding bunches corrected for IBS
- `ev_interp_noncoll_IBScorr` : Vertical emittance of the non-colliding bunches corrected for IBS
- `eh_interp_coll_IBScorr` : Horizontal emittance of the colliding bunches corrected for IBS
- `ev_interp_coll_IBScorr` : Vertical emittance of the colliding bunches corrected for IBS

- `bl_interp_m_noncoll` : Bunch Length of the non-colliding bunches
- `bl_interp_m_coll` : Bunch Length of the colliding bunches

- `b_inten_interp_noncoll` : Bunch intensity of the non-colliding bunches
- `b_inten_interp_coll` : Bunch intensity of the non-colliding bunches


The luminosity is given by the variable : 
- `bunch_lumi` : Luminosity with the keys `ATLAS` and `CMS`

And the time of the measurements : 
- `time_range` : UNIX time for the measurements

The last two variables are dictionaries holding extra info on the settings of the model and the case you are looking at:


- `case`: The current case(s) and its parameters
    + `case` : The index (number) of the case
    + `cor_fact_1h` : The correction factor in terms of [%] that was applied in the horizontal emittance of beam 1
    + `cor_fact_1v` : The correction factor in terms of [%] that was applied in the vertical emittance of beam 1
    + `cor_fact_2h` : The correction factor in terms of [%] that was applied in the horizontal emittance of beam 2
    + `cor_fact_2v` : The correction factor in terms of [%] that was applied in the vertical emittance of beam 2


- `settings`: The settings with which the lumi model was initiated
    + `tau_empirical_h1_coll`
    + `tau_empirical_h2_coll`
    + `tau_empirical_v1_coll`
    + `tau_empirical_v2_coll`   
    + `tau_empirical_h1_noncol`
    + `tau_empirical_h2_noncoll`
    + `tau_empirical_v1_noncoll`
    + `tau_empirical_v2_noncoll`
    + `blengthBU`
    + `BOff`
    + `emitBU`







--------
--------
*09.10.2017 - Nikos Karastathis*