# -*- coding: utf-8 -*-

def emittance_dictionary(filln=None, rescale=False, period = None):

    e_dict = {'betaf_h':{}, 'betaf_v':{}, 'gamma':{}, 
          'sigma_corr_h':{}, 'sigma_corr_v':{},
          'rescale_sigma_h':{}, 'rescale_sigma_v':{}, 'scale_h': {}, 'scale_v': {}}
          
    if filln is None:
        raise ValueError('A fill number must be provided to select calibration!')
    
    print 'rescale = %s'%rescale
    print 'period = %s'%period

    if filln>5690:        

        for kk in e_dict.keys():
            e_dict[kk] = {450:{}, 6500:{}}

        ###Beam 1:
        e_dict['betaf_h'][450][1]           = 206.8 #-- updated 5839 203.4 #203.4
        e_dict['betaf_h'][6500][1]          = 188.2 #-- updated 5839 188.2 #200.7 #200.7

        e_dict['betaf_v'][450][1]           = 287.3 #-- updated 5839 317.54 #317.54
        e_dict['betaf_v'][6500][1]          = 301.  #-- updated 5839 301.0 #329.9 #329.9
        
        e_dict['sigma_corr_h'][450][1]      = 0.44843 #updated 6055 #0.4109 #-- updated 5839 0.74#0.7400
        e_dict['sigma_corr_h'][6500][1]     = 0.2527 #updated 6055 #0.2252 #-- updated 5839 0.2252 #0.415#0.4150 
        
        e_dict['sigma_corr_v'][450][1]      = 0.494 #updated 6055 #0.6352#-- updated 5839 0.646#0.6460 
        e_dict['sigma_corr_v'][6500][1]     = 0.3218 #updated 6055 #0.281 #-- updated 5839 0.281 #0.346#0.3460
        
        e_dict['scale_h'][450][1]           = 0.02626 #updated 6055 #0.0247 #-- updated 5839 0.0274#0.0274
        e_dict['scale_h'][6500][1]          = 0.023 #updated 6055 #0.02184 #-- updated 5839 0.02184#0.0283#0.0283
        
        e_dict['scale_v'][450][1]           = 0.02648 #updated 6055 #0.0265 #-- updated 5839 0.0272#0.0272
        e_dict['scale_v'][6500][1]          = 0.0236 #updated 6055 #0.02284 #-- updated 5839 0.02284 #0.0246#0.0246
        
        e_dict['rescale_sigma_h'][450][1]   = 1.
        e_dict['rescale_sigma_h'][6500][1]  = 1.
        
        e_dict['rescale_sigma_v'][450][1]   = 1.
        e_dict['rescale_sigma_v'][6500][1]  = 1.

        #### Beam 2:
        e_dict['betaf_h'][450][2]           = 193.1 #-- updated 5839 200.6 #200.6
        e_dict['betaf_h'][6500][2]          = 208.8 #-- updated 5839 208.8#200.0 #200.
        
        e_dict['betaf_v'][450][2]           = 337.6#-- updated 5839 328.1 #328.1
        e_dict['betaf_v'][6500][2]          = 340.3#-- updated 5839 340.3#328.2 #328.2
        
        e_dict['sigma_corr_h'][450][2]      = 0.38769 #updated 6055 #0.442#-- updated 5839 0.709#0.7090
        e_dict['sigma_corr_h'][6500][2]     = 0.3323 #updated 6055 #0.3352#-- updated 5839 0.3352 #0.489 #0.4890
        
        e_dict['sigma_corr_v'][450][2]      = 0.48528 #updated 6055 #0.6291 #-- updated 5839 0.661#0.6610
        e_dict['sigma_corr_v'][6500][2]     = 0.29511 #updated 6055 #0.32374#-- updated 5839 0.32374#0.305#0.3050
        
        e_dict['scale_h'][450][2]           = 0.02814 #updated 6055 #0.0273 #-- updated 5839 0.0301#0.0301
        e_dict['scale_h'][6500][2]          = 0.03016 #updated 6055 #0.02948#-- updated 5839 0.02948#0.0403#0.0403
        
        e_dict['scale_v'][450][2]           = 0.02898 #updated 6055 #0.02888 #-- updated 5839 0.0285#0.0285
        e_dict['scale_v'][6500][2]          = 0.0319#updated 6055 #0.03126 #-- updated 5839 0.03126 #0.0315#0.0315

        e_dict['rescale_sigma_h'][450][2]   = 1.
        e_dict['rescale_sigma_h'][6500][2]  = 1.
        
        e_dict['rescale_sigma_v'][450][2]   = 1.
        e_dict['rescale_sigma_v'][6500][2]  = 1.

        # gamma
        e_dict['gamma'][450]                = 479.6 
        e_dict['gamma'][6500]               = 6927.6
        
        print 'Using calibration A-2017'

        # e_dict['betaf_h'][450][1]           = 206.8 #-- updated 5839 203.4 #203.4
        # e_dict['betaf_h'][6500][1]          = 188.2 #-- updated 5839 188.2 #200.7 #200.7
        # e_dict['betaf_v'][450][1]           = 287.3 #-- updated 5839 317.54 #317.54
        # e_dict['betaf_v'][6500][1]          = 301.  #-- updated 5839 301.0 #329.9 #329.9
        # e_dict['sigma_corr_h'][450][1]      = 0.4109 #-- updated 5839 0.74#0.7400
        # e_dict['sigma_corr_h'][6500][1]     = 0.2252 #-- updated 5839 0.2252 #0.415#0.4150 
        # e_dict['sigma_corr_v'][450][1]      = 0.6352#-- updated 5839 0.646#0.6460 
        # e_dict['sigma_corr_v'][6500][1]     = 0.281 #-- updated 5839 0.281 #0.346#0.3460
        # e_dict['scale_h'][450][1]           = 0.0247 #-- updated 5839 0.0274#0.0274
        # e_dict['scale_h'][6500][1]          = 0.02184 #-- updated 5839 0.02184#0.0283#0.0283
        # e_dict['scale_v'][450][1]           = 0.0265 #-- updated 5839 0.0272#0.0272
        # e_dict['scale_v'][6500][1]          = 0.02284 #-- updated 5839 0.02284 #0.0246#0.0246
        # e_dict['rescale_sigma_h'][450][1]   = 1.
        # e_dict['rescale_sigma_h'][6500][1]  = 1.
        # e_dict['rescale_sigma_v'][450][1]   = 1.
        # e_dict['rescale_sigma_v'][6500][1]  = 1.
        # # Beam 2:
        # e_dict['betaf_h'][450][2]           = 193.1 #-- updated 5839 200.6 #200.6
        # e_dict['betaf_h'][6500][2]          = 208.8 #-- updated 5839 208.8#200.0 #200.
        # e_dict['betaf_v'][450][2]           = 337.6#-- updated 5839 328.1 #328.1
        # e_dict['betaf_v'][6500][2]          = 340.3#-- updated 5839 340.3#328.2 #328.2
        # e_dict['sigma_corr_h'][450][2]      = 0.442#-- updated 5839 0.709#0.7090
        # e_dict['sigma_corr_h'][6500][2]     = 0.3352#-- updated 5839 0.3352 #0.489 #0.4890
        # e_dict['sigma_corr_v'][450][2]      = 0.6291 #-- updated 5839 0.661#0.6610
        # e_dict['sigma_corr_v'][6500][2]     = 0.32374#-- updated 5839 0.32374#0.305#0.3050
        # e_dict['scale_h'][450][2]           = 0.0273 #-- updated 5839 0.0301#0.0301
        # e_dict['scale_h'][6500][2]          = 0.02948#-- updated 5839 0.02948#0.0403#0.0403
        # e_dict['scale_v'][450][2]           = 0.02888 #-- updated 5839 0.0285#0.0285
        # e_dict['scale_v'][6500][2]          = 0.03126 #-- updated 5839 0.03126 #0.0315#0.0315
        # e_dict['rescale_sigma_h'][450][2]   = 1.
        # e_dict['rescale_sigma_h'][6500][2]  = 1.
        # e_dict['rescale_sigma_v'][450][2]   = 1.
        # e_dict['rescale_sigma_v'][6500][2]  = 1.
        # # gamma
        # e_dict['gamma'][450]                = 479.6 
        # e_dict['gamma'][6500]               = 6927.6
        
    else:         
        raise ValueError('What?!')     
    
         
    
    return(e_dict)
