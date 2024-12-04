#!/usr/bin/python3
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os
import pandas as pd

def celcius_to_kelvin(temp):
    return temp + 273.15

def rmse_cost_func(params, strain, strain_rate, temp, stress_exp, model_func):
    #TODO CHENAGE DOCS
    """
    Calculates RMSE for a given model. To use when optimizing parameters.
    INPUTS
        params      : list of float : List of parameters for chosen model, size 1xp.
        x           : np array      : Array of x-values, size 1xN.
        y_exp       : np array      : Array of experimental y-values, size 1xN.
        model_func  : func handle   : Name of chosen model function.
    RETURNS 
        rmse        : float         : Root Mean Square Error calculated for given model and parameters.
    """
    y_model = model_func(strain, strain_rate, temp, params)
    rmse = calc_rmse(y_model, stress_exp)
    return rmse


def calc_rmse(y_model, y_exp):
    """
    Calculates Root Mean Square Error.
    INPUTS
        y_model : np array  : Modeled y-values, size 1/N.
        y_exp   : np array  : Experimental y-values, size 1xN.
    RETURNS
        rmse    : float     : RMSE of model to experimental values.
    """
    assert len(y_model) == len(y_exp)
    N = len(y_exp)
    rmse = np.sqrt(1/N * np.sum((y_model - y_exp)**2))
    return rmse



def dsgz_model(strain, strain_rate, temp_abs, params):
    # define 8 model parameters
    K = params[0]
    C1 = params[1]
    C2 = params[2]
    C3 = params[3]
    C4 = params[4]
    a = params[5]
    m = params[6]
    alpha = params[7]

    # calculate f(strain); initial elastic and strain hardening
    f = (np.exp(-C1 * strain) + strain**(C2)) * (1 - np.exp(-alpha*strain))
    # calculate h(strain_rate, temp)
    h = strain_rate**m * np.exp(a/temp_abs)
    # calculate stress model
    yield_behavior = (strain * np.exp(1 - strain/(C3 * h))) / (C3 * h)


    stress_model = K * h * (f + np.exp(strain*(np.log(h)-C4)) * (yield_behavior - f))

    return stress_model


def johnson_cook_model(strain, strain_rate, temp_abs, params):

    C1 = params[0]      # material constant
    C2 = params[1]      # material constant
    C3 = params[2]      # strain rate sensitivity
    N = params[3]       # strain hardening exponent
    M = params[4]       # temperature sensitivity

    # calculate homologous temp
    temp_melt = 339
    temp_hom = temp_abs / celcius_to_kelvin(temp_melt)
    
    strain_hardening = C1 + C2 * strain**N
    strain_rate_sens = 1 + C3 * np.log(strain_rate)
    thermal_softening = 1 - temp_hom**M

    stress_model = strain_hardening * strain_rate_sens * thermal_softening

    return stress_model


def gsell_jonas_model(strain, strain_rate, temp_abs, params):
    K = params[0]     
    a = params[1]     
    W = params[2]     
    h = params[3]     
    m = params[4]    

    nonlin_elastic = (1 - np.exp(-W*strain)) 
    strain_hardening = np.exp(h * strain**2)
    viscous = strain_rate**m

    stress_model = K * strain_hardening * nonlin_elastic * viscous * np.exp(a / temp_abs)

    return stress_model

