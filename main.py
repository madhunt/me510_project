#!/usr/bin/python3
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os
import pandas as pd

def main():

    # define paths to data and to save figures
    path_home = os.path.dirname(os.path.realpath(__file__))
    path_fig = os.path.join(path_home, "figures")
    path_data = os.path.join(path_home, "data")

    # define parameters from Drozdov, et. al. 2021 experiment
    temp_list = [20, 80, 120, 130, 140, 150, 160, 170, 180]
    strain_rate = 3.1e-3

    # use initial parameters from Duan, et. al. 2001
    params_init = [3.9, 1.91, 1.49, 0.0029, 11, 1191, 0.064, 11.7]

    # (1) compare experimental data with initial parameters
    #TODO need to do

    # (2) optimize parameters for experimental data
    #TODO clean up code
    #optimize_params()

    # (3) compare parameters for different temperatures
    #compare_params(path_data, path_fig, temp_list)

    # (4)

    return

def compare_params(path_data, path_fig, temp_list):
    path_params = os.path.join(path_data, "params_optimized.csv")
    params_all = pd.read_csv(path_params)
    params_all = params_all.rename(columns={"Unnamed: 0": "Parameter"})

    fig, ax = plt.subplots(ncols=1, nrows=len(params_all.index), 
                           tight_layout=True, sharex=True, sharey=True,
                           figsize=[8,12])
    colors = plt.cm.rainbow_r(np.linspace(0, 1, len(temp_list)))
    for i in params_all.index:
        param_vals = params_all.iloc[i][1:].to_numpy()
        param_mean = np.mean(param_vals)
        param_norm = param_vals/param_mean
        ax[i].plot(temp_list, param_norm, "o-", color='red',
                label=params_all.iloc[i]["Parameter"])
        ax[i].set_ylabel(params_all.iloc[i]["Parameter"],
                         rotation=0, fontsize=16)
        ax[i].xaxis.grid(True)
    ax[7].set_xlabel("Temperature ($^o$C)", fontsize=16)
    fig.suptitle("Normalized DSGZ Parameters for Different Temperatures",
                 fontsize=18)
    plt.savefig(os.path.join(path_fig, f"stage3_compare_params.png"), dpi=500)
    return


def optimize_params(path_data, path_fig):

    # initialize array to hold all parameters
    params_all = pd.DataFrame(index=["K", "C1", "C2", "C3", "C4", "a", "m", "alpha"], 
                              columns=temp_list)
    # initialize plot
    fig, ax = plt.subplots(1, 1)
    colors = plt.cm.rainbow_r(np.linspace(0, 1, len(temp_list)))
    
    # loop through all temperatures
    for i, temp in enumerate(temp_list):
        # load in data for given temp
        path_file = os.path.join(path_data, f"stress_strain_{temp}C.csv")
        data = pd.read_csv(path_file, names=["Strain", "Stress", ""])
        strain = data["Strain"]
        stress_exp = data["Stress"]

        # find absolute temp (in K)
        temp_abs = celcius_to_kelvin(temp)

        # use gradient-descent to find optimal parameter values for this temp
        result = sci.optimize.minimize(rmse_cost_func,
                                    x0=params_init,
                                    args=(strain, strain_rate, temp_abs, stress_exp, dsgz_model))
        params_all[temp] = result.x

        # add result to plot
        stress_model = dsgz_model(strain, strain_rate, temp_abs, result.x)
        rmse = result.fun
        ax.plot(strain, stress_exp, "o", color=colors[i])
        ax.annotate(f"{temp} C, RMSE={np.round(rmse, 2)}", 
                    [0.063, max(stress_exp)], annotation_clip=False)
        ax.plot(strain, stress_model, "-", color=colors[i], label=f"{temp} C, RMSE = {np.round(rmse, 2)}")

    ax.set_xlabel("True Strain $\epsilon$")
    ax.set_ylabel("True Stress $\sigma$ [MPa]")
    fig.suptitle(f"Polyetheretherketone Stress-Strain Curves Fit with DSGZ,\n(Strain Rate = {strain_rate:.2E} $1/s$)")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(path_fig, f"stage3_optimize_params.png"), dpi=500)

    # save parameters
    path_params = os.path.join(path_data, "params_optimized.csv")
    params_all.to_csv(path_params)
    return



    #####################################################
    # STAGE 2
    #####################################################
    # Drozdov 2021 parameters
    paper_str = "Drozdov et. al., 2021"
    fig_str = "drozdov_2021"
    # use parameters from Duan 2001
    params = [3.9, 1.91, 1.49, 0.0029, 11, 1191, 0.064, 11.7]
    temp_list = [20, 80, 120, 130, 140, 150, 160, 170, 180]
    strain = np.arange(0.001, 0.06, 0.001)
    strain_rate = 3.1e-3




    # initialize plot
    fig, ax = plt.subplots(1, 1, tight_layout=True)
    colors = plt.cm.rainbow(np.linspace(0, 1, len(temp)))

    for i, temp in enumerate(temp_list):
        # load in data for given temp
        path_file = os.path.join(path_data, f"stress_strain_{T}C.csv")
        data = pd.read_csv(path_file, names=["Strain", "Stress", ""])
        ax.plot(data["Strain"], data["Stress"], "o", color=colors[i])

        # find absolute temp (in Kelvin)
        temp_abs = celcius_to_kelvin(temp)
        stress_model = dsgz_model(strain, strain_rate, T_K,
                                          params)
        ax.plot(strain, stress_model, color=colors[i], label=f"{T} C")
    ax.set_xlabel("True Strain $\epsilon$")
    ax.set_ylabel("True Stress $\sigma$ [MPa]")
    ax.legend(title=f"{strain_rate:.2E} $1/s$")
    ax.grid(True)
    #ax.set_ylim([0, 80])
    fig.suptitle(f"Replicated Stress-Strain Curves from {paper_str}")
    plt.savefig(os.path.join(path_fig, f"testing_{fig_str}.png"))
        


    return

def celcius_to_kelvin(temp):
    return temp + 273.15
def kelvin_to_celcius(temp):
    return temp - 273.15





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


def johnson_cook_model(strain, strain_rate, hom_temp, params):

    C1 = params[0]
    C2 = params[1]
    C3 = params[2]
    N = params[3]
    M = params[4]


    term1 = C1 + C2 * strain**N
    term2 = 1 + C3 * np.log(strain_rate)
    term3 = 1 - hom_temp**M

    stress_model = term1 * term2 * term3

    return stress_model

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






if __name__ == "__main__":
    main()





    # Zheng 2017 parameters -- WORKS
    #paper_str = "Zheng et. al., 2017" 
    #fig_str = "zheng_2017"
    #params = [0.5684, -2.155, -0.477, 0.0071, 6.1578, 1358.1, 0.004, 10]
    #temp = [293, 333, 373, 416, 438, 473, 508, 543]
    #strain = np.arange(0.001, 0.9, 0.01)
    #strain_rate = 8.3e-3
    
    # Duan 2001 parameters -- WORKS
    #paper_str = "Duan et. al., 2001"
    #fig_str = "duan_2001"
    #params = [3.9, 1.91, 1.49, 0.0029, 11, 1191, 0.064, 11.7]
    #temp = [296, 323]
    #strain = np.arange(0.001, 1, 0.01)
    #strain_rate = 0.001