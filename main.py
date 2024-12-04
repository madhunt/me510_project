#!/usr/bin/python3
import numpy as np
import scipy as sci
import matplotlib.pyplot as plt
import os
import pandas as pd

import utils

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
    #initial_params_plot(path_data, path_fig, temp_list, params_init, strain_rate)

    # (2) optimize parameters for experimental data
    #optimize_params_dsgz(path_data, path_fig, temp_list, params_init, strain_rate)

    # (3) compare parameters for different temperatures
    #compare_params(path_data, path_fig, temp_list)

    # (4) sensitivity study for 20 and 180 C
    #sensitivity_study(path_data, path_fig, strain_rate)

    # (5) compare to other models
    params_jc = [132, 10, 0.034, 1.2, 0.7]          # Garcia-Gonzolez 2015
    optimize_params_other(path_data, path_fig, temp_list, params_jc, strain_rate, 
                          model_name=utils.johnson_cook_model, model_str="Johnson-Cook")
    params_gj = [141.1, 1.27, 28.27, 24.2, 0.015]   # Trufasu 2014
    optimize_params_other(path_data, path_fig, temp_list, params_gj, strain_rate, 
                          model_name=utils.gsell_jonas_model, model_str="G'Sell-Jonas")

    return

def optimize_params_other(path_data, path_fig, temp_list, params_init, strain_rate, model_name, model_str):

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

        # calculate absolute temperature (K)
        temp_abs = utils.celcius_to_kelvin(temp)

        # use gradient-descent to find optimal parameter values for this temp
        result = sci.optimize.minimize(utils.rmse_cost_func,
                                    x0=params_init,
                                    args=(strain, strain_rate, temp_abs, stress_exp, model_name))
        # add result to plot
        stress_model = model_name(strain, strain_rate, temp_abs, result.x)
        rmse = result.fun
        ax.plot(strain, stress_exp, "o", color=colors[i])
        ax.annotate(f"{temp} C, RMSE={np.round(rmse, 2)}", 
                    [0.063, max(stress_exp)], annotation_clip=False)
        ax.plot(strain, stress_model, "-", color=colors[i], label=f"{temp} C, RMSE = {np.round(rmse, 2)}")

        print(result)

    ax.set_xlabel("True Strain $\epsilon$")
    ax.set_ylabel("True Stress $\sigma$ [MPa]")
    fig.suptitle(f"Polyetheretherketone Stress-Strain Curves Fit with {model_str} Model,\n(Strain Rate = {strain_rate:.2E} $1/s$)")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(path_fig, f"stage3_optimize_params_{model_str}.png"), dpi=500)

    return



def sensitivity_study(path_data, path_fig, strain_rate):
    # load in parameters
    path_params = os.path.join(path_data, "params_optimized.csv")
    params_all = pd.read_csv(path_params)

    # initialize figure
    fig, ax = plt.subplots(ncols=2, nrows=4, tight_layout=True,
                           sharex=True, figsize=[12,12])
    colors = ['red', 'purple']
    for j, temp in enumerate([20, 180]):
        # calculate best fit model
        temp_abs = utils.celcius_to_kelvin(temp)
        strain = np.arange(0, 0.06, 0.001)
        params = params_all[str(temp)]
        model_best = utils.dsgz_model(strain, strain_rate, temp_abs, params)

        # loop through each parameter
        for i in params.index:
            axi = ax.flatten()[i]
            # plot best fit model
            axi.plot(strain, model_best, "-", color=colors[j],
                     label=f"{temp} C Best Fit")
            # calculate and plot upper bound
            params_high = params.copy()
            params_high[i] = params[i] + 0.5*params[i]
            model_high = utils.dsgz_model(strain, strain_rate, temp_abs, params_high)
            axi.plot(strain, model_high, "--", color=colors[j],
                     label="$\pm$50%")
            # calculate and plot lower bound
            params_low = params.copy()
            params_low[i] = params[i] - 0.5*params[i]
            model_low = utils.dsgz_model(strain, strain_rate, temp_abs, params_low)
            axi.plot(strain, model_low, "--", color=colors[j])

            # set axis title
            axi.set_title(params_all.iloc[i]["Unnamed: 0"], fontsize=16)
            axi.legend(loc="upper left")

            # set axis bounds to neaten figure
            if i == 5:
                axi.set_ylim([-10, 530])
            else:
                axi.set_ylim([-5, 180])
            # set labels to neaten figure
            if i == 6 or i == 7:
                axi.set_xlabel("True Strain $\epsilon$")
            if i in [0, 2, 4, 6]:
                axi.set_ylabel("True Stress $\sigma$ [MPa]")


    fig.suptitle("Sensitivity Study for DSGZ Model Parameters", fontsize=18)
    plt.savefig(os.path.join(path_fig, f"stage3_sensitivity.png"), dpi=500)
    return


def compare_params(path_data, path_fig, temp_list):
    path_params = os.path.join(path_data, "params_optimized.csv")
    params_all = pd.read_csv(path_params)
    params_all = params_all.rename(columns={"Unnamed: 0": "Parameter"})

    fig, ax = plt.subplots(ncols=1, nrows=len(params_all.index), 
                           tight_layout=True, sharex=True, sharey=True,
                           figsize=[8,12])
    for i in params_all.index:
        param_vals = params_all.iloc[i][1:].to_numpy()
        param_mean = np.mean(param_vals)
        param_norm = param_vals/param_mean
        ax[i].plot(temp_list, param_norm, "o-", color='red')
        ax[i].set_ylabel(params_all.iloc[i]["Parameter"],
                         rotation=0, fontsize=16)
        ax[i].xaxis.grid(True)
    ax[7].set_xlabel("Temperature ($^o$C)", fontsize=16)
    fig.suptitle("Normalized DSGZ Parameters for Different Temperatures",
                 fontsize=18)
    plt.savefig(os.path.join(path_fig, f"stage3_compare_params.png"), dpi=500)
    return


def optimize_params_dsgz(path_data, path_fig, temp_list, params_init, strain_rate):

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
        temp_abs = utils.celcius_to_kelvin(temp)

        # use gradient-descent to find optimal parameter values for this temp
        result = sci.optimize.minimize(utils.rmse_cost_func,
                                    x0=params_init,
                                    args=(strain, strain_rate, temp_abs, stress_exp, utils.dsgz_model))
        params_all[temp] = result.x

        # add result to plot
        stress_model = utils.dsgz_model(strain, strain_rate, temp_abs, result.x)
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
    plt.savefig(os.path.join(path_fig, f"stage3_optimize_params_dsgz.png"), dpi=500)

    # save parameters
    path_params = os.path.join(path_data, "params_optimized.csv")
    params_all = params_all.rename(columns={"Unnamed: 0": "Parameter"})
    params_all.to_csv(path_params)
    return



def initial_params_plot(path_data, path_fig, temp_list, params_init, strain_rate):
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
        temp_abs = utils.celcius_to_kelvin(temp)

        # calculate stress model with initial params
        stress_model = utils.dsgz_model(strain, strain_rate, temp_abs, params_init)
        rmse = utils.calc_rmse(stress_model, stress_exp)
        ax.plot(strain, stress_exp, "o", color=colors[i])
        ax.annotate(f"{temp} C, RMSE={np.round(rmse, 2)}", 
                    [0.063, max(stress_exp)], annotation_clip=False)
        ax.plot(strain, stress_model, "-", color=colors[i], label=f"{temp} C, RMSE = {np.round(rmse, 2)}")

    ax.set_xlabel("True Strain $\epsilon$")
    ax.set_ylabel("True Stress $\sigma$ [MPa]")
    fig.suptitle(f"Initial Parameters for DSGZ Model,\n(Strain Rate = {strain_rate:.2E} $1/s$)")
    ax.grid(True)
    fig.tight_layout()
    plt.savefig(os.path.join(path_fig, f"stage2_initial_params.png"), dpi=500)
    return


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