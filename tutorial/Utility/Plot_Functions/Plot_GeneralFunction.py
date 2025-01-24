import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import KPIUtility


def plotSingleProperty(x_axis, y_axis, plotname):
    fig, ax = plt.subplots(nrows=1, figsize=(30, 18))
    ax.plot(x_axis, y_axis, 'bs', markersize=10)
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%m-%d'))
    #ax.text(0.05, 0.95, "2 for heatpump, 3 for heatpump and boiler, 4 for boiler", transform=ax.transAxes, fontsize=40, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5))
    #ax.set_ylim(-1, 6)
    plt.savefig('../Plots/' + plotname + '.png')
    plt.close()


def plot_AOverB(x_axis, y_left_upper, y_right_upper, y_right_lower, y_leftLabel_upper, y_rightLabel_upper, y_leftLabel_lower, y_rightLabel_lower, plotname):

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(40, 36))

    # Upper plot
    # Left Y axis
    ax1.plot(x_axis, y_left_upper, color='blue')
    ax1.set_xlabel("Time (UTC)")
    ax1.set_ylabel(y_leftLabel_upper, color='blue')
    # Right Y axis
    ax1_twin = ax1.twinx()  
    ax1_twin.plot(x_axis, y_right_upper, color='red')
    ax1_twin.set_ylabel(y_rightLabel_upper, color='red')

    ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # Lower plot 
    # Left Y axis 
    ax2.plot(x_axis, y_left_upper / y_right_upper, color='blue', label='')
    ax2.set_xlabel("Time (UTC)")
    ax2.set_ylabel(y_leftLabel_lower, color='blue')
    # Right Y axis
    ax2_twin = ax2.twinx()  
    ax2_twin.plot(x_axis, y_right_lower, color='red')
    ax2_twin.set_ylabel(y_rightLabel_lower, color='red')

    ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H-%M'))

    plt.savefig('../Plots/' + plotname + '.png')
    plt.close()


