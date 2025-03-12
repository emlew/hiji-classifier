import itertools
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

def plot_vs_time(df,type,title):
    stds = [np.std(df[i]).round(5) for i in df.columns[:6]]
    max_value = max(map(abs, list(itertools.chain(df[type+'X'], df[type+'Y'], df[type+'Z']))))
    start = min(df['timestamp']+2)
    
    x_patch = mpatches.Patch(color='tab:blue', label='X')
    y_patch = mpatches.Patch(color='tab:orange', label='Y')
    z_patch = mpatches.Patch(color='tab:green', label='Z')

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(df['isoTimestamp'], df[type+'X'])
    axs[1].plot(df['isoTimestamp'], df[type+'Y'], 'tab:orange')
    axs[2].plot(df['isoTimestamp'], df[type+'Z'], 'tab:green')
    
    axs[0].text(start,-max_value,"std="+str(stds[0]), label='X (Blue)')
    axs[1].text(start,-max_value,str(stds[1]), label='Y (Orange)')
    axs[2].text(start,-max_value,str(stds[2]), label='Z (Green)')
    
    for ax in axs.flat:
        ax.label_outer()
        ax.set_ylim([-max_value-1,max_value+1])
        ax.set_xticks([])
    
    fig.legend(handles=[x_patch, y_patch, z_patch], loc='lower center', ncol=3, fontsize=10)
    fig.suptitle(title, fontsize=16, fontweight='bold')
        
    plt.show()
    
def comparison_time_plot(df1,df2,type,title1,title2):
    stds1 = [np.std(df1[i]).round(5) for i in df1.columns[:6]]
    stds2 = [np.std(df2[i]).round(5) for i in df2.columns[:6]]
        
    max_value = max(max(map(abs, list(itertools.chain(df1[type+'X'], df1[type+'Y'], df1[type+'Z'])))), 
                    max(map(abs, list(itertools.chain(df2[type+'X'], df2[type+'Y'], df2[type+'Z'])))))  
     
    x_patch = mpatches.Patch(color='tab:blue', label='X')
    y_patch = mpatches.Patch(color='tab:orange', label='Y')
    z_patch = mpatches.Patch(color='tab:green', label='Z')

    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(df1['timestamp'], df1[type+'X'])
    axs[1, 0].plot(df1['timestamp'], df1[type+'Y'], 'tab:orange')
    axs[2, 0].plot(df1['timestamp'], df1[type+'Z'], 'tab:green')
    axs[0, 1].plot(df2['timestamp'], df2[type+'X'])
    axs[1, 1].plot(df2['timestamp'], df2[type+'Y'], 'tab:orange')
    axs[2, 1].plot(df2['timestamp'], df2[type+'Z'], 'tab:green')

    axs[0, 0].set_title(title1)
    axs[0, 1].set_title(title2)

    text1 = min(df1['timestamp']+2)
    text2 = min(df2['timestamp']+2)
    axs[0, 0].text(text1,-max_value,"std="+str(stds1[0]), label='X (Blue)')
    axs[1, 0].text(text1,-max_value,str(stds1[1]), label='Y (Orange)')
    axs[2, 0].text(text1,-max_value,str(stds1[2]), label='Z (Green)')
    axs[0, 1].text(text2,-max_value,str(stds2[0]))
    axs[1, 1].text(text2,-max_value,str(stds2[1]))
    axs[2, 1].text(text2,-max_value,str(stds2[2]))

    for ax in axs.flat:
        ax.label_outer()
        ax.set_ylim([-max_value-1,max_value+1])
        # ax.set_xticks([])
        # for impact in impacts:
        #     ax.axvline(x=impact, color='r', linestyle='--')

    fig.legend(handles=[x_patch, y_patch, z_patch], loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def comparison_time_plot_centered(df1, df2, type, title1, title2):
    # Compute standard deviations
    stds1 = [np.std(df1[i]).round(5) for i in df1.columns[:6]]
    stds2 = [np.std(df2[i]).round(5) for i in df2.columns[:6]]

    # Compute mean values for each acceleration component
    means1 = {axis: np.mean(df1[type + axis]) for axis in ['X', 'Y', 'Z']}
    means2 = {axis: np.mean(df2[type + axis]) for axis in ['X', 'Y', 'Z']}

    # Compute distance from the mean for each acceleration component
    df1_centered = {axis: abs(df1[type + axis] - means1[axis]) for axis in ['X', 'Y', 'Z']}
    df2_centered = {axis: abs(df2[type + axis] - means2[axis]) for axis in ['X', 'Y', 'Z']}

    # Find max absolute value for y-axis limits
    max_value = max(max(map(abs, itertools.chain(*df1_centered.values()))),
                    max(map(abs, itertools.chain(*df2_centered.values()))))

    # Create legend patches
    x_patch = mpatches.Patch(color='tab:blue', label='X')
    y_patch = mpatches.Patch(color='tab:orange', label='Y')
    z_patch = mpatches.Patch(color='tab:green', label='Z')

    # Create subplots
    fig, axs = plt.subplots(3, 2)
    axs[0, 0].plot(df1['timestamp'], df1_centered['X'])
    axs[1, 0].plot(df1['timestamp'], df1_centered['Y'], 'tab:orange')
    axs[2, 0].plot(df1['timestamp'], df1_centered['Z'], 'tab:green')
    axs[0, 1].plot(df2['timestamp'], df2_centered['X'])
    axs[1, 1].plot(df2['timestamp'], df2_centered['Y'], 'tab:orange')
    axs[2, 1].plot(df2['timestamp'], df2_centered['Z'], 'tab:green')

    # Set subplot titles
    axs[0, 0].set_title(title1)
    axs[0, 1].set_title(title2)

    # Text positioning
    text1 = min(df1['timestamp']) + 2
    text2 = min(df2['timestamp']) + 2

    # Add standard deviation text
    axs[0, 0].text(text1, -max_value, "std=" + str(stds1[0]))
    axs[1, 0].text(text1, -max_value, str(stds1[1]))
    axs[2, 0].text(text1, -max_value, str(stds1[2]))
    axs[0, 1].text(text2, -max_value, str(stds2[0]))
    axs[1, 1].text(text2, -max_value, str(stds2[1]))
    axs[2, 1].text(text2, -max_value, str(stds2[2]))

    # Formatting
    for ax in axs.flat:
        ax.label_outer()
        ax.set_ylim([0, max_value + 1])
        ax.set_xticks([])

    # Add legend
    fig.legend(handles=[x_patch, y_patch, z_patch], loc='lower center', ncol=3, fontsize=10)
    plt.show()
