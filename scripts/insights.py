import itertools
from matplotlib import pyplot as plt
import numpy as np

from scripts.utils import get_patches, type_is_valid

def plot_vs_time(df,type,title):
    """
    Create and show plot of all axes of either linear or angular acceleration
    
    Parameters:
    - df: DataFrame with time-series data
    - type: either rate or accel, specifies which type of acceleration to display
    - title: title to be displayed on graph
    """
    if not type_is_valid(type):
        print("Incorrect specification of type. Please use either rate or accel.")
        return
    
    stds = [np.std(df[i]).round(5) for i in df.columns[:6]]
    max_value = max(map(abs, list(itertools.chain(df[type+'X'], df[type+'Y'], df[type+'Z']))))
    start = min(df['timestamp']+2)
    
    x_patch, y_patch, z_patch = get_patches()

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
    return
    
def comparison_time_plot(df1,df2,type,title1,title2):
    """
    Create and show plot of all axes of either linear or angular acceleration for two datasets
    
    Parameters:
    - df1: DataFrame with time-series data
    - df2: second DataFrame with time-series data
    - type: either rate or accel, specifies which type of acceleration to display
    - title1: title to be displayed above first series of data
    - title2: title to be displayed above second series of data
    """
    if not type_is_valid(type):
        print("Incorrect specification of type. Please use either rate or accel.")
        return
    
    stds1 = [np.std(df1[i]).round(5) for i in df1.columns[:6]]
    stds2 = [np.std(df2[i]).round(5) for i in df2.columns[:6]]
        
    max_value = max(max(map(abs, list(itertools.chain(df1[type+'X'], df1[type+'Y'], df1[type+'Z'])))), 
                    max(map(abs, list(itertools.chain(df2[type+'X'], df2[type+'Y'], df2[type+'Z'])))))  
     
    x_patch, y_patch, z_patch = get_patches()

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

    fig.legend(handles=[x_patch, y_patch, z_patch], loc='lower center', ncol=3, fontsize=10)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

def comparison_time_plot_centered(df1, df2, type, title1, title2):
    """
    Create and show standardized plot of all axes of either linear or angular acceleration for two datasets
    
    Parameters:
    - df1: DataFrame with time-series data
    - df2: second DataFrame with time-series data
    - type: either rate or accel, specifies which type of acceleration to display
    - title1: title to be displayed above first series of data
    - title2: title to be displayed above second series of data
    """
    if not type_is_valid(type):
        print("Incorrect specification of type. Please use either rate or accel.")
        return
    
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
    x_patch, y_patch, z_patch = get_patches()

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
