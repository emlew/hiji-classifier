import matplotlib.patches as mpatches

def get_patches():
        
    x_patch = mpatches.Patch(color='tab:blue', label='X')
    y_patch = mpatches.Patch(color='tab:orange', label='Y')
    z_patch = mpatches.Patch(color='tab:green', label='Z')
    
    return x_patch, y_patch, z_patch

def type_is_valid(type):
    if type is "accel" or type is "rate":
        return True
    return False