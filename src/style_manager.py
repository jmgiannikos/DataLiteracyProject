import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tueplots import bundles, axes, fonts
from tueplots.constants.color import rgb
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
logger = logging.getLogger(__name__)

def to_hex(color_array, alpha=None):
    """Converts color to hex. If alpha is provided (0-1), adds it to the hex string."""
    hex_color = mcolors.to_hex(color_array)
    if alpha is not None:
        # Convert 0.0-1.0 to 00-FF
        alpha_hex = format(int(alpha * 255), '02x')
        return hex_color + alpha_hex
    return hex_color

def generate_style():
    """Generates a .mplstyle file compliant with Tuebingen's brand identity."""
    
    
    style_dir = Path(__file__).parent.parent / 'styles'
    style_dir.mkdir(exist_ok=True)
    style_path = style_dir / 'tuebingen_style.mplstyle'

    #Get base configuration from tueplots
    base_config = {
        **bundles.beamer_moml(), 
        **axes.lines(),
        **fonts.beamer_moml(),
    }

    #Layout Overrides 
    layout_config = {
        'figure.figsize': '8, 4.5',
        'figure.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.axisbelow': True,
        'font.size': 10,
        'axes.titlesize': 11,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.loc': 'best',
        # Fallback fonts
        'font.sans-serif': 'Roboto Condensed, Arial, Helvetica, sans-serif',
    }

    final_config = {**base_config, **layout_config}

    color_keys = ['text.color', 'axes.labelcolor', 'axes.edgecolor', 
                  'xtick.color', 'ytick.color', 'grid.color', 
                  'legend.edgecolor', 'axes.prop_cycle']

    with open(style_path, 'w') as f:
        for key, value in final_config.items():
            if key not in color_keys:
                f.write(f'{key}: {str(value)}\n')

    print(f" Generated layout stylesheet at: {style_path}")


def apply_style(mode='standard'):
    """
    Applies the Tübingen style with specific overrides.
    Modes: 'standard', 'scatter', 'histogram'
    """
    # Base Layout
    current_file = Path(__file__).resolve()
    # Ensure the path points to where your .mplstyle file actually is
    style_path = current_file.parent.parent / "styles" / "tuebingen_style.mplstyle"
    
    if not style_path.exists():
        print(f"Warning: {style_path} not found. You need to generate it first with generate_style()")
        return
    else:
        plt.style.use(str(style_path))

    #Define the Tübingen Palette 
    tue_raw = [
        rgb.tue_red, rgb.tue_lightblue, rgb.tue_dark, rgb.tue_gray, 
        rgb.tue_orange, rgb.tue_green, rgb.tue_violet, rgb.tue_brown
    ]

    # Create a custom Tübingen Colormap (Light Gray -> Tübingen Red)
    tue_red_cmap = LinearSegmentedColormap.from_list("TueRed", ["#F5F5F5", to_hex(rgb.tue_red)])
    plt.colormaps.register(tue_red_cmap, force=True)

    #Create Mode-Specific Updates
    updates = {
        'text.color': to_hex(rgb.tue_dark),
        'axes.labelcolor': to_hex(rgb.tue_dark),
        'axes.edgecolor': to_hex(rgb.tue_dark),
        'xtick.color': to_hex(rgb.tue_dark),
        'ytick.color': to_hex(rgb.tue_dark),
        'grid.color': to_hex(rgb.tue_gray),
        'legend.edgecolor': to_hex(rgb.tue_gray),
        'grid.alpha': 0.1, 
        'image.cmap': 'TueRed', 
    }

    if mode == 'scatter':
        '''
        This mode is thought for tsne/pca analysis. It can support up to 24 different categories (authors) 
        with different colors/shapes.
        Gives an aplha value = 0.7
        '''
        # Use Alpha for scatter so overlapping points are visible
        tue_hex_alpha = [to_hex(c, alpha=0.7) for c in tue_raw]
        markers = ['o', 's', '^']
        
        # 8 colors * 3 markers = 24 unique combos
        updates['axes.prop_cycle'] = cycler('color', tue_hex_alpha * 3) + \
                                     cycler('marker', [m for m in markers for _ in range(8)])
        updates['lines.markersize'] = 7
        updates['lines.linestyle'] = 'None' 
        
        print("Mode: SCATTER (Markers + Colors + Alpha)")

    elif mode == 'histogram':
        '''
        Thought for histograms.
        Provides white edgecolor, alpha = 0.7
        '''
        # TRICK: Apply alpha (0.7) directly to the Hex codes (e.g., #RRGGBBB3)
        # This bypasses the need for 'patch.alpha'
        tue_hex_alpha = [to_hex(c, alpha=0.7) for c in tue_raw]
        updates['axes.prop_cycle'] = cycler('color', tue_hex_alpha)
        updates['patch.edgecolor'] = 'white'
        updates['patch.linewidth'] = 0.7
        # This is a valid rcParam that ensures white lines show up between bars
        updates['axes.spines.left'] = True 
        updates['patch.force_edgecolor'] = True 
        print("Mode: HISTOGRAM (Alpha via Hex)")

    else: # 'standard'
        tue_hex = [to_hex(c) for c in tue_raw]
        updates['axes.prop_cycle'] = cycler('color', tue_hex)
        updates['lines.marker'] = 'None'
        print("Mode: STANDARD (Lines)")

    # 4. Apply everything
    plt.rcParams.update(updates)
    print(f"Tübingen Style Applied successfully!")

if __name__ == "__main__":
    # Test the implementation, can comment the plot beneath 
    generate_style()
    
    import numpy as np
    apply_style()
    x = np.linspace(0, 10, 100)
    fig, ax = plt.subplots()
    ax.plot(x, np.sin(x), label='Sine Wave (Tue Red)')
    ax.plot(x, np.cos(x), label='Cosine Wave (Tue Blue)')
    ax.set_title("University of Tübingen Styled Plot")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.legend()
    plt.show()