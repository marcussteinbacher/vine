import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Literal
import json

class Theme:
    """
    Static class that holds different plotting themes.
    Theme can be set with 'Theme.set_theme(theme)'
    Themes are:
    - 'scientific': Matches te Latex style in scientific journals, default
    - 'solarized_dark': Matches the corresponding VS Code theme
    - 'solarized_light': Matches the corresponding VS Code theme 

    Example:
    from tools.Plotting import Theme

    Theme.set_theme('solarized_light')

    plt.plot(...)

    """
    solarized_dark = rc={"axes.edgecolor": '#B7B7B7',
                        "axes.facecolor": "#00212B",
                         "axes.grid": True, 
                         "axes.labelcolor": "#B7B7B7",
                         "axes.linewidth": 0.75,
                         'figure.facecolor': "#002B36",
                         'grid.color': "#B7B7B7",
                         "grid.linestyle": ":",
                         "grid.linewidth": .5,
                         "legend.facecolor": "#1F3744",
                         "text.color": "#B7B7B7",
                         "xtick.color": "#B7B7B7",
                         "ytick.color": "#B7B7B7",
                         }
    solarized_light = rc={'axes.facecolor': "#F7F0E0",
                         "axes.grid": True, 
                         "axes.labelcolor": "#7F7F7F",
                         "axes.linewidth": 0.75,
                         'figure.facecolor': "#FDF6E3",
                         'grid.color': "#7F7F7F",
                         "grid.linestyle": ":",
                         "grid.linewidth": .5,
                         'axes.edgecolor': '#7F7F7F',
                         "legend.facecolor": "#EEE8D5",
                         "text.color": "#7F7F7F",
                         "xtick.color": "#7F7F7F",
                         "ytick.color": "#7F7F7F",
                         }
    scientific = {"axes.grid": True,"grid.linestyle":":","grid.color":".5","legend.facecolor":"#ffffff"}

    @classmethod
    def _set_solarized_dark(cls):
        sns.set_theme(style="ticks", rc=cls.solarized_dark,palette="bright")

    @classmethod
    def _set_solarized_light(cls):
        sns.set_theme(style="ticks",rc=cls.solarized_light)

    @classmethod
    def _set_scientific(cls):
        sns.set_theme(style='ticks',font='STIXGeneral', rc=cls.scientific)

    @classmethod
    def set_theme(cls, theme:Literal["auto","scientific","solarized_light","solarized_dark"]="auto"):
        """
        Sets the plotting theme.
        Arguments:
        - 'theme': 
            - 'auto', automatically detect the workbench theme, default
            - 'scientific', fallback if 'auto' can't detect the current VS Code theme
            - 'solarized_dark', force Solarized Dark theme
            - 'solarized_light', force Solarized Light theme
        """

        match theme:
            case "auto": 
                implemented_themes = {
                    "Solarized Light":cls._set_solarized_light, 
                    "Solarized Dark":cls._set_solarized_dark
                    }

                with open('.vscode/settings.json', 'r') as file:
                    settings = json.load(file)

                if "workbench.colorTheme" in settings.keys():
                    vstheme = settings["workbench.colorTheme"]
                    try:
                        implemented_themes[vstheme]()
                    except KeyError as e:
                        raise NotImplementedError(f"Theme {vstheme} not implemented! Currently implemented are: {implemented_themes.keys()}")
                else:
                    # Fallback to default seaborn style
                    sns.set_style()

            case "scientific":
                cls._set_scientific()
            case "solarized_dark":
                cls._set_solarized_dark()
            case "solarized_light":
                cls._set_solarized_light()
            case _:
                raise NotImplementedError
        

def default_color_generator():
    """
    Endlessly cycles through the default colors. See plt.rcParams['axes.prop_cycle'].
    
    Example:
    
    gen = default_color_generator() \n
    fig, ax = plt.subplots() \n
    [...] \n
    for i in range(10): \n
        ax.plot(..., color = next(gen)) \n
    [...]
    """
    i = 0
    while True:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        yield colors[i%len(colors)]
        i+=1
    
