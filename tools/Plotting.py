import matplotlib.pyplot as plt
import seaborn as sns 
from typing import Literal
import json
import networkx as nx
from tools.Graphs import get_node_labels, get_edge_labels, make_graph_network

class Theme:
    """
    Static class that holds different plotting themes.
    Theme can be set with 'Theme.set_theme(theme)'
    Themes are:
    - 'scientific': Matches te Latex style in scientific journals, default
    - 'solarized_dark': Matches the corresponding VS Code theme
    - 'solarized_light': Matches the corresponding VS Code theme 

    Example:
    ```python
    from tools.Plotting import Theme
    Theme.set_theme('solarized_light')
    plt.plot(...)
    ```
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
    ```python
    gen = default_color_generator() \n
    fig, ax = plt.subplots() \n
    [...] \n
    for i in range(10): \n
        ax.plot(..., color = next(gen)) \n
    [...]
    ```
    """
    i = 0
    while True:
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        yield colors[i%len(colors)]
        i+=1
    

def plot_vine(vine, trees:list[int],ax, layout="graphviz",edge_labels=True):
    """
    Plots the specified trees of a vine. 
    Arguments:
    - vine: Vine object
    - trees: List of tree indices to plot, e.g. [0,1,2]
    - ax: Matplotlib axis to plot on
    - layout: Layout for the graph, default "graphviz". Options: "graphviz" or "spring_layout". See networkx.draw_networkx for more information.
    - edge_labels: Whether to display edge labels w/ copula info, default True.
    """

    G = make_graph_network(vine, trees)
    node_labels = get_node_labels(G)
    edge_labels = get_edge_labels(G)

    match layout:
        case "graphviz":
            pos = nx.drawing.nx_pydot.graphviz_layout(G, prog="dot") # dot, twopi, fdp, sfdp, circo
        case "spring_layout":
            pos = nx.spring_layout(G)
        case _ as e:
            raise NotImplementedError(f"Layout {e} not implemented!")
        
    for node, data in G.nodes(data=True):
        tree = G.nodes[node]["tree"]
        shape = "s" if tree == 0 else "o"
        nx.draw_networkx_nodes(G,pos,nodelist=[node],node_shape=shape,ax=ax,node_size=750)

    nx.draw_networkx_labels(G,pos,labels=node_labels,ax=ax)
    nx.draw_networkx_edges(G,pos,ax=ax)
    if edge_labels:
        nx.draw_networkx_edge_labels(G,pos,edge_labels=edge_labels,ax=ax,bbox=dict(boxstyle="round", fc="0.8", ec="0.5", alpha=0.75))
