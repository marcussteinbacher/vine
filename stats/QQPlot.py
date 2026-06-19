import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tools.Plotting import default_color_generator

def qq_plot_samples(
    reference: np.ndarray,
    samples: dict[str, np.ndarray],
    ax: plt.Axes,
    colors: list[str] = None,
    quantile_range: tuple[float, float] = (0.01, 0.99),
    n_quantiles: int = 100,
    band: bool = False,
    prob_ticks: list[float] | None = None,
    prob_labels: bool = True,
    plot_stats: list[str] = [], #
) -> plt.Axes:
    """
    Q-Q plot comparing one or more samples against a reference sample.
 
    Parameters
    ----------
    reference : np.ndarray
        1-D array of reference observations (e.g. in-sample data or
        historical simulation losses).
    samples : dict[str, np.ndarray]
        Mapping of label -> 1-D array of observations to compare against
        the reference (e.g. vine copula simulated losses per specification).
    ax : plt.Axes
        Existing axes to draw on.
    colors : list[str]
        List of colors to use for each sample. Should be same length as
        number of samples. Default None uses matplotlib defaults.
    quantile_range : (float, float)
        Lower and upper probability bounds for the quantile grid.
        Default (0.01, 0.99) avoids extreme quantile instability.
        Use (0.001, 0.05) to focus on the lower tail only.
    n_quantiles : int
        Number of evenly spaced quantile levels to evaluate.
        Default 100.
    prob_ticks : list[float] or None
        Probability levels at which to draw tick marks on a secondary
        x-axis (top). If None, a sensible default grid is chosen based
        on quantile_range. Pass an empty list [] to suppress the
        secondary axis entirely.
    prob_labels : bool
        Whether to show probability labels on the secondary x-axis ticks.
        Default True.
    plot_stats : list[str]
        List of additional statistics to plot across samples. Supported
        values are "mean", "median", "min", "max". These are computed
    
    Returns
    -------
    ax : plt.Axes
    """
    probs = np.linspace(quantile_range[0], quantile_range[1], n_quantiles)
    ref_quantiles = np.quantile(reference, probs)
 
 
    # 45-degree reference line spanning the full quantile range
    lims = [ref_quantiles[0], ref_quantiles[-1]]
    ax.plot(lims, lims, color="black", linewidth=1,
            linestyle="--", zorder=5, label="45° line")
    ax.grid(False)
    ax.set_zorder(5)  # Put main plot above the grid
    ax.patch.set_alpha(0) # Make axes background transparent to show grid lines

    # Pointwise 95% confidence band around the 45° line.
    # Uses the DKW inequality: the band half-width in probability space is
    # epsilon = sqrt(log(2/alpha) / (2*n)), then converted to quantile space
    # by re-querying the reference empirical CDF.
    if band:
        n_ref = len(reference)
        eps = np.sqrt(np.log(2.0 / 0.05) / (2 * n_ref))  # DKW at 95%
        band_lo = np.quantile(reference, np.clip(probs - eps, 0, 1))
        band_hi = np.quantile(reference, np.clip(probs + eps, 0, 1))
        ax.fill_between(ref_quantiles, band_lo, band_hi,
                        alpha=0.12, color="black", zorder=0,
                        label="95% pointwise band (ref)")
    
    # Plot each sample
    if colors is None:
        gen = default_color_generator()
        colors = [next(gen) for _ in range(len(samples))]


    all_quantiles = []
    for (label, sample), color in zip(samples.items(), colors):
        sample_quantiles = np.quantile(sample, probs)
        all_quantiles.append(sample_quantiles)
        ax.plot(ref_quantiles, sample_quantiles, linewidth=0.75, color=color, label=label, zorder=2)
        
    if len(plot_stats)>0:
        all_quantiles = np.array(all_quantiles)
        mean_quantiles = np.mean(all_quantiles, axis=0)
        median_quantiles = np.median(all_quantiles, axis=0)
        min_quantiles = np.min(all_quantiles, axis=0)
        max_quantiles = np.max(all_quantiles, axis=0)

        if "mean" in plot_stats:
            ax.plot(ref_quantiles, mean_quantiles, color="orange", linestyle="--", label="Mean quantiles", zorder=3)
        if "median" in plot_stats:
            ax.plot(ref_quantiles, median_quantiles, color="blue", linestyle="-.", label="Median quantiles", zorder=3)
        if "min" in plot_stats:
            ax.plot(ref_quantiles, min_quantiles, color="red", linestyle=":", label="Min quantiles", zorder=3)
        if "max" in plot_stats:
            ax.plot(ref_quantiles, max_quantiles, color="red", linestyle=":", label="Max quantiles", zorder=3)
    
    # Secondary x-axis showing probability levels
    if prob_ticks is None:
        # Choose a sensible default grid clipped to quantile_range
        candidates = [
            0.001, 0.005, 0.01, 0.02, 0.05,
            0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
        ]
        prob_ticks = [p for p in candidates
                      if quantile_range[0] <= p <= quantile_range[1]]
 
    if prob_ticks:
        # Interpolate quantile values at the desired probability ticks
        tick_positions = np.interp(prob_ticks, probs, ref_quantiles)
        tick_labels = [
            f"{p:.1%}" if p >= 0.01 else f"{p:.2%}"
            for p in prob_ticks
        ]
 
        ax2 = ax.twiny()
        ax2.set_zorder(0)  # Put secondary axis behind the main plot
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(tick_positions)

        ax2.set_xticklabels(tick_labels, fontsize=10, rotation=90)
        ax2.set_xlabel("Probability (reference)", fontsize=10, labelpad=8)
    
    if not prob_labels:
        ax2.set_xlabel("")
        ax2.set_xticklabels([])
        ax2.tick_params(axis="x", length=0)  # Hide tick marks if no labels
    
    return ax