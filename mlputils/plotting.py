import numpy as np
import matplotlib.pyplot as plt


def _pava(z):
    """Pool Adjacent Violators: unweighted isotonic regression (non-decreasing)."""
    levels = []  # list of [value, weight]
    for zi in z:
        levels.append([zi, 1])
        while len(levels) > 1 and levels[-2][0] > levels[-1][0]:
            v2, w2 = levels.pop()
            v1, w1 = levels.pop()
            levels.append([(v1 * w1 + v2 * w2) / (w1 + w2), w1 + w2])
    out = []
    for v, w in levels:
        out.extend([v] * w)
    return np.array(out)


def _declutter(values, min_sep):
    """
    Return y-positions with at least `min_sep` between neighbors (in sorted
    order), staying as close as possible (least squares) to the originals.
    Order of values is preserved; only spacing is adjusted.
    """
    values = np.asarray(values, dtype=float)
    order = np.argsort(values)
    y_sorted = values[order]
    z = y_sorted - np.arange(len(y_sorted)) * min_sep
    z_iso = _pava(z)
    y_adj_sorted = z_iso + np.arange(len(y_sorted)) * min_sep
    y_adj = np.empty_like(y_adj_sorted)
    y_adj[order] = y_adj_sorted
    return y_adj


def plot_energy_correspondence(
    energies_A,
    energies_B,
    labels=None,
    left_label='A',
    right_label='B',
    min_gap_frac=0.018,
    fontsize=9,
    figsize=(8, 10),
    dpi=300,
    cmap_name='tab20',
    inversions=None,
):
    energies_A = np.asarray(energies_A, dtype=float)
    energies_B = np.asarray(energies_B, dtype=float)

    if len(energies_A) != len(energies_B):
        raise ValueError("Arrays must have same length")

    n = len(energies_A)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    x_left, x_right = 0, 1

    # color by isomer identity (rank index), not by raw energy value --
    # a continuous colormap makes close-lying levels nearly indistinguishable
    cmap = plt.get_cmap(cmap_name, n)
    colors = [cmap(i % cmap.N) for i in range(n)]

    # minimum label separation, scaled to the combined data range
    y_all = np.concatenate([energies_A, energies_B])
    y_range = y_all.max() - y_all.min()
    min_sep = min_gap_frac * y_range

    # decluttered label y-positions; the tick marks below stay at true energies
    label_y_A = _declutter(energies_A, min_sep)
    label_y_B = _declutter(energies_B, min_sep)

    for i in range(n):
        ea, eb, c = energies_A[i], energies_B[i], colors[i]

        # level ticks at the true energy
        ax.plot([x_left - 0.05, x_left + 0.05], [ea, ea], lw=2, color=c)
        ax.plot([x_right - 0.05, x_right + 0.05], [eb, eb], lw=2, color=c)

        # connecting line between the true energies
        ax.plot([x_left, x_right], [ea, eb], alpha=0.6, color=c, lw=1)

        if labels is not None:
            # leader line from the true energy to the (possibly shifted) label
            ax.plot([x_left - 0.05, x_left - 0.12], [ea, label_y_A[i]],
                     color=c, lw=0.6, alpha=0.7)
            ax.plot([x_right + 0.05, x_right + 0.12], [eb, label_y_B[i]],
                     color=c, lw=0.6, alpha=0.7)

            ax.text(x_left - 0.14, label_y_A[i], labels[i],
                    ha='right', va='center', fontsize=fontsize)
            ax.text(x_right + 0.14, label_y_B[i], labels[i],
                    ha='left', va='center', fontsize=fontsize)

    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([x_left, x_right])
    ax.set_xticklabels([left_label, right_label])
    ax.set_ylabel('Excess energy (eV)')
    ax.set_title('Energy ordering correspondence')
    ax.grid(axis='y', alpha=0.3)

    if inversions is not None:
        ax.text(
            0.02, 0.98,
            f'{inversions} crossings',
            transform=ax.transAxes,
            ha='left', va='top',
            fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='gray', alpha=0.8),
        )

    plt.tight_layout()
    plt.show()

    return fig, ax