def plot_metric_vs_n(
    results,
    sigma,
    metric_attr,
    error_attr,
    ylabel,
    title,
    output_path=None,
):
    import matplotlib.pyplot as plt

    sigma_results = sorted((row for row in results if row.sigma == sigma), key=lambda row: row.n)
    if not sigma_results:
        raise ValueError(f"No results found for sigma={sigma}.")

    ns = [row.n for row in sigma_results]
    values = [getattr(row, metric_attr) for row in sigma_results]
    errors = [getattr(row, error_attr) for row in sigma_results]

    fig, ax = plt.subplots()
    ax.errorbar(ns, values, yerr=errors, marker="o", linestyle="-", capsize=4)
    ax.set_xlabel("n")
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if output_path is not None:
        fig.savefig(output_path, bbox_inches="tight")

    return fig, ax


def plot_excess_risk_vs_n(results, sigma, output_path=None):
    return plot_metric_vs_n(
        results=results,
        sigma=sigma,
        metric_attr="estimated_excess_risk",
        error_attr="loss_std",
        ylabel="Estimated expected excess risk",
        title=f"Estimated excess risk vs n (sigma={sigma})",
        output_path=output_path,
    )


def plot_classification_error_vs_n(results, sigma, output_path=None):
    return plot_metric_vs_n(
        results=results,
        sigma=sigma,
        metric_attr="error_mean",
        error_attr="error_std",
        ylabel="Estimated expected classification error",
        title=f"Classification error vs n (sigma={sigma})",
        output_path=output_path,
    )
