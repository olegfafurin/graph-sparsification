#
# Created by Oleg Fafurin
#

import os
import random

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import re

index_re = re.compile("^Approx_(\d+)_SPTs$")


def index_extract(line):
    m = index_re.fullmatch(line)
    if m is not None:
        return int(m[1])
    return 0


### To apply to real graphs to learn when to stop adding SPTs
def plot_edges_ari(filename, title, algo="Louvain", weighted=True):
    df = pd.read_csv(filename, sep=' ', header=None, names=['graph', 'vs', 'algo', 'ari', 'iter', 'edges', 'weighted'])
    df["n_trees"] = df['graph'].apply(index_extract)

    fig, ax1 = plt.subplots(figsize=(10, 6))  # Create a figure with one set of axes

    plot_reference_lines(df[df['algo'] == algo], ax1)

    df = df[(df['algo'] == algo) & (df['vs'] == "Meta") & (df['weighted'] == int(weighted))]

    df_diff = pd.DataFrame({'n_trees': pd.Series(dtype='int'),
                            'e_diff': pd.Series(dtype='int'),
                            'ari': pd.Series(dtype='float'),
                            'iter': pd.Series(dtype='int')})

    for i in range(df['iter'].max() + 1):
        df_approx = df[(df['iter'] == i) & (df['n_trees'] > 0)]
        df_approx['e_diff'] = df_approx['edges'].diff().fillna(df_approx['edges'])
        df_diff = pd.concat([df_diff, df_approx[['n_trees', 'e_diff', 'ari', 'iter']]])

    group_stats = df_diff[df_diff['n_trees'] > 0].groupby('n_trees')[['ari', 'e_diff']].agg(
        ['mean', 'std']).reset_index()

    ax1.set_xlabel('iteration')
    ax1.set_ylabel('ARI (mean)', color='#d62728')
    ax1.tick_params(axis='y', labelcolor='#d62728')

    # ax1.errorbar(
    #     group_stats['n_trees'],
    #     group_stats['ari']['mean'],
    #     group_stats['ari']['std'],
    #     fmt='.-',
    #     color='#d62728',
    #     label='ARI of approximation'
    # )

    ax1.plot(
        group_stats['n_trees'],  # X values (group)
        group_stats['ari']['mean'],  # Y values (mean)
        label='ARI of approximation',
        color='#d62728',
        markersize=1,
    )

    ax2 = ax1.twinx()  # Create a twin Y-axis sharing the same X-axis
    ax2.set_ylabel('Number of edges added at step', color='#1f77b4')
    ax2.tick_params(axis='y', labelcolor='#1f77b4')

    # ax2.errorbar(
    #     group_stats['n_trees'],
    #     group_stats['e_diff']['mean'],
    #     group_stats['e_diff']['std'],
    #     fmt='.-',
    #     color='#1f77b4',
    #     label='Edges of SPT added at iteration'
    # )

    ax2.plot(
        group_stats['n_trees'],  # X values (group)
        group_stats['e_diff']['mean'],  # Y values (mean)
        label='Edges of SPT added at iteration',
        color='#1f77b4',
        markersize=1,
    )


    ls1, labels1 = ax1.get_legend_handles_labels()
    ls2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(ls1 + ls2, labels1 + labels2, loc="right")

    plt.title(f"Number of SPTs and convergence, {title}")
    plt.xlabel("Number of SPTs")

    plt.savefig(f"{filename[:filename.rfind(".")]}-edges-convergence1.png")


def plot_heuristicts(dirname, filenames=None, algo='Louvain'):
    heuristics = ("random", "min_deg", "furthest")
    if filenames is None:
        filenames = tuple(f"approx_MB_ari-{s}-5-iter.txt" for s in heuristics)
    fig, ax = plt.subplots(figsize=(9, 6))
    for (filename, h) in zip(filenames, heuristics):
        df = pd.read_csv(f"{dirname}/{filename}", sep=' ', header=None,
                         names=['graph', 'vs', 'algo', 'ari', 'iter', 'edges', 'weighted'])
        df = df[df['algo'] == algo]
        df["n_trees"] = df['graph'].apply(index_extract)
        draw_mean_with_errors(df[(df['vs'] == "Meta") & (df['weighted'] == 1)], label=f'{h}-weighted')
        draw_mean_with_errors(df[(df['vs'] == "Meta") & (df['weighted'] == 0)], label=f'{h}-unweighted')

    plot_reference_lines(df, ax)
    plt.title("Heuristics for root selection and partition quality convergence\nPrimary school dataset, $n=242$, $m=8559$")
    plt.legend(loc="lower right")
    plt.savefig(f"{dirname}/{filenames[0][:filenames[0].rfind(".")]}-heuristics-{algo}1.png", dpi=100)
    plt.clf()


def plot_approx(filename, algo='Louvain', weighted=True):
    df = pd.read_csv(filename, sep=' ', header=None, names=['graph', 'vs', 'algo', 'ari', 'iter', 'edges'])
    df["n_trees"] = df['graph'].apply(index_extract)

    fig, ax = plt.subplots(figsize=(12, 6))

    plot_reference_lines(df[(df['algo'] == algo) & (df['vs'] == "Meta")], ax)
    plot_single_line(df[(df['algo'] == algo) & (df['vs'] == "Meta") & (df['weighted'] == int(weighted))])
    plt.savefig(f"{filename[:filename.rfind(".")]}-single-{algo}.png", dpi=100)
    plt.clf()

    plot_reference_lines(df[(df['algo'] == algo) & (df['vs'] == "Meta")], ax)
    draw_mean_with_errors(df[(df['algo'] == algo) & (df['vs'] == "Meta") & (df['weighted'] == int(weighted))])
    plt.savefig(f"{filename[:filename.rfind(".")]}-mean-{algo}.png", dpi=100)
    plt.clf()


def plot_reference_lines(df, ax, labels=None, lines=None, plot_std=True):
    if labels is None:
        labels = {'orig': 'Original Graph', 'unweighted': 'Unweighted Graph', 'mb': 'Metric Backbone'}
    if lines is None:
        lines = {'orig', 'unweighted', 'mb'}
    n_trees_max = df['n_trees'].max()
    if 'orig' in lines:
        perf_d = df[(df['vs'] == 'Meta') & (df['graph'] == 'Original')]['ari']
        perf_d_mean = perf_d.mean()
        perf_d_std = perf_d.std()
        ax.axhline(y=perf_d_mean, color='r', linestyle=':', linewidth=2, alpha=0.6, label=labels['orig'])
        if plot_std:
            ax.fill_between(list(range(n_trees_max)), perf_d_mean - perf_d_std, perf_d_mean + perf_d_std,
                        color='red', alpha=0.15)
    if 'mb' in lines:
        perf_b = df[(df['vs'] == 'Meta') & (df['graph'] == 'Backbone')]['ari']
        perf_b_mean = perf_b.mean()
        perf_b_std = perf_b.std()

        plt.axhline(y=perf_b.mean(), color='orange', linestyle='--', linewidth=2, alpha=0.6, label=labels['mb'])
        if plot_std:
            plt.fill_between(list(range(n_trees_max)), perf_b_mean - perf_b_std, perf_b_mean + perf_b_std,
                         color='orange', alpha=0.15)
    if 'unweighted' in lines:
        perf_g = df[(df['vs'] == 'Meta') & (df['graph'] == 'Unweighted')]['ari']
        perf_g_mean = perf_g.mean()
        perf_g_std = perf_g.std()
        plt.axhline(y=perf_g_mean, color='grey', linestyle='-.', linewidth=2, alpha=0.8, label=labels['unweighted'])
        if plot_std:
            plt.fill_between(list(range(n_trees_max)), perf_g_mean - perf_g_std, perf_g_mean + perf_g_std,
                         color='grey', alpha=0.25)


def plot_single_line(df):
    for i_iter in range(df['iter'].max() + 1):
        df_local = df[
            (df['iter'] == i_iter) & (df['n_trees'] > 0)]
        plt.plot(df_local['n_trees'], df_local['ari'])

    plt.xlabel('Number of SPTs')
    plt.ylabel('ARI')
    # plt.xscale('log')
    plt.legend()
    # plt.savefig("../tmp/single-line.png")


def draw_mean_with_errors(df, label="Approximate Metric Backbone (mean ± std)"):
    group_stats = df[df['n_trees'] > 0].groupby('n_trees')['ari'].agg(['mean', 'std']).reset_index()

    plt.errorbar(
        group_stats['n_trees'],  # X values (group)
        group_stats['mean'],  # Y values (mean)
        yerr=group_stats['std'],  # Error bars (std)
        fmt='.-',  # Line and marker style
        capsize=0,  # Capsize for error bars
        label=label,
    )

    # plt.plot(
    #     group_stats['n_trees'],  # X values (group)
    #     group_stats['mean'],  # Y values (mean)
    #     label=label,
    #     markersize=1,
    # )

    # plt.title('Mean ARI')
    plt.xlabel('iteration')
    plt.ylabel('ARI (mean $\\pm$ std)')
    plt.legend()
    # plt.savefig("../tmp/mean.png")


def plot_time():
    dfs = list()
    for n in [5000, 10000, 20000, 30000]:
        df = pd.read_csv(f"../Datasets/abcd-larger-deg/n={n}/edge_d_time_mb.csv", sep=',').join(pd.DataFrame({"n": [n]}))
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    for label in ['t_apsp','t_ksl','t_1_and_apsp','t_dijkstra_log2','t_fb_log2']:
        df[label] /= 10**6
    df.plot(kind="bar", x="n", logy=True, ylabel= "Time, s")
    plt.savefig("../tmp/time-MB-construction.png")


def plot_xi(dirname, xis, algo='Louvain'):
    fig, ax = plt.subplots(figsize=(9, 6))
    for xi in xis:
        df = pd.read_csv(f"{dirname}/ari_xi={xi}-random-5-iter.txt", sep=' ', header=None,
                         names=['graph', 'vs', 'algo', 'ari', 'iter', 'edges', 'weighted'])
        df = df[df['algo'] == algo]
        df["n_trees"] = df['graph'].apply(index_extract)
        df=df[df['n_trees'] < 60]
        draw_mean_with_errors(df[(df['vs'] == "Meta") & (df['weighted'] == 1)], label=f'Approximation, $\\xi={xi}$')
        plot_reference_lines(df, ax, labels={"orig": f"Original graph, $\\xi={xi}$",
                                             "mb": f"Metric Backbone, $\\xi={xi}$"},
                             lines={"mb"})
    plt.title("Dependency of partition quality convergence from community strength\n$n=10^4$, $k=10$")
    plt.legend(loc="lower right")
    # ax.axhline(y=0, color='k', alpha=0.5)
    # ax.axvline(x=0, color='k', alpha=0.5)
    plt.savefig(f"{dirname}/convergence-vs-xi-{algo}1.png", dpi=100)
    plt.clf()


def plot_k(dirname, ks, algo='Louvain'):
    fig, ax = plt.subplots(figsize=(9, 6))
    for k in ks:
        df = pd.read_csv(f"{dirname}/ari_k={k}-random-5-iter.txt", sep=' ', header=None,
                         names=['graph', 'vs', 'algo', 'ari', 'iter', 'edges', 'weighted'])
        df = df[df['algo'] == algo]
        df["n_trees"] = df['graph'].apply(index_extract)
        df=df[df['n_trees'] < 60]
        draw_mean_with_errors(df[(df['vs'] == "Meta") & (df['weighted'] == 1)], label=f'Approximation, $k={k}$')
        plot_reference_lines(df, ax, labels={"orig": f"Original graph, $k={k}$",
                                             "mb": f"Metric Backbone, $k={k}$"},
                             lines={"mb"},
                             plot_std=False)
    plt.title("Dependency of partition quality convergence from number of communities\nABCD graph, $n=10^4$, $\\xi=0.2$")
    plt.legend(loc="lower right")
    # ax.axhline(y=0, color='k', alpha=0.5)
    # ax.axvline(x=0, color='k', alpha=0.5)
    plt.savefig(f"{dirname}/convergence-vs-k-{algo}1.png", dpi=100)
    plt.clf()



if __name__ == '__main__':

    # plot_xi(dirname="../Datasets/abcd-n-spt/vs_xi/non-regular_n=10000_k=10", xis=[0.1, 0.3, 0.5])
    # plot_k(dirname="../Datasets/abcd-n-spt/vs_k/n=10000,xi=0.2", ks=[2,5,10,100])



    # plot_edges_ari("../Datasets/abcd-larger-deg/n=10000/approx_MB_ari-random-5-iter.txt", "\nABCD graph, $k=22$, $n=10^4$")
    plot_edges_ari("../tmp/amazon_approx-random-5-iter.txt", title="Amazon dataset, $n=8035$, $m=191698$")
    # plot_edges_ari("../tmp/DBLP_approx-random-5-iter.txt")
    # plot_edges_ari("../tmp/DBLP_approx-min_deg-5-iter.txt", title="DBLP dataset, min degree SPT root heuristic")
    # plot_edges_ari("../tmp/high_school_approx-random-5-iter.txt")

    # plot_edges_ari("../Datasets/abcd-larger-deg/n=30000/approx_MB_ari-random-5-iter.txt")
    # plot_edges_ari("../Datasets/abcd-larger-deg/n=10000/approx_MB_ari-random-5-iter.txt")
    # plot_edges_ari("../Datasets/abcd-larger-deg/n=20000/approx_MB_ari-random-5-iter.txt")
    # plot_approx('../Datasets/abcd-larger-deg/n=30000/approx_MB_ari-furthest-5-iter.txt', algo='Louvain')
    # plot_approx('../Datasets/abcd-larger-deg/n=30000/approx_MB_ari-min_deg-5-iter.txt', algo='Louvain')
    # plot_approx('../Datasets/abcd-larger-deg/n=30000/approx_MB_ari-random-5-iter.txt', algo='Louvain')

    # plot_heuristicts("../tmp", filenames=["amazon_approx-random-5-iter.txt", "amazon_approx-min_deg-5-iter.txt", "amazon_approx-furthest-5-iter.txt"])
    # plot_heuristicts("../tmp", filenames=["DBLP_approx-random-5-iter.txt", "DBLP_approx-min_deg-5-iter.txt", "DBLP_approx-furthest-5-iter.txt"])
    # plot_heuristicts("../tmp", filenames=["high_school_approx1-random-5-iter.txt", "high_school_approx1-min_deg-5-iter.txt", "high_school_approx1-furthest-5-iter.txt"])
    # plot_heuristicts("../tmp", filenames=["primary_school_approx1-random-5-iter.txt", "primary_school_approx1-min_deg-5-iter.txt", "primary_school_approx1-furthest-5-iter.txt"])
    # plot_heuristicts("../tmp", filenames=["high_school_approx1-random-5-iter.txt", "high_school_approx1-min_deg-5-iter.txt", "high_school_approx1-furthest-5-iter.txt"], algo="Leiden")
    # plot_heuristicts("../tmp", filenames=["primary_school_approx1-random-5-iter.txt", "primary_school_approx1-min_deg-5-iter.txt", "primary_school_approx1-furthest-5-iter.txt"], algo="Leiden")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=5000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=10000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=20000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=30000", algo="Leiden")

    # plot_heuristicts("../Datasets/abcd-larger-deg/n=5000")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=5000")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=20000")
    # plot_heuristicts("../Datasets/abcd-larger-deg/n=30000")

    # plot_heuristicts("../Datasets/abcd/n=1000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd/n=2000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd/n=5000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd/n=10000", algo="Leiden")
    # plot_heuristicts("../Datasets/abcd/n=20000", algo="Leiden")

    # plot_heuristicts("../Datasets/abcd/n=1000", filenames=["approx_MB_ari-furthest-5-iter.txt"])
    # plot_heuristicts("../Datasets/abcd/n=2000")
    # plot_heuristicts("../Datasets/abcd/n=5000")
    # plot_heuristicts("../Datasets/abcd/n=10000")
    # plot_heuristicts("../Datasets/abcd/n=20000")

    # plt.savefig("../tmp/mean.png")
    # plot_approx('../Datasets/abcd-larger-deg/n=30000/approx_MB_ari.txt', algo='Leiden')

    # plot_approx('../Datasets/abcd/n=20000/approx_MB_ari-random-5-iter.txt', algo='Louvain')
    # plot_approx('../Datasets/abcd/n=1000/approx_MB_ari.txt', algo='Leiden')

    # plot_approx("../tmp/high_school_approx.txt", algo='Louvain')
    # plot_approx("../tmp/high_school_approx.txt", algo='Leiden')

    # plot_approx("../tmp/primary_school_approx.txt", algo='Louvain')
    # plot_approx("../tmp/primary_school_approx.txt", algo='Leiden')

    # plot_approx("../tmp/amazon_approx.txt", algo='Louvain')
    # plot_approx("../tmp/amazon_approx.txt", algo='Leiden')

    # plot_approx("../tmp/DBLP_approx.txt", algo='Louvain')
    # plot_approx("../tmp/DBLP_approx.txt", algo='Leiden')

    # plt.title("Louvain")
    # draw_mean_with_errors(df[df['algo'] == 'Louvain'])
    # plt.savefig("../tmp/fig2.png")
    # plt.clf()
    # plt.title("Leiden")
    # draw_mean_with_errors(df[df['algo'] == 'Leiden'])
    # plt.savefig("../tmp/fig3.png")
