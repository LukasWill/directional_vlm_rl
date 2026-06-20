import matplotlib.pyplot as plt
import os
import json
import pandas as pd
import numpy as np
from scipy.stats import sem
from dataclasses import dataclass
from typing import Union
import matplotlib as mpl


@dataclass
class Smoothing:
    method : str
    param : Union[int, float]

    def __eq__(self, value):
        if isinstance(value, Smoothing):
            return value.method == self.method
        elif isinstance(value, str):
            return value == self.method


def figure(figsize=(7, 12), style = {}, subplots=(1, 1)):
    fig, axs = plt.subplots(subplots[0], subplots[1])
    fig.set_figheight(figsize[0])
    fig.set_figwidth(figsize[1])
    if sum(subplots) > 2:
        for ax in axs:
            ax.tick_params(labelfontfamily=style.get("fontname"), labelsize=style.get("fontsize") - 2)
    else:
        axs.tick_params(labelfontfamily=style.get("fontname"), labelsize=style.get("fontsize") - 2)

    fig.tight_layout()
    return fig, axs


def running_avg(values, k):
    return [np.mean(values[ max(0, t-k+1) : t+1 ]) for t in range(len(values))]


def step_filter(steps, stepmin = None, stepmax = None):
    step_filter = np.ones(len(steps), dtype=bool)
    if stepmin is not None:
        step_filter = np.logical_and(step_filter, np.asarray(steps) >= stepmin)
    if stepmax is not None:
        step_filter = np.logical_and(step_filter, np.asarray(steps) <= stepmax)
    
    return step_filter


def plot_mean(steps, value_lists, ax = None, label = None, style = {}):
    if ax is None:
        plt.plot(steps, np.nanmean(value_lists, axis=0), label=label, **style)
    else:
        ax.plot(steps, np.nanmean(value_lists, axis=0), label=label, **style)


def plot_var(steps, value_lists, var="minmax", ax = None, style = {}):
    means = np.nanmean(value_lists, axis=0)

    if var == "minmax":
        bot = np.min(value_lists, axis=0)
        top = np.max(value_lists, axis=0)
    elif var == "std":
        stds = np.nanstd(value_lists, axis=0)
        bot = means - stds
        top = means + stds
    elif var == "err":
        errors = sem(value_lists, axis=0)
        bot = means - errors
        top = means + errors
    
    if ax is None:
        plt.fill_between(steps, top, bot, alpha=0.1, **style)
    else:
        ax.fill_between(steps, top, bot, alpha=0.1, **style)

linestyles = [
    "solid",
    "dashed",
    "dashdot",
    (0, (3, 5, 1, 5, 1, 5)),
    "dotted"
]

def plot_group(
        env, palg, dalg, group, lam = None, beta = None, mult_norms = None, baseline = None, not_baseline = None, rho = None,
        metric = "eval_success", 
        var = "minmax", 
        smooth : Smoothing = None,
        stepmin = None,
        stepmax = None,
        ax = None,
        label = None,
        style = {}):
    step_lists = []
    value_lists = []

    collected_seeds = []

    path = f"runs/{env}/p{palg}/d{dalg}/{group}/"
    for filename in os.listdir(path):        
        with open(path + filename, "r") as f:
            run_dict = json.load(f)
            if lam is not None and run_dict["lam"] != lam:
                continue
            if beta is not None and run_dict["beta"] != beta:
                continue
            if mult_norms is not None and mult_norms != run_dict["mult_norms"]:
                continue
            if baseline is not None and baseline != run_dict["baseline"]:
                continue
            if not_baseline is not None and not_baseline == run_dict["baseline"]:
                continue
            if rho is not None and rho != run_dict["rho"]:
                continue

            seed = int(filename.split("s")[-1].split("_")[0])
            if seed in collected_seeds:
                continue
            else:
                collected_seeds.append(seed)

            data = pd.DataFrame(run_dict["data"])
            steps = data['step']
            filter = step_filter(steps, stepmin, stepmax)
            step_lists.append(steps[filter])
            value_lists.append(data[metric][filter])
    
    print([len(values) for values in value_lists])

    if metric == "success_cnt":
        steps = np.arange(0, 502500, 2500) // 500
        for i in range(len(value_lists)):
            extended_values = np.ones(201) * 50
            extended_values[:len(value_lists[i])] = value_lists[i]
            value_lists[i] = extended_values
    else:
        steps = max(step_lists, key=len) // 500
    
    if smooth is not None:
        for i in range(len(value_lists)):
            if smooth == "running_avg":
                value_lists[i] = running_avg(value_lists[i], smooth.param)

    print(f"{len(step_lists)} runs for: ", palg, dalg, lam, beta, mult_norms, env)
    if len(step_lists) != 4:
        print("NOT 4 RUNS^^^^^\n\n")
    
    max_length = max([len(value_list) for value_list in value_lists])
    for value_list in value_lists:
        if len(value_list) != max_length:
            value_list.extend([np.nan for _ in range(max_length - len(value_list))])

    if var is not None:
        plot_var(steps, value_lists, var=var, ax=ax, style=style)
    plot_mean(steps, value_lists, ax=ax, label=label, style=style)


def plot_x_value(x, index, label = None):
    if label is None:
        plt.plot([x, x], [0, 1], linestyle=linestyles[index], linewidth=1.5, color="dimgray")
    else:
        plt.plot([x, x], [0, 1], linestyle=linestyles[index], linewidth=1.5, color="dimgray", label=label)

def plot_group_runs(
        env, palg, dalg, group, lam = None, beta = None, mult_norms = None, baseline = None, not_baseline = None,
        metric = "train_success", 
        smooth : Smoothing = None,
        stepmin = None,
        stepmax = None,
        ax = None,
        style = {},
        plot_first_success = False,
        label = None):
    
    step_lists = []
    value_lists = []

    path = f"runs/{env}/p{palg}/d{dalg}/{group}/"
    for filename in os.listdir(path):        
        with open(path + filename, "r") as f:
            run_dict = json.load(f)
            if lam is not None and run_dict["lam"] != lam:
                continue
            if beta is not None and run_dict["beta"] != beta:
                continue
            if mult_norms is not None and mult_norms != run_dict["mult_norms"]:
                continue
            if baseline is not None and baseline != run_dict["baseline"]:
                continue
            if not_baseline is not None and not_baseline == run_dict["baseline"]:
                continue

            data = pd.DataFrame(run_dict["data"])
            steps = data['step']
            filter = step_filter(steps, stepmin, stepmax)
            step_lists.append(steps[filter])
            value_lists.append(data[metric][filter])
    

    steps = step_lists[0] // 500
    print(f"{len(step_lists)} runs for: ", palg, dalg, lam, beta, mult_norms, env)

    if plot_first_success:
        for i in range(len(value_lists)):
            where_success = np.where(value_lists[i] > 0)[0]
            if len(where_success) > 0:
                plot_x_value(steps.iloc[where_success[0]], i, label = None if i > 0 else "TTFS")
    
    if smooth is not None:
        for i in range(len(value_lists)):
            if smooth == "running_avg":
                value_lists[i] = running_avg(value_lists[i], smooth.param)

    for i, values in enumerate(value_lists):
        if label is not None and i == 0:
            ax.plot(steps, values, **style, linewidth=1.5, linestyle=linestyles[i], label=label)
        else:
            ax.plot(steps, values, **style, linewidth=1.5, linestyle=linestyles[i])




def set_ax_font(ax, fontname, fontsize):
    ax.xaxis.get_offset_text().set_fontfamily(fontname)
    ax.xaxis.get_offset_text().set_fontsize(fontsize-2)


colormap = mpl.colormaps['tab10'].colors

"""
0 - FuRL
1 - Delta features, before and after
2 - Goal baseline regularization, before and after
3 - Only directional input, before and after
4 - Concatenated input, different positional representations
"""
colors = [
    ["blue"],
    [colormap[-1], colormap[1]],
    colormap[2:4],
    [colormap[4]],
    colormap[5:7] + colormap[8:9],
]

def plot_stage2_positional(env, display = True, bad_baseline=False):
    smoothing = Smoothing("running_avg", 30)
    var = "std"
    stepmax = 1e6
    metric = "train_success"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 15

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=12)

    fig, (ax1, ax2) = figure(style=style, subplots=(1, 2), figsize=(3.25, 10))

    set_ax_font(ax1, fontname, fontsize)
    set_ax_font(ax2, fontname, fontsize)

    if bad_baseline:
        plot_group(env, 1, 0, "stage2", mult_norms=False, not_baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax1, label="$r_{P1}^{VLM}$", style={"color" : colors[1][0]})
        plot_group(env, 4, 0, "stage2", mult_norms=False, not_baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax1, label="$r_{P2}^{VLM}$", style={"color" : colors[1][1]})
        plot_group(env, 9, 0, "stage2", mult_norms=False, not_baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax2, label="$r_{P3}^{VLM}$", style={"color" : colors[2][0]})
        plot_group(env, 7, 0, "stage2", mult_norms=False, not_baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax2, label="$r_{P4}^{VLM}$", style={"color" : colors[2][1]})
    else:
        plot_group(env, 1, 0, "stage2", mult_norms=False, baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax1, label="$r_{P1}^{VLM}$", style={"color" : colors[1][0]})
        plot_group(env, 4, 0, "stage2", mult_norms=False, baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax1, label="$r_{P2}^{VLM}$", style={"color" : colors[1][1]})
        plot_group(env, 9, 0, "stage2", mult_norms=False, baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax2, label="$r_{P3}^{VLM}$", style={"color" : colors[2][0]})
        plot_group(env, 7, 0, "stage2", mult_norms=False, baseline="robot arm beside table", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax2, label="$r_{P4}^{VLM}$", style={"color" : colors[2][1]})

    plot_group(env, 0, 0, "stage2", mult_norms=False, metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax1, label="FuRL", style={"color" : colors[0][0]})
    plot_group(env, 0, 0, "stage2", mult_norms=False, metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax2, label="FuRL", style={"color" : colors[0][0]})
    plot_group(env, 0, 0, "sac", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax1, label="SAC", style={"color" : "dimgray"})
    plot_group(env, 0, 0, "sac", metric=metric, var=var, smooth=smoothing, stepmax=stepmax, ax=ax2, label="SAC", style={"color" : "dimgray"})

    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)

    ax1.set_ylim([-0.05, 1.05])
    ax2.set_ylim([-0.05, 1.05])

    ax1.set_ylabel("running avg success rate", **style)
    ax1.set_xlabel("episode", **style)
    ax2.set_xlabel("episode", **style)
    ax2.set_yticks([])
    fig.tight_layout()
    ax1.legend(prop=font, loc="lower right")
    ax2.legend(prop=font, loc="lower right")
    
    if display:
        plt.show()
    else:
        os.makedirs("images/", exist_ok=True)
        prefix = "bad_" if bad_baseline else ""
        plt.savefig(f"images/pos_{prefix}{env}", bbox_inches='tight')
    
    plt.close()


def plot_stage2_directional(env, display = True):
    smoothing = Smoothing("running_avg", 30)
    var = "std"
    metric = "train_success"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 15

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=12)

    fig, ax = figure(style=style, subplots=(1, 1), figsize=(3.25, 10))

    set_ax_font(ax, fontname, fontsize)

    plot_group(env, 0, 2, "stage2", mult_norms=False, baseline="robot arm beside table", lam=80, metric=metric, smooth=smoothing, var=var, ax=ax, label="$r_{D2}^{VLM}$", style={"color" : colors[3][0]})
    plot_group(env, 0, 3, "stage2", mult_norms=False, baseline="robot arm beside table", lam=80, metric=metric, smooth=smoothing, var=var, ax=ax, label="$r_{D3}^{VLM}$", style={"color" : colors[4][0]})
    plot_group(env, 0, 4, "stage2", mult_norms=False, baseline="robot arm beside table", lam=80, metric=metric, smooth=smoothing, var=var, ax=ax, label="$r_{D4}^{VLM}$", style={"color" : colors[4][1]})
    plot_group(env, 0, 5, "stage2", mult_norms=False, baseline="robot arm beside table", lam=80, metric=metric, smooth=smoothing, var=var, ax=ax, label="$r_{D5}^{VLM}$", style={"color" : colors[4][2]})
    plot_group(env, 0, 0, "tune_rho", mult_norms=False, rho=1, metric=metric, smooth=smoothing, var=var, ax=ax, label="FuRL", style={"color" : colors[0][0]})
    plot_group(env, 0, 0, "sac", metric=metric, var=var, smooth=smoothing, ax=ax, label="SAC", style={"color" : "dimgray"})


    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)

    ax.set_ylabel("running average success rate", **style)
    ax.set_xlabel("steps", **style)
    fig.tight_layout()
    ax.legend(prop=font, loc="lower right")
    
    if display:
        plt.show()
    else:
        os.makedirs("images/", exist_ok=True)
        plt.savefig(f"images/gen_dir_{env}", bbox_inches='tight')
    
    plt.close()


def plot_stage2_directional_lambda(env, dalg, display = True):
    smoothing = Smoothing("running_avg", 30)
    var = "std"
    metric = "train_success"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 15

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=12)

    fig, ax = figure(style=style, subplots=(1, 1), figsize=(3.25, 10))

    set_ax_font(ax, fontname, fontsize)

    c1 = 3 if dalg == 2 else 4
    c2 = 0 if dalg == 2 else dalg-3

    # baseline = "robot arm beside table" if dalg == 2 else None
    # not_baseline = "robot arm beside table" if dalg != 2 else None

    plot_group(env, 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=30, metric=metric, smooth=smoothing, var=var, ax=ax, label="λ = 30", style={"color" : colors[c1][c2], "linestyle" : linestyles[2]})
    plot_group(env, 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=80, metric=metric, smooth=smoothing, var=var, ax=ax, label="λ = 80", style={"color" : colors[c1][c2], "linestyle" : linestyles[1]})
    plot_group(env, 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=150, metric=metric, smooth=smoothing, var=var, ax=ax, label="λ = 150", style={"color" : colors[c1][c2], "linestyle" : linestyles[0]})
    plot_group(env, 0, 0, "tune_rho", mult_norms=False, rho=1, metric=metric, smooth=smoothing, var=var, ax=ax, label="FuRL", style={"color" : colors[0][0]})
    plot_group(env, 0, 0, "sac", metric=metric, var=var, smooth=smoothing, ax=ax, label="SAC", style={"color" : "dimgray"})

    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)

    ax.set_ylabel("running avg success rate", **style)
    ax.set_xlabel("episode", **style)
    fig.tight_layout()
    ax.legend(prop=font, loc="lower right")
    
    if display:
        plt.show()
    else:
        os.makedirs("images/", exist_ok=True)
        plt.savefig(f"images/param_{dalg}_{env}", bbox_inches='tight')
    
    plt.close()

    
def plot_stage1(env, display = True):
    var = None
    metric = "success_cnt"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 12

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=10)

    fig, ax = figure(style=style, subplots=(1, 1), figsize=(6, 8))

    set_ax_font(ax, fontname, fontsize)

    plot_group(env, 4, 0, "stage1", mult_norms=False, metric=metric, var=var, ax=ax, label="DF - after heads", style={"color" : colors[1][1]})
    plot_group(env, 7, 0, "stage1", mult_norms=False, metric=metric, var=var, ax=ax, label="GBR - after heads", style={"color" : colors[2][1]})
    plot_group(env, 0, 3, "stage1", mult_norms=False, metric=metric, var=var, ax=ax, label="Directional w/ concatenated input", style={"color" : colors[4][1]})
    plot_group(env, 0, 0, "stage1", mult_norms=False, metric=metric, var=var, ax=ax, label="FuRL", style={"color" : colors[0][0]})

    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)

    ax.set_ylabel("successful episodes", **style)
    ax.set_xlabel("steps", **style)
    fig.tight_layout()
    ax.legend(prop=font, loc="lower right")
    
    if display:
        plt.show()
    else:
        os.makedirs("images", exist_ok=True)
        plt.savefig(f"images/stage1_{env}", bbox_inches='tight')
    
    plt.close()



def plot_both_stages(env, display = True):
    smoothing = Smoothing("running_avg", 30)
    stepmax = 1e6
    metric = "train_success"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 16

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=12)

    fig, ax = figure(style=style, subplots=(1, 1), figsize=(4, 10))

    set_ax_font(ax, fontname, fontsize)
    set_ax_font(ax, fontname, fontsize)

    plot_group_runs(env, 0, 0, "both_stages", mult_norms=False, metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0][0]}, plot_first_success=True, label="FuRL")

    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)

    ax.set_ylabel("running avg success rate", **style)
    ax.set_xlabel("episode", **style)
    fig.tight_layout()
    ax.legend(prop=font, loc="lower right", bbox_to_anchor=(1, 0.05))
    
    if display:
        plt.show()
    else:
        os.makedirs("images/", exist_ok=True)
        plt.savefig(f"images/both_stages_{env}", bbox_inches='tight')
    
    plt.close()


def plot_rho(env, display = True):
    smoothing = Smoothing("running_avg", 30)
    var = "std"
    metric = "train_success"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 15

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=12)

    fig, ax = figure(style=style, subplots=(1, 1), figsize=(3.25, 10))

    set_ax_font(ax, fontname, fontsize)

    plot_group(env, 0, 0, "stage2", mult_norms=False, metric=metric, smooth=smoothing, var=var, ax=ax, label="ρ = 0.05", style={"color" : colors[0][0], "linestyle" : linestyles[4]})
    plot_group(env, 0, 0, "tune_rho", mult_norms=False, rho=0.25, metric=metric, smooth=smoothing, var=var, ax=ax, label="ρ = 0.25", style={"color" : colors[0][0], "linestyle" : linestyles[3]})
    plot_group(env, 0, 0, "tune_rho", mult_norms=False, rho=1, metric=metric, smooth=smoothing, var=var, ax=ax, label="ρ = 1.0", style={"color" : colors[0][0], "linestyle" : linestyles[2]})
    plot_group(env, 0, 0, "tune_rho", mult_norms=False, rho=2, metric=metric, smooth=smoothing, var=var, ax=ax, label="ρ = 2.0", style={"color" : colors[0][0], "linestyle" : linestyles[1]})
    plot_group(env, 0, 0, "tune_rho", mult_norms=False, rho=4, metric=metric, smooth=smoothing, var=var, ax=ax, label="ρ = 4.0", style={"color" : colors[0][0], "linestyle" : linestyles[0]})

    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)

    ax.set_ylabel("running avg success rate", **style)
    ax.set_xlabel("episode", **style)
    fig.tight_layout()
    ax.legend(prop=font, loc="lower right")
    
    if display:
        plt.show()
    else:
        os.makedirs("images/", exist_ok=True)
        plt.savefig(f"images/rho_{env}", bbox_inches='tight')
    
    plt.close()


def plot_release_successrate(dalg, display = True):
    smoothing = Smoothing("running_avg", 30)
    var = "std"
    metric = "train_success"

    # colors = mpl.color_sequences['Dark2']

    fontname = "Open Sans"
    fontsize = 15

    style = {
        "fontname" : fontname,
        "fontsize" : fontsize
    }
    font = mpl.font_manager.FontProperties(
        family=fontname,
        style='normal', size=10)

    fig, (ax1, ax2) = figure(style=style, subplots=(2, 1), figsize=(6.5, 6))

    set_ax_font(ax1, fontname, fontsize)
    set_ax_font(ax2, fontname, fontsize)

    baseline = "robot arm beside table" if dalg == 2 else None
    not_baseline = "robot arm beside table" if dalg != 2 else None

    plot_group("door-open-v2-goal-hidden", 0, 0, "sac", metric=metric, var=var, smooth=smoothing, ax=ax1, label="SAC", style={"color" : "green"})
    plot_group("door-open-v2-goal-hidden", 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=30, metric=metric, smooth=smoothing, var=var, ax=ax1, label="λ = 30", style={"color" : "red", "linestyle" : linestyles[2]})
    plot_group("door-open-v2-goal-hidden", 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=80, metric=metric, smooth=smoothing, var=var, ax=ax1, label="λ = 80", style={"color" : "red", "linestyle" : linestyles[1]})
    plot_group("door-open-v2-goal-hidden", 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=150, metric=metric, smooth=smoothing, var=var, ax=ax1, label="λ = 150", style={"color" : "red", "linestyle" : linestyles[0]})
    plot_group("door-open-v2-goal-hidden", 0, 0, "stage2", mult_norms=False, rho=0.05, metric=metric, smooth=smoothing, var=var, ax=ax1, label="ρ = 0.05", style={"color" : colors[0][0], "linestyle" : linestyles[4]})
    plot_group("door-open-v2-goal-hidden", 0, 0, "tune_rho", mult_norms=False, rho=0.25, metric=metric, smooth=smoothing, var=var, ax=ax1, label="ρ = 0.25", style={"color" : colors[0][0], "linestyle" : linestyles[2]})
    plot_group("door-open-v2-goal-hidden", 0, 0, "tune_rho", mult_norms=False, rho=1, metric=metric, smooth=smoothing, var=var, ax=ax1, label="ρ = 1.0", style={"color" : colors[0][0], "linestyle" : linestyles[1]})
    plot_group("door-open-v2-goal-hidden", 0, 0, "tune_rho", mult_norms=False, rho=2, metric=metric, smooth=smoothing, var=var, ax=ax1, label="ρ = 2.0", style={"color" : colors[0][0], "linestyle" : linestyles[0]})

    plot_group("button-press-topdown-v2-goal-hidden", 0, 0, "sac", metric=metric, var=var, smooth=smoothing, ax=ax2, label="SAC", style={"color" : "green"})
    plot_group("button-press-topdown-v2-goal-hidden", 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=30, metric=metric, smooth=smoothing, var=var, ax=ax2, label="λ = 30", style={"color" : "red", "linestyle" : linestyles[2]})
    plot_group("button-press-topdown-v2-goal-hidden", 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=80, metric=metric, smooth=smoothing, var=var, ax=ax2, label="λ = 80", style={"color" : "red", "linestyle" : linestyles[1]})
    plot_group("button-press-topdown-v2-goal-hidden", 0, dalg, "stage2", baseline=baseline, not_baseline=not_baseline, mult_norms=False, lam=150, metric=metric, smooth=smoothing, var=var, ax=ax2, label="λ = 150", style={"color" : "red", "linestyle" : linestyles[0]})
    plot_group("button-press-topdown-v2-goal-hidden", 0, 0, "stage2", mult_norms=False, rho=0.05, metric=metric, smooth=smoothing, var=var, ax=ax2, label="ρ = 0.05", style={"color" : colors[0][0], "linestyle" : linestyles[4]})
    plot_group("button-press-topdown-v2-goal-hidden", 0, 0, "tune_rho", mult_norms=False, rho=0.25, metric=metric, smooth=smoothing, var=var, ax=ax2, label="ρ = 0.25", style={"color" : colors[0][0], "linestyle" : linestyles[2]})
    plot_group("button-press-topdown-v2-goal-hidden", 0, 0, "tune_rho", mult_norms=False, rho=1, metric=metric, smooth=smoothing, var=var, ax=ax2, label="ρ = 1.0", style={"color" : colors[0][0], "linestyle" : linestyles[1]})
    plot_group("button-press-topdown-v2-goal-hidden", 0, 0, "tune_rho", mult_norms=False, rho=2, metric=metric, smooth=smoothing, var=var, ax=ax2, label="ρ = 2.0", style={"color" : colors[0][0], "linestyle" : linestyles[0]})

    # plot_group_runs(env, 0, 0, "stage2", metric=metric, smooth=smoothing, stepmax=stepmax, ax=ax, style={"color" : colors[0, 0][0]}, plot_first_success=True)
    

    ax1.set_ylabel("running avg success rate", **style)
    ax1.set_xticks([])
    ax2.set_ylabel("running avg success rate", **style)
    ax2.set_xlabel("episode", **style)
    fig.tight_layout()
    ax1.legend(prop=font, loc="lower right", ncol=2)
    ax2.legend(prop=font, loc="lower right", ncol=2)
    
    if display:
        plt.show()
    else:
        os.makedirs("release_images/", exist_ok=True)
        plt.savefig(f"release_images/door_button_comparison", bbox_inches='tight')
    
    plt.close()




if __name__ == "__main__":
    envs = [
        "drawer-open-v2-goal-hidden",
        "door-open-v2-goal-hidden",
        "window-open-v2-goal-hidden",
        "window-close-v2-goal-hidden",
        "push-v2-goal-hidden",
        "button-press-topdown-v2-goal-hidden",
    ]

    plot_both_stages("door-open-v2-goal-hidden", False)

    for env in envs:
        plot_stage2_directional(env, display=False)

    plot_release_successrate(3, False)
