import os
on_cluster = not os.getenv("HOME") == "/home/lauro"
if on_cluster:
    results_path = "/cluster/home/dlauro/projects-2020-Neural-SVGD/experiments/results/"
else:
    results_path = "/home/lauro/code/msc-thesis/main/experiments/results/"
figure_path = results_path + "figures/"
# figure_path = "/home/lauro/documents/msc-thesis/paper/latex/figures/"
