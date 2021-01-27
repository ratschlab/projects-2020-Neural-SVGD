import os
on_cluster = not os.getenv("HOME") == "/home/lauro"
if on_cluster:
    results_path = "/cluster/home/dlauro/projects-2020-Neural-SVGD/experiments/results/"
    batch_size = 1024
    model_size = 'large'
    data_dir = "./data"
else:
    results_path = "/home/lauro/code/msc-thesis/main/experiments/results/"
    batch_size = 128
    model_size = 'small'
    data_dir = "/tmp/tfds"
figure_path = results_path + "figures/"
# figure_path = "/home/lauro/documents/msc-thesis/paper/latex/figures/"
