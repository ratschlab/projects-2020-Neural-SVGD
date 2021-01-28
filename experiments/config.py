import os
try:
    location = os.environ['CLUSTERNAME']
except KeyError:
    if os.getenv("HOME") == "/home/lauro":
        location = "local"
else:
    raise


model_size = 'large' if location == "leonhard" else 'small'

if location in ['euler', 'leonhard']:
    results_path = "/cluster/home/dlauro/projects-2020-Neural-SVGD/experiments/results/"
    batch_size = 128
    data_dir = "./data"
elif location in ['local']:
    results_path = "/home/lauro/code/msc-thesis/main/experiments/results/"
    batch_size = 128
    data_dir = "/tmp/tfds"

figure_path = results_path + "figures/"
n_samples = 100
# figure_path = "/home/lauro/documents/msc-thesis/paper/latex/figures/"
