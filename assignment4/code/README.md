# CS330 Homework 4

This project contains a set of experiments comparing MAML with ES.
The results of these experiments are logged and visualized for further analysis.
You must run this command on a machine with a GPU, since we measure the GPU memory usage.

To run all experiments and record the runtime and memory statistics, use the following command:

```bash
bash all_experiments.sh
```

This will log all the runtime statistics into csv files in `results/`.

After running the experiments, you can visualize the results by running the plotting script:

```bash
python plot.py
```

This script reads the data from `results/` and generates plots comparing MAML with ES.
These plots will be saved in `figures/`, and you should incorporate these into your writeup.
