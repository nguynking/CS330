import os
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("figures", exist_ok=True)

# Shared colors; feel free to change
maml_color = "#FF6150"
es_color = "#134E6F"


# Plot inner steps experiment
exp_df = pd.read_csv("results/inner_steps_exp.csv")

plt.figure(figsize=(5, 3))
plt.plot(exp_df["Run Name"], exp_df["MAML (time)"], "o-", label='MAML', color=maml_color)
plt.plot(exp_df["Run Name"], exp_df["ES (time)"], "o-", label='ES', color=es_color)
plt.xlabel('Inner Steps')
plt.ylabel('Time')
plt.title('Inner Steps vs Time')
plt.legend()
plt.savefig(f"figures/steps_time.pdf", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(exp_df["Run Name"], exp_df["MAML (memory)"], "o-", label='MAML', color=maml_color)
plt.plot(exp_df["Run Name"], exp_df["ES (memory)"], "o-", label='ES', color=es_color)
plt.xlabel('Inner Steps')
plt.ylabel('Memory')
plt.title('Inner Steps vs Memory')
plt.legend()
plt.savefig(f"figures/steps_memory.pdf", bbox_inches='tight', dpi=300)
plt.show()


# Plot layers experiment
exp_df = pd.read_csv("results/layers_exp.csv")

plt.figure(figsize=(5, 3))
plt.plot(exp_df["Run Name"], exp_df["MAML (time)"], "o-", label='MAML', color=maml_color)
plt.plot(exp_df["Run Name"], exp_df["ES (time)"], "o-", label='ES', color=es_color)
plt.xlabel('Layers')
plt.ylabel('Time')
plt.title('Layers vs Time')
plt.legend()
plt.savefig(f"figures/layers_time.pdf", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(exp_df["Run Name"], exp_df["MAML (memory)"], "o-", label='MAML', color=maml_color)
plt.plot(exp_df["Run Name"], exp_df["ES (memory)"], "o-", label='ES', color=es_color)
plt.xlabel('Layers')
plt.ylabel('Memory')
plt.title('Layers vs Memory')
plt.legend()
plt.savefig(f"figures/layers_memory.pdf", bbox_inches='tight', dpi=300)
plt.show()


# Plot hidden nodes experiment
exp_df = pd.read_csv("results/hidden_exp.csv")

plt.figure(figsize=(5, 3))
plt.plot(exp_df["Run Name"], exp_df["MAML (time)"], "o-", label='MAML', color=maml_color)
plt.plot(exp_df["Run Name"], exp_df["ES (time)"], "o-", label='ES', color=es_color)
plt.xlabel('Hidden Size')
plt.ylabel('Time')
plt.title('Hidden Size vs Time')
plt.legend()
plt.savefig(f"figures/hidden_time.pdf", bbox_inches='tight', dpi=300)
plt.show()

plt.figure(figsize=(5, 3))
plt.plot(exp_df["Run Name"], exp_df["MAML (memory)"], "o-", label='MAML', color=maml_color)
plt.plot(exp_df["Run Name"], exp_df["ES (memory)"], "o-", label='ES', color=es_color)
plt.xlabel('Hidden Size')
plt.ylabel('Memory')
plt.title('Hidden Size vs Memory')
plt.legend()
plt.savefig(f"figures/hidden_memory.pdf", bbox_inches='tight', dpi=300)
plt.show()
