import argparse
import csv
import os
import time
from collections import defaultdict

import higher
import numpy as np
import torch
from torch import nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    gpu_properties = torch.cuda.get_device_properties(0)
    print(f"GPU:      {gpu_properties.name}")
    print(f"Capacity: {gpu_properties.total_memory / (1024 ** 2):.2f} MB")
else:
    raise Exception("This assignment requires a GPU for memory logging purposes!")


def measure_memory():
    """Measure the current GPU memory usage in MB."""
    torch_allocated = torch.cuda.memory_allocated() / 1024**2
    # torch_reserved = torch.cuda.memory_reserved() / 1024**2
    return torch_allocated


def generate_sine_data(
    train_range=(0, 2 * np.pi),
    test_range=(0, 2 * np.pi),
    N=10,
    test_N=1000,
    amplitude_range=(0.5, 1),
    phase_range=(0, 2 * np.pi),
    bias_range=(-0.3, 0.3),
):
    """Generate one train-test split of sine task."""
    train_x = np.random.uniform(low=train_range[0], high=train_range[1], size=N)
    test_x = np.linspace(test_range[0], test_range[1], test_N)
    amplitude = np.random.uniform(low=amplitude_range[0], high=amplitude_range[1])
    phase = np.random.uniform(low=phase_range[0], high=phase_range[1])
    bias = np.random.uniform(low=bias_range[0], high=bias_range[1])
    f = (
        lambda x: amplitude * np.sin(x + phase) + bias
    )  # sampled function: x -> a*sin(x+phase)+bias
    train_y = f(train_x)
    test_y = f(test_x)
    train_x = torch.tensor(train_x).float().reshape(-1, 1).to(device)
    train_y = torch.tensor(train_y).float().reshape(-1, 1).to(device)
    test_x = torch.tensor(test_x).float().reshape(-1, 1).to(device)
    test_y = torch.tensor(test_y).float().reshape(-1, 1).to(device)
    return (train_x, train_y), (test_x, test_y)


def get_mlp(layers, hidden_size):
    """Construct an MLP with ReLU activations."""
    layer_list = [nn.Linear(1, hidden_size), nn.ReLU()]
    for _ in range(layers - 1):
        layer_list += [nn.Linear(hidden_size, hidden_size), nn.ReLU()]
    layer_list.append(nn.Linear(hidden_size, 1))
    return nn.Sequential(*layer_list)


def train_maml(net, args, warmup=False):
    """Train a model using MAML."""
    mse_loss = nn.MSELoss()
    meta_opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    net = net.to(device)

    start_time = time.perf_counter()
    resources = defaultdict(list)
    outer_steps = 1 if warmup else args.outer_steps
    for outer_step in range(outer_steps):
        metrics = defaultdict(list)
        for _ in range(args.meta_batch_size):
            data = generate_sine_data()
            train_data, test_data = data
            train_x, train_y = train_data

            inner_opt = torch.optim.SGD(net.parameters(), lr=1e-2)
            higher_context = higher.innerloop_ctx(
                net, inner_opt, copy_initial_weights=False, track_higher_grads=True
            )
            with higher_context as (fnet, diffopt):
                # Inner loop context using the higher package.
                # Documentation: https://higher.readthedocs.io/en/latest/toplevel.html#higher.innerloop_ctx
                for _ in range(args.inner_steps):
                    preds = fnet(train_x)
                    loss = mse_loss(preds, train_y)
                    diffopt.step(loss)

                test_x, test_y = test_data
                test_preds = fnet(test_x)
                test_loss = mse_loss(test_preds, test_y)
                test_loss.backward()
                metrics["test_loss"].append(test_loss.item())

        now = time.perf_counter()
        resources["time"].append(now - start_time)
        resources["memory"].append(measure_memory())
        torch.cuda.empty_cache()
        start_time = now

        meta_opt.step()
        meta_opt.zero_grad()
    return resources


def finetune(_net, data, steps):
    """Finetune a model on a given dataset for a given number of steps."""
    train_data, test_data = data
    train_x, train_y = train_data

    mse_loss = nn.MSELoss()
    inner_opt = torch.optim.SGD(_net.parameters(), lr=1e-2)
    for _ in range(steps):
        preds = _net(train_x)
        loss = mse_loss(preds, train_y)
        inner_opt.zero_grad()
        loss.backward()
        inner_opt.step()
    test_x, test_y = test_data
    test_preds = _net(test_x)
    test_loss = mse_loss(test_preds, test_y)
    return test_loss.item()


def train_es(net, args, warmup=False):
    """Train a model using Evolution Strategies."""
    meta_opt = torch.optim.SGD(net.parameters(), lr=1.0)

    start_time = time.perf_counter()
    resources = defaultdict(list)
    outer_steps = 1 if warmup else args.outer_steps
    for outer_step in range(outer_steps):
        current_losses = []
        meta_grad = {k: torch.zeros_like(v) for k, v in net.named_parameters()}
        for _ in range(args.meta_batch_size // 2):
            data = generate_sine_data()
            parameters = net.state_dict()
            noise_scale = 1e-3
            epsilon = {
                k: torch.randn_like(v) * noise_scale for k, v in parameters.items()
            }
            # Antithetic sampling: we evaluate both p+epsilon and p-epsilon
            # see e.g. https://arxiv.org/pdf/1703.03864.pdf
            params_p_epsilon = {k: p + epsilon[k] for k, p in parameters.items()}
            net.load_state_dict(params_p_epsilon)
            meta_loss_p = finetune(net, data, args.inner_steps)
            del params_p_epsilon

            params_m_epsilon = {k: p - epsilon[k] for k, p in parameters.items()}
            net.load_state_dict(params_m_epsilon)
            meta_loss_m = finetune(net, data, args.inner_steps)
            del params_m_epsilon

            objective = meta_loss_p - meta_loss_m
            for k in parameters.keys():
                meta_grad[k] += objective * epsilon[k] / (2 * noise_scale)

            net.load_state_dict(parameters)
            meta_loss_current = finetune(net, data, args.inner_steps)
            current_losses.append(meta_loss_current)

        now = time.perf_counter()
        resources["time"].append(now - start_time)
        resources["memory"].append(measure_memory())
        torch.cuda.empty_cache()
        start_time = now

        meta_opt.zero_grad()
        for k, p in net.named_parameters():
            p.grad = meta_grad[k] / args.meta_batch_size
        meta_opt.step()
    return resources


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--inner_steps", type=int, default=5)
    parser.add_argument("--outer_steps", type=int, default=3)
    parser.add_argument("--meta_batch_size", type=int, default=32)
    parser.add_argument("--experiment_name", type=str, default="debug")
    parser.add_argument("--run_name", type=str, default="")
    args = parser.parse_args()

    net = get_mlp(layers=args.layers, hidden_size=args.hidden_size)

    # warmup, so time and memory measurements are more accurate
    train_maml(net, args, warmup=True)
    train_es(net, args, warmup=True)
    print("Warmup done!")

    torch.cuda.empty_cache()
    resources_maml = train_maml(net, args)

    torch.cuda.empty_cache()
    resources_es = train_es(net, args)

    maml_time = np.mean(resources_maml["time"])
    maml_memory = np.max(resources_maml["memory"])
    es_time = np.mean(resources_es["time"])
    es_memory = np.max(resources_es["memory"])

    print(f"Results:")
    print(f"MAML time per outer loop (s): {maml_time:.4f}")
    print(f"MAML peak memory (MB):        {maml_memory:.4f}")
    print(f"ES time per outer loop (s):   {es_time:.4f}")
    print(f"ES peak memory (MB):          {es_memory:.4f}")

    # Save results to csv
    csv_data = [args.run_name, maml_time, maml_memory, es_time, es_memory]
    output_path = f"results/{args.experiment_name}.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "a", newline="") as f:
        writer = csv.writer(f, delimiter=",")
        if os.stat(output_path).st_size == 0:
            column_names = [
                "Run Name",
                "MAML (time)",
                "MAML (memory)",
                "ES (time)",
                "ES (memory)",
            ]
            writer.writerow(column_names) # Write header if file is empty
        writer.writerow(csv_data) # Write experiment results to csv file
