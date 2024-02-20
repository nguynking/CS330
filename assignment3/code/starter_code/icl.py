from cProfile import label
from multiprocessing.sharedctypes import Value
from typing import List
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import transformers
import numpy as np
import random

import argparse
from collections import defaultdict
import json
import os
from rouge_score import rouge_scorer
import tqdm

try:
    import utils
except ModuleNotFoundError:
    from . import utils

parser = argparse.ArgumentParser()
parser.add_argument("--task")
parser.add_argument("--model")
parser.add_argument("--dataset")
parser.add_argument("--k")
parser.add_argument("--prompt", default="babi")
parser.add_argument("--debug", action="store_true")
parser.add_argument("--repeats", default=1, type=int)
parser.add_argument("--device", default="cuda")
parser.add_argument("--plot_name", default="plot.png")
args = parser.parse_args()


if os.environ.get("FORCE_DEVICE", False):
    DEVICE = torch.device(os.environ["FORCE_DEVICE"])
else:
    DEVICE = torch.device(args.device)

print("In-context learning using device: ", DEVICE)


def get_icl_prompts(
    support_inputs: List[str],
    support_labels: List[str],
    test_input: str,
    prompt_mode: str = "none",
) -> str:
    """
    Take a list of contexts and combine them into k-shot prompts.

    **Note**: Be sure to shuffle the support examples and labels
      *together* (i.e. so the pairings of support input/label is preserved)
      before constructing the prompt. np.random.permutation may be helpful.

    Args:
      support_inputs: The k inputs used for in-context learning (k may be zero!)
      support_labels: The k labels used for in-context learning (k may be zero!)
      test_input: The input we are evaluating on
      prompt_mode: The task description mode we're using; 'none' means we're only using
        k-shot examples, 'tldr' means we're using the tl;dr prompt from the GPT-2 paper,
        etc.

    Returns:
      A string containing the complete input to the model.

    Examples:
      get_icl_prompts(
        support_inputs=["Andy is in the kitchen. Where's Andy?", "Betina is in the office. Where's
                        Betina?"],
        support_labels=["kitchen", "office"],
        test_input="Carlos is in the office. Where's Carlos?",
        prompt_mode="babi"
      ) -> "Betina is in the office. Where's Betina? In the office. Andy is in the kitchen. Where's
            Andy? In the kitchen. Carlos is in the office. Where's Carlos? In the"

      get_icl_prompts(
        support_inputs=["article 1", "article 2", "article 3"],
        support_labels=["summary 1", "summary 2", "summary 3"],
        test_input="test article",
        prompt_mode="none"
        ) -> "article 3 summary 3 article 1 summary 1 article 2 summary 2 test article"

      get_icl_prompts(
        support_inputs=["article 1", "article 2", "article 3"],
        support_labels=["summary 1", "summary 2", "summary 3"],
        test_input="test article",
        prompt_mode="tldr"
        ) -> "article 2 TL;DR: summary 2 article 3 TL;DR: summary 3 article 1 TL;DR: summary 1 test
              article TL;DR:"
    """
    assert prompt_mode in ["babi", "none", "tldr", "custom"]
    prompt = ""

    # Shuffle the support inputs and labels
    permutation = np.random.permutation(
        len(support_inputs)
    )  # Your code should use this ordering!

    ### START CODE HERE ###
    assert False, "Complete this for Q1.1a"
    ### END CODE HERE ###

    assert prompt[-1] != " "
    return prompt


def get_performance_metric(
    predictions: List[str], targets: List[str], metric: str
) -> float:
    if metric == "rouge":
        scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)
        scores = []
        for p, t in zip(predictions, targets):
            score = scorer.score(p, t)["rouge1"].fmeasure
            scores.append(score)
        return sum(scores) / len(scores)
    elif metric == "exact match":
        if isinstance(targets[0], str):
            return sum(
                [p.strip() == t.strip() for p, t in zip(predictions, targets)]
            ) / len(predictions)
        else:

            def _normalize(prediction):
                if prediction.endswith("Q"):
                    prediction = prediction[:-1]
                elif "Q:" in prediction:
                    prediction = prediction[: prediction.index("Q:")]
                return prediction.strip(". ").lower()

            normalized = [_normalize(p) for p in predictions]

            def contains(key, candidates):
                for c in candidates:
                    if key in c:
                        return True
                return False

            return sum([contains(n, t) for n, t in zip(normalized, targets)]) / len(
                normalized
            )
    else:
        raise NotImplementedError()


def do_sample(
    model: transformers.GPT2LMHeadModel,
    input_ids: torch.Tensor,
    stop_tokens: List[int],
    max_tokens: int,
) -> List[int]:
    """
    Sample from the model using the given input_ids as a prefix until we either
    hit the stop token or we have sampled max_tokens tokens.

    (Don't use model.generate; implement this yourself in a loop)

    Note: when calling the model here, be sure to wrap the call with
      torch.inferece_mode() to save memory!

    For more information on how to sample from the GPT2 model, see the documentation for the forward
      function: https://huggingface.co/docs/transformers/model_doc/gpt2#transformers.GPT2LMHeadModel
      In particular, see the past_key_values and use_cache arguments.
      If you use caching, the first call to the model will include all initial input_ids, but
      subsequent calls will only include the last token generated autoregressively, since embeddings
      for all prior tokens are cached. This speeds up generation significantly.


    Args:
        model: A transformers.PreTrainedModel that we will sample from.
        input_ids: An integer tensor of shape [1, prefix_len]
        stop_tokens: A list of token ids that indicates that we should stop sampling (e.g., a period)
        max_tokens: Stop sampling if we've sampled this many tokens

    Returns:
        The sampled tokens (a python list of ints/zero-dim tensors), not including the input_ids prefix
          OR the stop token (if we hit the stop token before max_tokens)
    """
    sampled_tokens = []
    # Complete this for Q1.1b
    ### START CODE HERE ###
    assert False, "Complete this for Q1.1b"
    ### END CODE HERE ###
    return sampled_tokens


def run_icl(
    models: List[str],
    datasets_: List[str],
    ks: List[int],
    prompt_modes: List[str],
    n_val: int = 125,
):
    results = {}
    for model_name in models:
        print(f"Loading model {model_name}...")
        utils.fix_random_seeds()
        model, tokenizer = utils.get_model_and_tokenizer(
            model_name, transformers.AutoModelForCausalLM
        )
        stop_tokens = utils.stop_tokens(tokenizer)
        model.to(DEVICE)

        for dataset in datasets_:
            print(f"Loading dataset {dataset}...")
            if args.debug:
                n_val = 1
            utils.fix_random_seeds()
            max_tokens = utils.max_sampled_tokens_for_dataset(dataset)
            train, val = utils.get_dataset(dataset, n_train=max(ks), n_val=n_val)
            for prompt_mode in prompt_modes:
                for k in ks:
                    print(
                        f"Running in-context learning with {model_name} on {dataset} with k={k} and prompt_mode={prompt_mode}"
                    )
                    utils.fix_random_seeds()
                    for repeat in range(args.repeats):
                        if repeat > 0:
                            print(f"Beginning repeat #{repeat}")
                        support_idxs = random.choices(range(len(train["x"])), k=k)
                        support_x = [
                            train["x"][idx].replace("\n", " ") for idx in support_idxs
                        ]
                        support_y = [
                            train["simple_y"][idx].replace("\n", " ")
                            for idx in support_idxs
                        ]
                        targets = []
                        predictions = []
                        pbar = tqdm.tqdm(list(range(min(n_val, len(val["x"])))))
                        for row in pbar:
                            test_input = val["x"][row]
                            targets.append(val["y"][row])

                            # Ingredients you'll need:
                            #   get_icl_prompts() [which you implemented]
                            #   do_sample() [which you implemented]
                            #   tokenizer() (for encoding text into tokens) and tokenizer.decode() (for decoding tokens back into text)
                            #   See the documentation for the tokenizer encoder function here:
                            #   https://huggingface.co/docs/transformers/v4.23.1/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.__call__
                            # Note that the tokenizer by default will give you results on the CPU, so you will need to move them to the
                            # proper device. You can do this with the .to() method, e.g.,
                            #   my_tensor.to(DEVICE), where DEVICE is defined at lines 32-35.
                            decoded_prediction = ""
                            # YOUR CODE HERE, complete for Q1.1c. Should be ~5-10 lines of code.
                            assert False, "Complete this for Q1.1c"
                            # END YOUR CODE

                            predictions.append(decoded_prediction)
                            metric = get_performance_metric(
                                predictions, targets, utils.metric_for_dataset(dataset)
                            )
                            pbar.set_description(f"Eval: {metric:.04f}")
                        results[
                            "_".join([model_name, dataset, str(k), prompt_mode])
                        ] = metric

                        print("Evaluation results:", results)
                        if not os.path.exists(f"{utils.RESULTS_DIR}/icl"):
                            os.makedirs(f"{utils.RESULTS_DIR}/icl")

                        for k_, v in results.items():
                            print(
                                f"Writing results to: {utils.RESULTS_DIR}/icl/{k_}.json"
                            )
                            with open(f"{utils.RESULTS_DIR}/icl/{k_}.json", "w") as f:
                                json.dump({"metric": v}, f)
                        results = {}


def plot_icl(models, dataset, ks, prompt_modes, output_path: str):
    data = defaultdict(lambda: defaultdict(list))
    symbols = ["solid", "dashed", "dotted", "dashdot"]

    x_vals = set()
    for model in models:
        symbol = symbols.pop(0)
        for prompt_mode in prompt_modes:
            for k in ks:
                fn = "_".join([model, dataset, str(k), prompt_mode])
                id_ = "_".join([model, dataset, prompt_mode])
                with open(f"{utils.RESULTS_DIR}/icl/{fn}.json", "r") as f:
                    score = json.load(f)["metric"]
                    data[id_]["x"].append(k)
                    x_vals.add(k)
                    data[id_]["y"].append(score)
                    data[id_]["linestyle"] = symbol

    for k, v in data.items():
        plt.plot(v["x"], v["y"], label=k, linestyle=v["linestyle"])

    if max(x_vals) > 4:
        plt.xscale("symlog")

    ax = plt.gca()
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_ticks(v["x"])
    plt.legend()
    plt.title(dataset)
    plt.ylabel(utils.metric_for_dataset(dataset))
    plt.xlabel("Number of support examples")
    # plt.show()
    plt.savefig(output_path, bbox_inches="tight")


def run():
    ks = [int(k) for k in args.k.split(",")]
    if args.task == "icl":
        run_icl(
            args.model.split(","), args.dataset.split(","), ks, args.prompt.split(",")
        )
    elif args.task == "plot":
        assert "," not in args.dataset, "Only one dataset at a time for plotting"
        plot_icl(
            args.model.split(","),
            args.dataset,
            ks,
            args.prompt.split(","),
            args.plot_name,
        )


if __name__ == "__main__":
    run()
