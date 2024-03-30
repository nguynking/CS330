# CS330: Deep Multi-Task and Meta Learning - Stanford/Fall 2023

![image](https://github.com/nguynking/cs330/assets/110026135/de2a37b4-a0d0-44e3-b163-e872183c9600)

## Overview

While deep learning has achieved remarkable success in many problems such as image classification, natural language processing, and speech recognition, these models are, to a large degree, specialized for the single task they are trained for. This course will cover the setting where there are multiple tasks to be solved, and study how the structure arising from multiple tasks can be leveraged to learn more efficiently or effectively. This includes:

- self-supervised pre-training for downstream few-shot learning and transfer learning
- meta-learning methods that aim to learn efficient learning algorithms that can learn new tasks quickly
- curriculum and lifelong learning, where the problem requires learning a sequence of tasks, leveraging their shared structure to enable knowledge transfer

This is a graduate-level course. By the end of the course, students will be able to understand and implement the state-of-the-art multi-task learning and meta-learning algorithms and be ready to conduct research on these topics.

## Main sources
* [**Course page**](https://cs330.stanford.edu/)
* [**Lecture videos** (2022)](https://youtube.com/playlist?list=PLoROMvodv4rNjRoawgt72BBNwL2V7doGI&si=vKriWT96_bXBBp15)

## Requirements
For `pip` users, the instructions on how to set-up the environment are given in the handouts and can be installed as follows:

```shell
$ cd assignment1/code
$ pip install -r requirements.txt
```

For code that requires **Azure** _Virtual Machines_, I was able to run everything successfully on **Google Colab** with a free account.

> [!Note]
> Python 3.10 or newer should be used

## Structure

Each assignment within this course is organized into three primary components:

- `code`: Contains all the necessary code files for the assignment. Example content includes Python scripts (some_code.py) and a list of requirements (requirements.txt).
- `latex`: Supplies the Latex source files required to compile the assignment report. This includes the main report document (report.tex) and any additional files needed for report compilation.
- `handout`: Provides detailed instructions for each assignment, ensuring clarity on objectives, submission guidelines, and evaluation criteria.
```
assignment
│
├── code
│   ├── some_code.py
│   └── requirements.txt
│
├── latex
│   ├── report.tex
│   └── some_other_file
│
└── handout.pdf
```

## Solutions

* [**assignment 0**](assignment0): Multitask Training for Recommender Systems
* [**assignment 1**](assignment1): Data Processing and Black-Box Meta-Learning
* [**assignment 2**](assignment2): Prototypical Networks and Model-Agnostic Meta-Learning
* [**assignment 3**](assignment3): Few-Shot Learning with Pre-trained Language Models
* [**assignment 4**](assignment4): Advanced Meta-Learning Topics (__optional__)
