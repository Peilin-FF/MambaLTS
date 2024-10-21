# Mamba-Library
The repo is the implementation for analysing Long Time Series Tasks using Mamba1 \& Mamba2 architecture:
## &#x1F389; News
* **[2024.08.15]**  Our paper: [TinyLLaVA: A Framework of Small-scale Large Multimodal Models](https://arxiv.org/abs/2402.14289) is released!

## Installation and Requirements
Please note that our project depends on mamba-ssm. You need to carefully check the corresponding versions of torch, cuda, and mamba-ssm to prevent wheel installation from failing.
1. Clone this repository and navigate to the folder
```bash
git clone https://github.com/Peilin-FF/MambaLTS.git
cd MambaLTS
```
2. Create a conda environment, activate it and install Packages
```Shell
conda create -n MambaLTS python=3.10 -y
conda activate MambaLTS
pip install --upgrade pip 
pip install -e .
```
3. Install additional packages

## Contact
If you have any questions, feel free to either initiate an *Issue* or contact us by Email (Email: *fengpeilin@buaa.edu.cn*).

## ❤️ Community efforts
* Our codebase is built upon the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) project. Great work!
