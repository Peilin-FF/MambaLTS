# MambaLTS
<<<<<<< HEAD
This is a Library for ultilizing Mamba architecture to analyze Long Time Series Tasks.

=======
The repo is the implementation for analysing Long Time Series Tasks using Mamba1 \& Mamba2 architecture:
## &#x1F389; News
* **[2024.08.15]**  We find a good set of hyperparameters for Mamba1, Mamba2 
## Installation and Requirements
Please note that our project depends on mamba-ssm. You need to carefully check the corresponding versions of torch, cuda, mamba-ssm and python to prevent wheel installation from failing.

We use *python-3.11* \& *torch-2.0.0* \& *cuda-11.8*  
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

3. Install Environment
```Shell
pip install torch==2.0.0+cu118
```
```Shell
wget https://github.com/state-spaces/mamba/releases/download/v2.2.2/mamba_ssm-2.2.2+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```
```Shell
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.4.0/causal_conv1d-1.4.0+cu118torch2.0cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```
After all the work, you need to pip install triton==2.3.0 again, because there is a false version triton==2.0.0 bug in mamba-ssm package.
```Shell
pip install triton==2.3.0
```

## Contact
If you have any questions, feel free to either initiate an *Issue* or contact us by Email (Email: *fengpeilin@buaa.edu.cn*).

## ❤️ Community efforts
* Our codebase is built upon the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) project. Great work!
>>>>>>> 7cb32ea4b1edcacb6693bb23b5afa438951420cf
