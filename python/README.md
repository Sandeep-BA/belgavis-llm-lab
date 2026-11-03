# Python

Author: Sandeep Belgavi. Package `belgavi_llm`, one subpackage per episode (`lNN_<topic>`).

```
pip install -e "python[l01]"          # Episode 1: tokens
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -e "python[l02]"          # Episode 2: transformer (plus CPU PyTorch above)
pip install -e "python[dev,l01,l02]"  # plus tests and lint
python -m belgavi_llm.hardware        # hardware report
```
