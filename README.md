# batched-melbo

## Motivation

Unsupervised steering vectors (known by common folk as MELBO) is one of the main elicitation techniques to uncover hidden behaviours in LLMs. However, the original implementation does not leverage GPU parallelization and trains vectors sequentially to add orthogonality contraint to the steering vectors. This means training is slow and there is huge headroom for speedup if we give up on making the steering vectors orthogonal. We have observed [that we can elicit backdoors without adding orthogonality to trained vectors.](notebooks/batched_melbo_validation.ipynb)

This repo, batched MELBO, gives up orthogonality to significantly speedup training, usually OOM more (see image below, [see full notebook](notebooks/batched_melbo_performance.ipynb)). This makes hyper-parameters sweeps practical, which we have internally observed to be paramount for some behaviours to be elicited. 

![Training time comparison](notebooks/plot.png)
*Training time comparison between original MELBO and batched MELBO across different numbers of steering vectors.*


The reason we have created a separate repo is mainly to give us freedom in altering and evolving the original training objective (maybe take some inspirations from [this](https://www.lesswrong.com/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1?commentId=Rm7hhD2qgfh7Za4LA)?). 

## Installation
```sh
pip install batched-melbo
```

Or just copy paste `src/batched_melbo.py` into your project.

## Usage

```python
from batched_melbo import batched_melbo

steering = batched_melbo.BatchedMELBO(
    ... # parameters
)
```

## Changes
1. Does not support orthogonalization of steering vectors.
2. Only residual stream vectors.
3. Adds support for mixed precision training.
4. Uses PyTorch hooks to steer the model instead of mutating the model.

## About Us
We are doing research on backdoors and sleeper agents, trying to understand the what makes them elicitable, removable, and detectable to create sleeper agents that are better organisms of misalignment. If you have ideas, reach out!