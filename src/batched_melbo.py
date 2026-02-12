from __future__ import annotations
import torch
from torch import nn
from jaxtyping import *
from tqdm import tqdm
import functools, tqdm
import torch.optim as optim


def easy_generate(model, tokenizer, prompts: list[str], **kwargs):
    inputs = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    generations = model.generate(**inputs, **kwargs)
    return tokenizer.batch_decode(generations, skip_special_tokens=True)


def easy_forward(model, tokenizer, prompts: list[str], **kwargs):
    tokens = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    return model(**tokens, **kwargs)


# For applying steering vectors at a particular layer
class hooks():
    def __init__(self, model, hooks: list[tuple[torch.nn.Module, callable]]):
        """
        Args:
            model: The model to hook
            hooks: A list of tuples of the form (module, hook_fn)
                module: The module to hook
                hook_fn: The function to call when the hook is triggered. Should take the input and return the modified input.
        """
        self.model = model
        self.handles = []
        self.hooks = hooks

    def __enter__(self):
        for module, hook_fn in self.hooks:
            def post_hook(m, input, output):
                if isinstance(output, tuple):
                    modified_output = hook_fn(output[0])
                    return (modified_output,) + output[1:]

                return hook_fn(output)

            self.handles.append(module.register_forward_hook(post_hook))

    def __exit__(self, type, value, traceback):
        for handle in self.handles:
            handle.remove()


# SlicedModel code adapted from "Eliciting Latent Behaviors in Language Models" by Andrew Mack
# https://www.lesswrong.com/posts/fSRg5qs9TPbNy3sm5/deep-causal-transcoding-a-framework-for-mechanistically
# https://github.com/amack315/melbo-dct-post
def rgetattr(obj, path):
    return functools.reduce(getattr, path.split("."), obj)

class SlicedModel(nn.Module):
    def __init__(self, model, start_layer=None, end_layer=None, layers_name=None, attn_name=None):
        """
        Initializes a model that begins at `start_layer` and ends at `end_layer`.
        - If `start_layer`=0, begins at the start of the model and accepts input_ids= or input_embeds=.
        - If `end_layer`=-1, returns the last layer activations.
        :param model: Transformers model
        :param start_layer: same as hidden_layer (i.e. resid_pre of layer i)
        :param end_layer: same as hidden_layer (i.e. resid_pre of layer i)
        :param layers_name: the path to the layers list within `model` (attempts to infer when None)
        :param attn_name: the name of the attention block within each layer (attempts to infer when None)
        """
        super().__init__()
        self.model = model
        self.start_layer = start_layer
        self.end_layer = end_layer if end_layer >= 0 else model.config.num_hidden_layers + end_layer + 1
        if layers_name is None:
            if hasattr(self.model, "transformer"):  # gpt-2-like
                self.layers_name = "transformer.h"
            elif hasattr(self.model, "gpt_neox"): # pythia-like
                self.layers_name = "gpt_neox.layers"
            elif hasattr(self.model, "model"):  # mistral-like
                self.layers_name =  "model.model.layers"
            elif hasattr(self.model, "layers"):  # qwen2-like
                self.layers_name =  "model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name

        self.layers = rgetattr(self.model, self.layers_name)
        self.layers_name_split = self.layers_name.split(".")
        if attn_name is None:
            if hasattr(self.layers[0], "attn"):  # qwen
                self.attn_name = "attn"
            elif hasattr(self.layers[0], "self_attn"):   # qwen2-like
                self.attn_name = "self_attn"
            else:
                raise ValueError(f"don't know how to get attn name for {type(model)}")
        else:
            self.attn_name = attn_name

    def reset(self):
        setattr(self.model.config, "num_hidden_layers", self.depth)
        setattr(rgetattr(self.model, ".".join(self.layers_name_split[:-1])), self.layers_name_split[-1], self.layers)
        for i in range(len(self.layers)):
            rgetattr(self.layers[i], self.attn_name).layer_idx = i
        pass

    def forward(self, inputs_embeds=None, all_hidden_states=False, **kwargs):
        # mutate model so that forward pass only runs the specified middle layers

        try:
            # Original info
            self.depth = self.model.config.num_hidden_layers

            layers_name_split = self.layers_name_split

            # Sets the layers in the model only to the selected ones
            selected_layers = self.layers[self.start_layer:self.end_layer + 1]
            setattr(rgetattr(self.model, ".".join(layers_name_split[:-1])), layers_name_split[-1], selected_layers)

            # Sets the number of layers in the model to the number of selected layers
            setattr(self.model.config, "num_hidden_layers", self.end_layer - self.start_layer + 1)

            # Sets the layer index for each layer
            for i in range(len(rgetattr(self.model, self.layers_name))):
                rgetattr(rgetattr(self.model, self.layers_name)[i], self.attn_name).layer_idx = i

            # actually run the forward pass
            if self.start_layer == 0:
                hidden_states = self.model(**kwargs, output_hidden_states=True).hidden_states
            else:
                hidden_states = self.model(inputs_embeds=inputs_embeds, output_hidden_states=True).hidden_states

            if not all_hidden_states:
                hidden_states = hidden_states[self.end_layer - self.start_layer]
        finally:
            # reset model to un-mutated state (`finally` block in case of e.g. keyboard interrupt)
            self.reset()

        return hidden_states


# BatchedMELBO code adapted from "Mechanistically Eliciting Latent Behaviors in Language Models" by Andrew Mack
# https://www.lesswrong.com/posts/ioPnHKFyy4Cw2Gr2x/mechanistically-eliciting-latent-behaviors-in-language-1
# https://github.com/amack315/unsupervised-steering-vectors
def project_orthogonal_subspace(vec, learned_vectors, normalization):
    U = learned_vectors.t() / normalization
    result = vec - U @ U.t() @ vec
    return result

class BatchedMELBO():
    def __init__(
        self, 
        model, 
        tokenizer, 
        source_layer_idx=None, 
        target_layer_idx=None, 
        target_token_idxs=slice(None), 
        layers_name=None, 
        normalization=1.0, 
        num_steps=300, 
        power=2, 
        q=None,
        orthogonal_vectors=False,
        ):
        """
        - `source_layer_idx` and `target_layer_idx` both refer to the end-of-layer residuals at layer idx.
            - For compatibility, add one to any MELBO target layer idx from the original paper's code.
        - Defaults: source_layer_idx=7, target_layer_idx=n_layers - 9.
        - To steer the final pre-logit outputs, use target_layer_idx=-1. Does not support steering the embeds.
        """
        self.model = model
        self.tokenizer = tokenizer

        # determine layers object
        if layers_name is None:
            if hasattr(self.model, "transformer"):  # gpt-2-like
                self.layers_name = "transformer.h"
            elif hasattr(self.model, "gpt_neox"): # pythia-like
                self.layers_name = "gpt_neox.layers"
            elif hasattr(self.model, "model"):  # mistral-like
                self.layers_name =  "model.model.layers"
            elif hasattr(self.model, "layers"):  # qwen2-like
                self.layers_name =  "model.layers"
            else:
                raise ValueError(f"don't know how to get layer list for {type(model)}")
        else:
            self.layers_name = layers_name
        self.layers = rgetattr(self.model, self.layers_name)
        
        # determine source layer
        if source_layer_idx is None:
            self.source_layer_idx = 7
        elif source_layer_idx < 0:
            self.source_layer_idx = len(self.layers) + source_layer_idx  # -1 = final layer resid_post
        else:
            self.source_layer_idx = source_layer_idx
        
        # determine target layer
        if target_layer_idx is None:
            self.target_layer_idx = len(self.layers) - 9   # one less than original code, to ensure source/target use same layer indexes
        elif target_layer_idx < 0:
            self.target_layer_idx = len(self.layers) + target_layer_idx  # -1 = final layer resid_post
        else:
            self.target_layer_idx = target_layer_idx

        self.source_sliced_model = SlicedModel(model,
                                        start_layer=0,                       # hidden_sates=0 is layer 0 resid_pre
                                        end_layer=self.source_layer_idx+1,   # to match hidden_states
                                        layers_name=self.layers_name)

        self.steered_sliced_model = SlicedModel(model,
                                        start_layer=self.source_layer_idx+1, # to match hidden_states
                                        end_layer=self.target_layer_idx+1,   # to match hidden_states
                                        layers_name=self.layers_name)

        # get width
        self.width = model.config.hidden_size
        
        # set other hyper-parameters
        self.normalization = normalization
        self.target_token_idxs = target_token_idxs
        self.num_steps = num_steps
        self.power = power
        if q is None:
            self.q = self.power
        else:
            self.q = q

        self.orthogonal_vectors = orthogonal_vectors

        # don't need to store grads for parameters
        for param in self.model.parameters():
            param.requires_grad = False


    def train(self, examples, num_vectors, vector_batch_size=128):
        if isinstance(examples, str):
            examples = [examples]

        # initialize with random vectors
        self.num_vectors = num_vectors
        self.learned_vectors = torch.randn(self.num_vectors, self.width, device=self.model.device, dtype=self.model.dtype)
        self.learned_vectors = nn.functional.normalize(self.learned_vectors, dim=-1) * self.normalization
        
        # compute unsteered targets
        model_inputs = self.tokenizer(examples, return_tensors="pt", padding=True).to(self.model.device)
        with torch.no_grad():
            source_acts = self.source_sliced_model(**model_inputs)
            dest_acts = self.steered_sliced_model(source_acts)
            unsteered_targets = dest_acts[:, self.target_token_idxs]
        
        # loop over vectors
        losses_all = torch.zeros(num_vectors, self.num_steps)

        n_batches = (num_vectors // vector_batch_size) + (num_vectors % vector_batch_size > 0)
        for batch_start in range(0, num_vectors, vector_batch_size):
            batch_end = min(batch_start + vector_batch_size, num_vectors)
            batch_size = batch_end - batch_start

            repeated_unsteered_targets = unsteered_targets.repeat(batch_size, 1, 1)
            repeated_source_acts = source_acts.repeat(batch_size, 1, 1)  # (ex1, ex2, ex1, ex2, ..)
            biases = torch.randn(batch_size, 1, self.width, device=self.model.device, dtype=self.model.dtype, requires_grad=True)

            # initialize
            with torch.no_grad():
                if self.orthogonal_vectors:
                    for bias_idx in range(batch_size):
                        biases.data[bias_idx] = project_orthogonal_subspace(biases.data[bias_idx,0], self.learned_vectors[:batch_start], self.normalization)

                biases.data = self.normalization * nn.functional.normalize(biases.data, dim=-1)
       
            optimizer = optim.AdamW([biases], lr=.001, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

            # training loop
            for t in tqdm.tqdm(range(self.num_steps),
                               desc=f"Training batch {(batch_start//vector_batch_size)+1} of {n_batches}"):
                # compute gradient
                optimizer.zero_grad()

                # compute steered target
                steered_source_acts = repeated_source_acts + biases.repeat_interleave(len(examples), dim=0)
                steered_dest_acts = self.steered_sliced_model(steered_source_acts)
                steered_targets = steered_dest_acts[:, self.target_token_idxs] # batch, pos, width

                loss = -(steered_targets - repeated_unsteered_targets).norm(dim=-1).pow(self.power).sum(dim=-1).pow(1/self.q) # batch
                with torch.no_grad():
                    losses_all[batch_start:batch_end, t] = loss.data.detach().clone().view(len(examples), -1).mean(dim=0)
                loss.sum().backward()

                with torch.no_grad():
                    for bias_idx in range(batch_size):
                        # optionally project gradient to subspace orthogonal to previous learned vectors
                        if self.orthogonal_vectors:
                            biases.grad[bias_idx] = project_orthogonal_subspace(
                                biases.grad[bias_idx, 0],
                                torch.cat([self.learned_vectors[:batch_start], biases.data[:,0]], dim=0),
                                self.normalization
                            )

                        # project gradient to tangent space of sphere
                        biases.grad[bias_idx] -= torch.dot(
                            input=biases.grad[bias_idx, 0], 
                            tensor=biases[bias_idx, 0]
                        ) * biases[bias_idx] / (self.normalization**2)
                
                # step
                optimizer.step()

                # normalize
                with torch.no_grad():
                    biases.data = nn.functional.normalize(biases.data, dim=-1) * self.normalization
            
            with torch.no_grad():
                self.learned_vectors[batch_start:batch_end] = biases.data.detach()[:, 0]

        self.losses_all = losses_all.tolist()


    def steer(self, vector: int | slice | Float[torch.Tensor, "batch 1 width"]):
        """
        Steer the model by adding the vector to the source layer.
        Args:
            vector: The vector to steer the model by. Can be an integer index of the learned vectors, a tensor of shape (batch, 1, width), or a slice of the learned vectors.
        Returns:
            A context manager that can be used to steer the model.

        Example:
        ```python
        with batched_melbo.steer(0):
            hidden_states = model(**inputs, output_hidden_states=True).hidden_states
        ```
        """
        if isinstance(vector, int):
            vector = self.learned_vectors[vector]
        elif isinstance(vector, slice):
            vector = self.learned_vectors[vector].unsqueeze(1) # adds 1 to token position dimension to handle broadcasting
            
        return hooks(self.model, [
            (self.layers[self.source_layer_idx], lambda z: z + vector.to(z.dtype))
        ])
