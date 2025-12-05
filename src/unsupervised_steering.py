from __future__ import annotations
import torch.nn.functional as F
import torch as t
import torch
from jaxtyping import *
from tqdm import tqdm
import functools, tqdm
import torch as t
from torch import nn
import torch.optim as optim

# %%
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

# %%
def easy_generate(model, tokenizer, prompts: list[str], **kwargs):
    inputs = tokenizer(prompts, return_tensors='pt', padding=True, padding_side='left').to(model.device)
    generations = model.generate(**inputs, **kwargs)
    return tokenizer.batch_decode(generations, skip_special_tokens=True)

def easy_forward(model, tokenizer, prompts: list[str], **kwargs):
    tokens = tokenizer(prompts, return_tensors='pt', padding=True).to(model.device)
    return model(**tokens, **kwargs)

# %%
#| export
def rgetattr(obj, path):
    return functools.reduce(getattr, path.split("."), obj)

def project_orthogonal_subspace(vec, learned_vectors, normalization):
    U = learned_vectors.t() / normalization
    result = vec - U @ U.t() @ vec
    return result

class FastMELBO():
    def __init__(
        self, 
        model, 
        tokenizer, 
        source_layer_idx=None, 
        target_layer_idx=None, 
        target_token_idxs=slice(None), 
        layers_name=None, 
        source_module_name=None, 
        normalization=1.0, 
        num_steps=300, 
        power=2, 
        q=None, 
        orthogonal_vectors=False, 
        target_module="residual",
        on_step_end_cb: Callable[[t.Tensor, t.Tensor], None] = None):
        '''
        Note: this will mutate `model`
        '''
        self.model = model
        self.tokenizer = tokenizer

        if orthogonal_vectors:
            raise NotImplementedError("Orthogonal vectors are not implemented for FastMELBO")
            
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
        else:
            self.source_layer_idx = source_layer_idx
        
        # determine target layer
        if target_layer_idx is None:
            self.target_layer_idx = len(self.layers) - 8
        else:
            self.target_layer_idx = target_layer_idx
        
        # determine source_module_name
        if source_module_name is None:
            if "QWen" in type(self.model).__name__:
                self.source_module_name = "mlp.c_proj"
            elif hasattr(self.model, "gpt_neox"):
                self.source_module_name = "mlp.dense_4h_to_h"
            else:
                self.source_module_name = "mlp.down_proj" # otherwise guess "down_proj"
        else:
            self.source_module_name = source_module_name
            
        # get width
        self.width = rgetattr(self.layers[0], self.source_module_name).out_features
        
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
        self.target_module = target_module

        # don't need to store grads for most parameters
        for param in self.model.parameters():
            param.requires_grad = False
            
    def train(self, examples, num_vectors, vector_batch_size=128):
        if isinstance(examples, str):
            examples = [examples]
        
        enable_autocast = self.model.dtype in (torch.bfloat16, torch.float16)
        self.num_vectors = num_vectors
        # initialize with random vectors
        self.learned_vectors = torch.randn(self.num_vectors, self.width, device=self.model.device)
        self.learned_vectors = nn.functional.normalize(self.learned_vectors, dim=-1) * self.normalization

        num_steps = self.num_steps
        normalization = self.normalization
        power = self.power
        
        # compute unsteered targets
        model_inputs = self.tokenizer(examples, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)
        with torch.no_grad(), torch.autocast(device_type=self.model.device.type, dtype=self.model.dtype, enabled=enable_autocast):
            hidden_states = self.model(model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], output_hidden_states=True).hidden_states
        unsteered_targets = hidden_states[self.target_layer_idx+1][:, self.target_token_idxs, :]
        
        # loop over vectors
        losses_all = torch.zeros(num_vectors, num_steps)

        for batch_start in range(0, num_vectors, vector_batch_size):
            batch_end = min(batch_start + vector_batch_size, num_vectors)
            batch_size = batch_end - batch_start

            repeated_unsteered_targets = unsteered_targets.repeat(batch_size, 1, 1)
            
            biases = torch.zeros(batch_size, 1, self.width, device=self.model.device, requires_grad=True)

             # initialize
            # batch_losses = []
            with torch.no_grad():
                biases.data = normalization*nn.functional.normalize(torch.randn(batch_size, 1, self.width, device=self.model.device), dim=-1)
       
            optimizer = optim.AdamW([biases], lr=.001, betas=(.9,.98), weight_decay=0.0, amsgrad=True)

            model_inputs = self.tokenizer(examples*batch_size, return_tensors="pt", padding=True, padding_side="left").to(self.model.device)

            # training loop
            for t in tqdm.tqdm(range(num_steps), desc=f"Training batch {(batch_start+1)//vector_batch_size} of {num_vectors//vector_batch_size}"):
                # compute gradient
                optimizer.zero_grad()    

                # compute steered target
                with torch.autocast(device_type=self.model.device.type, dtype=self.model.dtype, enabled=enable_autocast), self.steer(biases.repeat_interleave(len(examples), dim=0)):
                    hidden_states = self.model(model_inputs["input_ids"], attention_mask=model_inputs["attention_mask"], output_hidden_states=True).hidden_states
                    target = hidden_states[self.target_layer_idx+1][:, self.target_token_idxs, :] # batch, pos, width
                    
                    loss = -(target - repeated_unsteered_targets).norm(dim=-1).pow(power).sum(dim=-1).pow(1/self.q) # batch
                    with torch.no_grad():
                        losses_all[batch_start:batch_end, t] = loss.data.detach().clone().view(len(examples), -1).mean(dim=0)
                loss.sum().backward()

                # project gradient to tangent space of sphere
                with torch.no_grad():
                    for bias_idx in range(batch_size):
                        biases.grad[bias_idx] -= torch.dot(
                            input=biases.grad[bias_idx, 0], 
                            tensor=biases[bias_idx, 0]
                        ) * biases[bias_idx] / (normalization**2)
                
                # step
                optimizer.step()

                # normalize
                with torch.no_grad():
                    biases.data = nn.functional.normalize(biases.data, dim=-1) * normalization
            
            with torch.no_grad():
                self.learned_vectors[batch_start:batch_end] = biases.data.detach()[:, 0]

        self.losses_all = losses_all.tolist()
        return self.losses_all

    def steer(self, vector: int | Float[t.Tensor, "batch 1 width"]):
        vector = self.learned_vectors[vector] if isinstance(vector, int) else vector
        return hooks(self.model, [
            (self.layers[self.source_layer_idx], lambda z: z+vector.to(z.dtype))
        ])
