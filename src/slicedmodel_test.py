# SlicedModel test for batched_melbo_validation.ipynb
# - does not run independently

from batched_melbo_sliced import SlicedModel

# Get actual model hidden states
tokens = tokenizer(EXAMPLES[0], return_tensors="pt", padding=True).to(model.device)
results = model(**tokens, return_dict=True, output_hidden_states=True)
logits, states = results.logits, results.hidden_states


# Compare with SlicedModel
source_layer_idx = 5
target_layer_idx = 15  # also can be -1 (final layer outputs), -2, etc.

# In SlicedModel, layer indices match hidden_states, so they are effectively resid_pre (start_layer=1 is layer 1 resid_pre)
source_sliced_model = SlicedModel(model, start_layer=0, end_layer=source_layer_idx)
steered_sliced_model = SlicedModel(model, start_layer=source_layer_idx, end_layer=target_layer_idx)

source_acts = source_sliced_model(**tokens)  # (1,38,2048)
dest_acts = steered_sliced_model(source_acts)
print(source_acts.shape, dest_acts.shape)

assert(torch.all(torch.isclose(states[source_layer_idx], source_acts)))
assert(torch.all(torch.isclose(states[target_layer_idx], dest_acts)))
