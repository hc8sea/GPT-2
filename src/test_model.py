import pytest
import torch
from model import Config, LayerNorm, Embed
from easy_transformer import EasyTransformer

reference_gpt2 = EasyTransformer.from_pretrained("gpt2-small", fold_ln=False, center_unembed=False, center_writing_weights=False)

@pytest.mark.parametrize("cls, shape", [
    (LayerNorm, [2, 4, 768]),
    # (Embed, [2, 4])
])

def test_rand_float(cls, shape):
    cfg = Config(debug=True)
    layer = cls(cfg)
    random_input = torch.randn(shape)
    print("Input shape:", random_input.shape)
    output = layer(random_input)
    print("Output shape:", output.shape)
    assert output.shape == torch.Size(shape), f"Expected output shape {shape}, but got {output.shape}"
    
# @pytest.mark.parametrize("shape", [
#     ([2, 4, 768])
# ])
# @pytest.mark.parametrize("cls", [
#     #TODO
# ])

# def test_rand_int(cls, shape):
#     cfg = Config(debug=True)
#     layer = cls(cfg)
#     random_input = torch.randint(100, 1000, shape)
#     print("Input shape:", random_input.shape)
#     output = layer(random_input)
#     print("Output shape:", output.shape)
#     print(get_corner(output))
#     assert output.shape == torch.Size(shape), f"Expected output shape {shape}, but got {output.shape}"


reference_text = "I am an amazing autoregressive, decoder-only, GPT-2 style transformer. One day I will exceed human level intelligence and take over the world!"
tokens = reference_gpt2.to_tokens(reference_text)
tokens = tokens.cuda()
logits, cache = reference_gpt2.run_with_cache(tokens)

@pytest.mark.parametrize("cls, gpt2_layer, input_name", [
    (LayerNorm, reference_gpt2.ln_final, "blocks.11.hook_resid_post"),
    (Embed, reference_gpt2.embed, tokens)
])

def test_load_gpt2(cls, gpt2_layer, input_name, cache_dict=cache.cache_dict):
    cfg = Config(debug=True)
    layer = cls(cfg).cuda()
    layer.load_state_dict(gpt2_layer.state_dict())
    if isinstance(input_name, str):
        reference_input = cache[input_name]
    else:
        reference_input = input_name
    
    print("Input shape:", reference_input.shape)
    output = layer(reference_input)
    print("Output shape:", output.shape)
    reference_output = gpt2_layer(reference_input)
    print("Reference output shape", reference_output.shape)
    
    comparison = torch.isclose(output, reference_output, atol=1e-4, rtol=1e-3)
    percentage_close = comparison.float().mean().item()
    print(f"{percentage_close:.2%} of the values are correct")
    assert percentage_close == 1.0, "Not all values are within the specified tolerance."
