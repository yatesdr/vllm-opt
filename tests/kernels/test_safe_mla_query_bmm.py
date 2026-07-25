# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _require_safe_mla_query_bmm():
    if not hasattr(torch.ops._C, "safe_mla_query_bmm"):
        pytest.skip("safe_mla_query_bmm is not built")


@pytest.mark.parametrize(
    "heads,tokens,q_dim",
    [
        (8, 1, 512),
        (8, 6, 512),
        (8, 11, 512),
        (11, 6, 512),
        (16, 6, 512),
        (8, 8192, 192),
    ],
)
@pytest.mark.parametrize("precise", [False, True])
def test_safe_mla_query_bmm_matches_torch_bmm(
    heads: int, tokens: int, q_dim: int, precise: bool
):
    _require_safe_mla_query_bmm()
    device = torch.device("cuda")
    rope_dim = 64
    latent_dim = 512
    torch.manual_seed(0)

    query_storage = torch.randn(
        tokens, heads, q_dim + rope_dim, dtype=torch.bfloat16, device=device
    )
    query = query_storage[..., :q_dim].transpose(0, 1)
    weight = torch.randn(heads, q_dim, latent_dim, dtype=torch.bfloat16, device=device)
    output = torch.empty(heads, tokens, latent_dim, dtype=torch.bfloat16, device=device)

    assert not query.is_contiguous()
    torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)
    expected = torch.bmm(query.contiguous(), weight)

    torch.testing.assert_close(output.float(), expected.float(), rtol=5e-2, atol=5e-2)


def test_safe_mla_query_bmm_three_argument_schema_compatibility():
    _require_safe_mla_query_bmm()
    device = torch.device("cuda")
    query = torch.randn(8, 1, 192, dtype=torch.bfloat16, device=device)
    weight = torch.randn(8, 192, 512, dtype=torch.bfloat16, device=device)
    output = torch.empty(8, 1, 512, dtype=torch.bfloat16, device=device)

    # The precise selector has a schema default so existing callers remain
    # source-compatible and retain the tensor-core-eligible route.
    torch.ops._C.safe_mla_query_bmm(query, weight, output)
    torch.cuda.synchronize(device)
    assert torch.isfinite(output).all()


def test_precise_safe_mla_query_bmm_preserves_fp8_boundary():
    """Cover the numerical boundary that the production route depends on."""
    _require_safe_mla_query_bmm()
    device = torch.device("cuda")
    heads = 8
    q_dim = 192
    latent_dim = 512
    saw_bf16_difference = False
    saw_fp8_difference = False

    for tokens in (4, 9, 16, 32):
        for seed in (7, 19, 41):
            torch.manual_seed(seed)
            query_storage = (
                torch.randn(
                    tokens,
                    heads,
                    q_dim + 64,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.5
            )
            query = query_storage[..., :q_dim].transpose(0, 1)
            weight = (
                torch.randn(
                    heads,
                    q_dim,
                    latent_dim,
                    dtype=torch.bfloat16,
                    device=device,
                )
                * 0.05
            )
            regular = torch.empty(
                heads,
                tokens,
                latent_dim,
                dtype=torch.bfloat16,
                device=device,
            )
            precise = torch.empty_like(regular)

            torch.ops._C.safe_mla_query_bmm(query, weight, regular, False)
            torch.ops._C.safe_mla_query_bmm(query, weight, precise, True)

            old_tf32 = torch.backends.cuda.matmul.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            try:
                reference = torch.bmm(query.float(), weight.float()).to(
                    torch.bfloat16
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = old_tf32

            regular_error = (regular.float() - reference.float()).abs().max()
            precise_error = (precise.float() - reference.float()).abs().max()
            assert precise_error <= regular_error
            saw_bf16_difference |= not torch.equal(regular, precise)
            saw_fp8_difference |= not torch.equal(
                regular.to(torch.float8_e4m3fn),
                precise.to(torch.float8_e4m3fn),
            )

    assert saw_bf16_difference
    assert saw_fp8_difference


@pytest.mark.parametrize("precise", [False, True])
def test_safe_mla_query_bmm_cuda_graph_replay(precise: bool):
    _require_safe_mla_query_bmm()
    device = torch.device("cuda")
    heads = 8
    tokens = 6
    q_dim = 512
    latent_dim = 512
    torch.manual_seed(1)

    query_storage = torch.randn(
        tokens, heads, q_dim + 64, dtype=torch.bfloat16, device=device
    )
    query = query_storage[..., :q_dim].transpose(0, 1)
    weight = torch.randn(heads, q_dim, latent_dim, dtype=torch.bfloat16, device=device)
    output = torch.empty(heads, tokens, latent_dim, dtype=torch.bfloat16, device=device)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)
    torch.cuda.current_stream().wait_stream(stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.safe_mla_query_bmm(query, weight, output, precise)

    graph.replay()
    graph.replay()
    expected = torch.bmm(query.contiguous(), weight)

    torch.testing.assert_close(output.float(), expected.float(), rtol=5e-2, atol=5e-2)
