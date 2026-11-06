"""vkv.sketch: storage == the harness's quant_sketch / top_energy_pairs; kernel == torch."""
import pytest
import torch

from offload_package.vkv import reference as ref
from offload_package.vkv.sketch import SketchStore

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("D,pairs,bits,group,n", [(128, 16, 2, 128, 1000), (64, 8, 2, 32, 257),
                                                  (128, 16, 4, 64, 300), (256, 32, 2, 128, 129)])
def test_build_and_score(D, pairs, bits, group, n):
    H, L, S, G = 3, 2, 2, 4
    torch.manual_seed(0)
    st = SketchStore(L=L, H=H, D=D, max_len=1024, n_pairs=pairs, bits=bits, group=group,
                     device="cuda")
    st.grow(S)
    keys = (torch.randn(H, n, D, device="cuda") * torch.rand(1, 1, D, device="cuda") * 3).half()
    st.build(1, 1, keys)
    ch = ref.top_energy_pairs(keys.unsqueeze(0), pairs)
    assert torch.equal(st.channels[1, 1], ch[0])
    kc = keys.unsqueeze(0).gather(-1, ch[:, :, None, :].expand(-1, -1, n, -1))
    kh_ref = ref.quant_sketch(kc, bits, group)[0]
    slots = torch.tensor([1, 1], device="cuda")
    assert torch.equal(st.k_hat(1, slots[:1], n)[0], kh_ref)
    q = torch.randn(2, H * G, D, device="cuda")
    s_ref = torch.matmul(q.view(2, H, G, D).gather(-1, ch[0][None, :, None, :].expand(2, H, G, -1)),
                         kh_ref.unsqueeze(0).transpose(-1, -2)).view(2, H * G, n)
    s_t = st.score_torch(1, slots, q, n)
    s_k = st.score(1, slots, q, n)
    assert torch.allclose(s_t, s_ref, rtol=1e-5, atol=1e-5)
    assert torch.allclose(s_k, s_t, rtol=1e-5, atol=1e-5)
