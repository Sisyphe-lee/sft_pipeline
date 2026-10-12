"""Measure one full-length GDN forward/backward on an explicitly exposed GPU."""
import argparse
import json
import torch
from fla.ops.gated_delta_rule import chunk_gated_delta_rule


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tokens', type=int, default=121051)
    args = parser.parse_args()
    from training.backends.flashnext.attention import configure_fla_l2norm
    configure_fla_l2norm()
    def memory(stage):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(json.dumps(dict(stage=stage, tokens=args.tokens,
                              allocated_gib=torch.cuda.memory_allocated()/2**30,
                              peak_gib=torch.cuda.max_memory_allocated()/2**30,
                              reserved_gib=torch.cuda.memory_reserved()/2**30,
                              free_gib=free/2**30)), flush=True)
    memory('start')
    inputs = [torch.randn(1, args.tokens, 48, 128, device='cuda', dtype=torch.bfloat16,
                         requires_grad=True) for _ in range(3)]
    g = (-torch.rand(1, args.tokens, 48, device='cuda')).requires_grad_()
    beta = torch.rand(1, args.tokens, 48, device='cuda', dtype=torch.bfloat16, requires_grad=True)
    memory('inputs')
    out = chunk_gated_delta_rule(*inputs, g=g, beta=beta, use_qk_l2norm_in_kernel=True)[0]
    memory('forward')
    out.backward(torch.ones_like(out))
    memory('backward')
    for value in (*inputs, g, beta):
        for piece in value.grad.split(1024, dim=1):
            assert torch.isfinite(piece).all()
    print('GDN_LONG_OK', flush=True)


if __name__ == '__main__':
    main()
