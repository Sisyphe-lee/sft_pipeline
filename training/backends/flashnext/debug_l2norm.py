"""IDE-only reproduction of long-context FLA L2Norm launch memory."""
import argparse
import importlib
import json
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--tokens', type=int, default=121051)
    parser.add_argument('--reserve-gib', type=int, default=0)
    args = parser.parse_args()
    norm = importlib.import_module('fla.modules.l2norm')
    if args.fixed:
        from training.backends.flashnext.attention import configure_fla_l2norm
        configure_fla_l2norm()
    def memory(stage):
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(json.dumps(dict(stage=stage, fixed=args.fixed, tokens=args.tokens,
                              allocated_gib=torch.cuda.memory_allocated()/2**30,
                              reserved_gib=torch.cuda.memory_reserved()/2**30,
                              free_gib=free/2**30, total_gib=total/2**30)), flush=True)
    memory('start')
    reservation = torch.empty(args.reserve_gib * 2**30, dtype=torch.uint8, device='cuda')
    memory('reserved')
    x = torch.randn(1, args.tokens, 48, 128, device='cuda', dtype=torch.bfloat16)
    dy = torch.randn_like(x)
    y, rstd = norm.l2norm_fwd(x)
    memory('forward')
    dx = norm.l2norm_bwd(y, rstd, dy)
    memory('backward')
    # Release artificial pressure before allocating correctness-check temporaries.
    del reservation
    torch.cuda.empty_cache()
    # Exact reference formula used by FLA; sample across the flattened sequence.
    for i in (0, args.tokens//2, args.tokens-1):
        yf, gf, rf = y[:, i].float(), dy[:, i].float(), rstd[:, i, :, None]
        expected = gf*rf - (gf*yf).sum(-1, keepdim=True)*yf*rf
        torch.testing.assert_close(dx[:, i].float(), expected, atol=.004, rtol=.02)
    assert torch.isfinite(dx).all()
    print('L2NORM_OK', flush=True)


if __name__ == '__main__':
    main()
