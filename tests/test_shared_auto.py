"""Regression tests for automatic exact-training dispatch, run on remote H100s."""
import pytest
import torch

from att3ntion._single_gather import window_mask
from att3ntion._single_gather_shared import single_gather_shared_attention
from test_single_gather_shared import NAMES,GRADS,draw,run_reference,check_all

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

def run(xs,dy,mask,window):
    leaves={name:x.clone().requires_grad_(True) for name,x in xs.items()}
    y=single_gather_shared_attention(*[leaves[n] for n in NAMES],mask,window=window)
    y.backward(dy)
    return y,{grad:leaves[name].grad for grad,name in zip(GRADS,NAMES)}

@pytest.mark.parametrize('window',[16,32,64,128])
@pytest.mark.parametrize('heads',[1,3,16])
@pytest.mark.parametrize('scale',[1.0,3.0])
def test_singleton_score_gradients_are_exact_zero(window,heads,scale):
    n=33
    xs,dy=draw(2,heads,n,seed=113,qrs_scale=scale)
    mask=torch.zeros(2,n,n,dtype=torch.bool,device='cuda')
    q=torch.arange(n,device='cuda')
    mask[0,q,q]=True
    mask[1,q,torch.clamp(q-3,min=0)]=True
    mask[:,::11]=False
    want_y,want_lse,want_g=run_reference(xs,dy,mask)
    y,grads=run(xs,dy,mask,window)
    for name in ('dQ','dR','dS'):
        assert torch.count_nonzero(grads[name])==0
    check_all(y,grads,want_y,want_lse,want_g)

@pytest.mark.parametrize('window',[16,32,64,128])
@pytest.mark.parametrize('heads',[3,16])
@pytest.mark.parametrize('n',[33,129])
def test_default_training_with_holes_and_padding(window,heads,n):
    xs,dy=draw(2,heads,n,seed=191)
    mask=window_mask(n,window,'cuda')[None].expand(2,-1,-1).clone()
    mask[0,n//2]=False
    mask[1,:,::7]=False
    want_y,want_lse,want_g=run_reference(xs,dy,mask)
    y,grads=run(xs,dy,mask,window)
    check_all(y,grads,want_y,want_lse,want_g)

@pytest.mark.skipif(torch.cuda.device_count()<2,reason='requires two CUDA devices')
@pytest.mark.parametrize('window',[16,32,64,128])
def test_one_extension_can_switch_devices(window):
    xs,dy=draw(1,16,128,seed=193,device='cuda:0')
    mask=window_mask(128,window,'cuda:0')[None].contiguous()
    reference=None
    for device in (0,1,0):
        with torch.cuda.device(device):
            moved={n:t.to(device) for n,t in xs.items()}
            y,g=run(moved,dy.to(device),mask.to(device),window)
            got=[t.detach().cpu() for t in [y,*g.values()]]
            if reference is None:reference=got
            else:
                assert all(torch.equal(x,y) for x,y in zip(reference,got))

def test_automatic_training_replays_cuda_graph():
    xs,dy=draw(1,16,128,seed=197)
    mask=window_mask(128,32,'cuda')[None].contiguous()
    import att3ntion._cuda_kernels as ck
    def step():
        out=ck.single_gather_shared_forward(*[xs[n] for n in NAMES],mask,32,0)
        return out,ck.single_gather_shared_backward(dy,*[xs[n] for n in NAMES],*out,mask,32,0)
    for _ in range(3):step()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):out,grads=step()
    graph.replay();torch.cuda.synchronize()
    saved=[x.clone() for x in [*out,*grads]]
    graph.replay();torch.cuda.synchronize()
    assert all(torch.equal(x,y) for x,y in zip(saved,[*out,*grads]))
