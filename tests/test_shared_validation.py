"""Focused guards for shared automatic and legacy raw-pointer entry points."""
import pytest
import torch
import att3ntion._cuda_kernels as ck

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')

def inputs(batch=1,heads=8,device='cuda:0'):
    q=torch.zeros(batch,heads,16,128,device=device,dtype=torch.bfloat16)
    kv=[torch.zeros(batch,1,16,128,device=device,dtype=torch.bfloat16) for _ in range(4)]
    mask=torch.ones(batch,16,16,device=device,dtype=torch.bool).tril()
    y=torch.zeros_like(q)
    m=torch.zeros(batch,heads,16,device=device,dtype=torch.float32)
    l=torch.ones_like(m)
    return [q,*kv],mask,[y.clone(),y,m,l]

@pytest.mark.skipif(torch.cuda.device_count()<2,reason='requires two GPUs')
@pytest.mark.parametrize('mode',[0,1,2,4])
@pytest.mark.parametrize('field',[0,1,2,3],ids=['dY','Y','m','l'])
def test_shared_backward_rejects_foreign_state_device(mode,field):
    xs,mask,state=inputs()
    state[field]=state[field].to('cuda:1')
    dy,y,m,l=state
    with pytest.raises(RuntimeError,match="Q.*device"):
        ck.single_gather_shared_backward(dy,*xs,y,m,l,mask,16,mode)

@pytest.mark.skipif(torch.cuda.device_count()<2,reason='requires two GPUs')
@pytest.mark.parametrize('field',[0,1,2,3],ids=['dY','Y','m','l'])
def test_odd_heads_validate_state_before_padding(field):
    xs,mask,state=inputs(heads=3)
    state[field]=state[field].to('cuda:1')
    dy,y,m,l=state
    with pytest.raises(RuntimeError,match="Q.*device"):
        ck.single_gather_shared_backward(dy,*xs,y,m,l,mask,16,0)

@pytest.mark.parametrize('mode',[0,1,2,4])
@pytest.mark.parametrize('entry',['forward','backward'])
@pytest.mark.parametrize('batch,heads',[(0,8),(1,0)])
def test_shared_rejects_empty_batch_or_heads(mode,entry,batch,heads):
    xs,mask,(dy,y,m,l)=inputs(batch=batch,heads=heads)
    with pytest.raises(RuntimeError,match="empty batch/head"):
        if entry=='forward':ck.single_gather_shared_forward(*xs,mask,16,mode)
        else:ck.single_gather_shared_backward(dy,*xs,y,m,l,mask,16,mode)
