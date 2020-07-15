from codebox.npbase import *
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset, random_split
from torch.utils.data.dataloader import default_collate
from torch.distributions.categorical import Categorical
from torch.nn.utils import spectral_norm
import torch.nn.functional as F
from codebox.utils import *
from torch.autograd import Variable
import torch.autograd as autograd
ftns = torch.FloatTensor
itns = torch.LongTensor

class Buffer(nn.Module):
    def __init__(self, buf):
        super().__init__()
        self.register_buffer('buf', buf)
    def forward(self, x):
        return x * Variable(self.buf)
class QuantileRegression(nn.Module):
    def __init__(self, natom=51, vmin=-10, vmax=10):
        super().__init__()
        self.atoms = Buffer(torch.FloatTensor(np.linspace(vmin, vmax, natom)))
    def forward(self, x):
        return torch.sum(self.atoms(x), -1)

def MakeTransportFunction(device_name='cuda', verbose=False, cpu_only=False):
    r'''detect device_count and make (GPU, CPU) function pairs'''
    if device_name=='cpu':
        cpu_only = True
    if verbose and cpu_only:
        print('[torchbase] cpu_only mode', file=sys.stderr)
    if torch.cuda.device_count()>0 and (not cpu_only):
        if verbose:
            print('[torchbase] GPU detected, use device_name=%s'%device_name, file=sys.stderr)
        GPU = lambda pytorch_obj: pytorch_obj.to(device_name)
        CPU = lambda pytorch_obj: pytorch_obj.cpu().data.numpy()
    else:
        if verbose:
            print('[torchbase] no GPU, transport functions are no-op', file=sys.stderr)
        GPU = lambda pytorch_obj: pytorch_obj
        CPU = lambda pytorch_obj: pytorch_obj.data.numpy()
    return GPU, CPU
def TransportList(todolist, trans_fn):
    r'''trans list of list to GPU'''
    return [
        TransportList(cmpo, trans_fn)
        if type(cmpo)==type([0.0]) or type(cmpo)==type((1, -1)) else 
        trans_fn(cmpo)  for cmpo in todolist]
def MakeSeq(module_list: list)->ClassVar:
    r'''flatten list of list of modules into nn.Sequential for you'''
    module = nn.Sequential(*FlattenList(module_list))
    return module

def NoGrad(module):
    for param in module.parameters():
        param.requires_grad = False
    return module

def ModelSize(module):
    r = 0
    for param in module.parameters():
        r += np.prod(param.size())
    return int(r)

def UnionDataset(list_of_list_of_dataset: list):
    r'''Flatten list of list and concat dataset_list'''
    return ConcatDataset(FlattenList(list_of_list_of_dataset))

def IndexTensor(x, dim=-1):
    return torch.arange(0, x.size(dim),
        dtype=x.dtype, device=x.device, requires_grad=False)

def EnergyBind(x, max_prob=0.95, dim=-1):
    d = x.size(dim)
    b = 0.5 * math.log(max_prob/((1-max_prob)/(int(d)-1)))
    return F.hardtanh(x, -b, b)

def SafeMax(x, max_prob=0.95, dim=-1):
    r'''cliped softmax, make it safer to learn'''
    return torch.softmax(EnergyBind(x, max_prob, dim), dim=dim)

class SafeMaxLayer(nn.Module):
    r'''same as SafeMax, just for nn.Sequential'''
    def __init__(self, max_prob=0.95, dim=-1):
        super().__init__()
        self.max_prob = max_prob
        self.dim = dim
    def forward(self, x):
        return SafeMax(x, self.max_prob, self.dim)

class GLULayer1d(nn.Module):
    r'''(B, in) -> (B, out), auto skip if in==out'''
    def __init__(self, in_features, out_features):
        super().__init__()
        self.skip = in_features==out_features
        self.f1 = nn.Linear(in_features, out_features*2)
        self.f1_bn = nn.BatchNorm1d(out_features*2, eps=0.8)
    def forward(self, x):
        fx = F.glu(self.f1_bn(self.f1(x)), dim=-1)
        if self.skip:
            return x + fx
        return fx
class GLUNet1d(nn.Sequential):
    r'''(B, in) -> (B, out), many GLULayer1d, auto skip'''
    def __init__(self, in_features, out_features, num_layers=1):
        super().__init__()
        self.add_module('f1', GLULayer1d(in_features, out_features))
        for i in range(num_layers-1):
            self.add_module('f%d'%(i+2), GLULayer1d(out_features, out_features))

class Resblock(nn.Module):
    def __init__(self, inch, outch, down=False):
        super().__init__()
        self.down = down
        if down:
            self.down_res = MakeSeq([
                nn.ReplicationPad2d(1),
                nn.Conv2d(inch, outch, 3, 2),
                nn.BatchNorm2d(outch, 0.8)
            ])
            conv1 = nn.Conv2d(inch, outch, 3, 2)
        else:
            if inch!=outch:
                self.down=True
                self.down_res = MakeSeq([
                    nn.ReplicationPad2d(1),
                    nn.Conv2d(inch, outch, 3),
                    nn.BatchNorm2d(outch, 0.8)
                ])
            conv1 = nn.Conv2d(inch, outch, 3)
        self.f = MakeSeq([
            nn.ReplicationPad2d(1),
            conv1,
            nn.BatchNorm2d(outch, 0.8),
            nn.PReLU(),
            nn.ReplicationPad2d(1),
            nn.Conv2d(outch, outch, 3),
            nn.BatchNorm2d(outch, 0.8),
        ])
    def forward(self, x):
        res = self.down_res(x) if self.down else x
        fx = self.f(x)
        return fx+res
        
class RandomNet1d(nn.Module):
    def __init__(self, in_dim=15, out_dim=15, seed=303902):
        r'''
        random map 1d vector to normalized vector
        useful when input are uniform distribution
        '''
        super().__init__()
        self.out_dim = out_dim
        torch.manual_seed(seed)
        self.f = MakeSeq([
            nn.Linear(in_dim, out_dim),
            nn.GroupNorm(1, out_dim, 0.8, affine=False)
        ])
    def forward(self, x):
        with torch.no_grad():
            y = self.f(x)
        return y

class RandomNet2d(nn.Module):
    def __init__(self, in_chn=3, out_chn=8, grid=4, grid_k=8, default_img_size=128, seed=303902):
        r'''
        (B, in_chn, W, H) -> (B, out_chn*grid*grid)
        random map 2d img to normalized vector
        '''
        assert grid_k>=grid
        assert grid_k%grid==0
        super().__init__()
        self.out_dim = out_chn*grid*grid
        ksize = default_img_size // grid_k
        torch.manual_seed(seed)
        conv = nn.Conv2d(in_chn, out_chn, [ksize, ksize], [ksize, ksize])
        self.f = MakeSeq([
            conv,
            nn.AdaptiveAvgPool2d([grid, grid]), # protect 256x256 input
            nn.GroupNorm(1, conv.out_channels, 0.8, affine=False)
        ])
    def forward(self, x):
        with torch.no_grad():
            y = self.f(x).view(x.size(0), -1)
        return y

class MinibatchVar1d(nn.Module):
    def forward(self, x):
        B, N = x.size()
        x2m = torch.mean(torch.pow(x, 2), dim=0, keepdim=True)
        xm2 = torch.pow(torch.mean(x, dim=0, keepdim=True), 2)
        xm = x2m - xm2
        xm = xm.view(1, N).expand(B, N)
        return xm
    

class MinibatchMoment2d(nn.Module):
    r'''append variance among minibatch as a new channel'''
    def __init__(self, mlist=[1, 2, 3]):
        super().__init__()
        self.mlist = mlist
        self.out_channels = len(mlist)
    def forward(self, x):
        B, C, W, H = x.size()
        x0 = x.view(B*C, W, H)
        fet = [x]
        for m in self.mlist:
            xm = torch.mean(torch.pow(x0, m), dim=0, keepdim=True)
            xm = xm.view(1, 1, W, H)
            xm = xm.expand(B, 1, W, H)
            fet.append(xm)
        return torch.cat(fet, dim=1)

# source: https://github.com/nashory/pggan-pytorch/blob/master/custom_layers.py
class PixelNorm(nn.Module):
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    def forward(self, x):
        return x / (torch.mean(x**2, dim=1, keepdim=True) + self.eps) ** 0.5
        # EX1 = torch.mean(x, dim=1, keepdim=True)
        # EX2 = torch.mean(x**2, dim=1, keepdim=True)
        # return (x-EX1) / (EX2 + self.eps) ** 0.5
    def __repr__(self):
        return f'PixelNorm(eps={self.eps})'

def TensorFin(tns):
    fin, _ = torch.nn.init._calculate_fan_in_and_fan_out(tns)
    return fin
class Equalized(nn.Module):
    r'''nvidia equalized learning rate'''
    def __init__(self, m):
        super().__init__()
        self.m = m
        torch.nn.init.normal_(m.weight.data, 0.0, self.param['std1'])
        fin = TensorFin(m.weight.data)
        self.scale = (2/(fin))**.5
        if hasattr(m, 'bias'):
            torch.nn.init.normal_(m.bias.data, 0.0, self.param['std2']*self.scale)
        # print(f'[Equalized] '
        #     f'{m.__class__.__name__} '
        #     f'tns({m.weight.data.size()}) '
        #     f'fin={fin} scale={self.scale:.8f}', file=sys.stderr)
    @property
    def param(self):
        return {
            'std1': 1.0,
            'std2': 0.1,
        }
    def forward(self, x):
        return self.m(self.scale * x)
    def __repr__(self):
        return f'Equalized({self.m.__repr__()})'

def _Wrapper(SN: bool):
    if SN:
        return lambda mod: Equalized(mod)
    else:
        # return lambda mod: spectral_norm(Equalized(mod))
        return lambda mod: Equalized(spectral_norm(mod)) # same???

# --------==== 1d ====--------
def AddLinear(layer: list, in_features: int, out_features: int, SN=False):
    wrapper = _Wrapper(SN)
    layer.append(wrapper(nn.Linear(in_features, out_features)))
    return layer
class WrappableGLU(nn.Module):
    def __init__(self, m, dim, skip=False):
        super().__init__()
        self.m = m
        self.dim = dim
        self.skip = skip
    def forward(self, x):
        fx = F.glu(self.m(x), dim=self.dim)
        if self.skip:
            return x+fx
        return fx
class WrapSkippable(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m
        self.gamma = torch.nn.Parameter(ftns([0.0]))
    def forward(self, x):
        fx = self.m(x)
        if fx.size(1)==x.size(1):
            g = self.gamma.sigmoid()
            return (1-g)*x+g*fx
        return fx
def AddLinearGLU(layer: list, in_features: int, out_features: int, SN=False):
    wrapper = _Wrapper(SN)
    layer.append(WrappableGLU(
        wrapper(nn.Linear(in_features, out_features*2)),
        dim=1, skip=(in_features==out_features)))
    return layer
def AddLinearTunnel(layer: list, in_features: int, out_features: int, hidden: int, nlayer: int=0, norm: str='None', GLU=False, SN=False):
    wrapper = _Wrapper(SN)
    if nlayer==0:
        AddLinear(layer, in_features, out_features, SN=SN)
        AddNorm1d(layer, norm, out_features)
        return layer
    for i in range(nlayer):
        if GLU:
            AddLinearGLU(layer, in_features if i==0 else hidden, hidden, SN=SN)
        else:
            if i>0:
                layer.append(nn.LeakyReLU(0.2, True))
            AddLinear(layer, in_features if i==0 else hidden, hidden, SN=SN)
    
    AddNorm1d(layer, norm, hidden)
    AddLinear(layer, hidden, out_features, SN=SN)
    return layer

# --------==== Conv ====--------
def AddNorm(layer, norm, out, **kwargs):
    if norm=='PixelNorm':
        layer.append(PixelNorm(**kwargs))
    if norm=='LayerNorm':
        layer.append(nn.GroupNorm(1, out, **kwargs))
    if norm=='InstanceNorm':
        layer.append(nn.GroupNorm(out, out, **kwargs))
    if norm=='BatchNorm':
        layer.append(nn.BatchNorm2d(out, **kwargs))
    return layer
def AddNorm1d(layer, norm, out, **kwargs):
    if norm=='PixelNorm':
        layer.append(PixelNorm(**kwargs))
    if norm=='LayerNorm':
        layer.append(nn.GroupNorm(1, out, **kwargs))
    if norm=='InstanceNorm':
        layer.append(nn.GroupNorm(out, out, **kwargs))
    if norm=='BatchNorm':
        layer.append(nn.BatchNorm1d(out, **kwargs))
    return layer
def AddConvTail(layer: list, norm: str, out=None, negative_slope=0.2, SN=False, **kwargs):
    r'''pad->conv->[ReLU->Norm]'''
    wrapper = _Wrapper(SN)
    layer.append(nn.LeakyReLU(negative_slope, True))
    AddNorm(layer, norm, out, **kwargs)
    return layer
def AddConv3x3(layer: list, in_channels: int, out_channels: int, norm: str, stride: int=1, SN=False, skip=False, **kwargs):
    r'''
    same_pad + conv3x3 + stride1 + leakyRELU
    norm = {PixelNorm, LayerNorm, SN, None}'''
    wrapper = _Wrapper(SN)
    layertmp = []
    layertmp.append(nn.ReplicationPad2d(1))
    layertmp.append(wrapper(nn.Conv2d(in_channels, out_channels, 3, stride, padding=0)))
    AddConvTail(layertmp, norm, out=out_channels, **kwargs)
    if skip: # skip always comes with two layers
        layertmp.append(nn.ReplicationPad2d(1))
        layertmp.append(wrapper(nn.Conv2d(out_channels, out_channels, 3, stride, padding=0)))
        AddConvTail(layertmp, norm, out=out_channels, **kwargs)
        layerseq = nn.Sequential(*layertmp)
        layerseq = WrapSkippable(layerseq)
    else:
        layerseq = nn.Sequential(*layertmp)
    layer.append(layerseq)
    return layer
def AddConv3x3GLU(layer: list, in_channels: int, out_channels: int, norm: str, stride: int=1, SN=False, skip=False, **kwargs):
    r'''same_pad + conv3x3 + stride1 + GLU'''
    wrapper = _Wrapper(SN)
    layertmp = []
    layertmp.append(nn.ReplicationPad2d(1))
    layertmp.append(wrapper(nn.Conv2d(in_channels, out_channels*2, 3, stride, padding=0)))
    layertmp.append(nn.GLU(dim=1))
    AddNorm(layertmp, norm, out_channels, **kwargs)
    if skip: # skip always comes with two layers
        layertmp.append(nn.ReplicationPad2d(1))
        layertmp.append(wrapper(nn.Conv2d(out_channels, out_channels*2, 3, stride, padding=0)))
        layertmp.append(nn.GLU(dim=1))
        AddNorm(layertmp, norm, out_channels, **kwargs)
        layerseq = nn.Sequential(*layertmp)
        layerseq = WrapSkippable(layerseq)
    else:
        layerseq = nn.Sequential(*layertmp)
    layer.append(layerseq)
    return layer
    
def AddConv1x1(layer: list, in_channels: int, out_channels: int, norm: str, SN=False, **kwargs):
    r'''conv1x1 + leakyRELU + norm'''
    wrapper = _Wrapper(SN)
    layer.append(wrapper(nn.Conv2d(in_channels, out_channels, 1, 1, padding=0)))
    AddConvTail(layer, norm, out=out_channels, **kwargs)
    return layer
def AddConv1x1GLU(layer: list, in_channels: int, out_channels: int, norm: str, SN=False, **kwargs):
    r'''conv1x1 + leakyRELU + norm'''
    wrapper = _Wrapper(SN)
    layer.append(wrapper(nn.Conv2d(in_channels, out_channels*2, 1, 1, padding=0)))
    layer.append(nn.GLU(dim=1))
    AddNorm(layer, norm, out=out_channels, **kwargs)
    return layer
def AddPixel4x4(layer: list, in_channels: int, out_channels: int, kernel_size: int, norm: str, SN=False, **kwargs):
    r'''(in, 1, 1)->(out, 4, 4), kind of fully connected'''
    wrapper = _Wrapper(SN)
    layer.append(wrapper(nn.ConvTranspose2d(in_channels, out_channels, kernel_size, 1, padding=0)))
    AddNorm(layer, norm, out_channels, **kwargs)
    # AddConvTail(layer, norm, out=out_channels, **kwargs)
    return layer
def Add4x4Pixel(layer: list, in_channels: int, out_channels: int, kernel_size: int, norm: str, SN=False, **kwargs):
    r'''(in, 4, 4)->(out, 1, 1), kind of fully connected'''
    wrapper = _Wrapper(SN)
    layer.append(wrapper(nn.Conv2d(in_channels, out_channels, kernel_size, 1, padding=0)))
    AddNorm(layer, norm, out_channels, **kwargs)
    # AddConvTail(layer, norm, out=out_channels, **kwargs)
    return layer
def AddAttention(layer: list, channels: int, SN=False, **kwargs):
    r'''conv1x1 + leakyRELU + norm'''
    wrapper = _Wrapper(SN)
    convQ = wrapper(nn.Conv2d(channels, channels, 1, 1, padding=0))
    convK = wrapper(nn.Conv2d(channels, channels, 1, 1, padding=0))
    convV = wrapper(nn.Conv2d(channels, channels, 1, 1, padding=0))
    layer.append(FSkipAttentionRow(convQ, convK, convV, dim=2))
    # AddConvTail(layer, norm, out=out_channels, **kwargs)
    return layer

# class HingeLoss(nn.Module):
#     def forward(self, y_pred, y_true, c)

class FAttention1d(nn.Module):
    def __init__(self, dim):
        r'''
        functional module calculates attention
        says dim=2 below
        input:
        (B, *, Q, d) query
        (B, *, K, d) key
        (B, *, K, v) value
        output:
        (B, *, Q, v)
        '''
        super().__init__()
        self.dim = dim
    def forward(self, qry, key, val, reg=0):
        qdim = self.dim
        ddim = self.dim+1
        # dot_scale = key.size(ddim) ** -0.5
        dot_scale = 1
        key = key.transpose(qdim, ddim)
        att = torch.matmul(qry, key)
        if reg>0:
            att += reg * torch.eye(att.size(qdim), out=torch.empty_like(att))
        att = torch.softmax(att * dot_scale, dim=qdim)
        out = torch.matmul(att, val)
        return out

class FSkipAttentionRow(nn.Module):
    def __init__(self, convQ, convK, convV, dim=2):
        super().__init__()
        self.att_fn = FAttention1d(dim)
        self.gamma = torch.nn.Parameter(ftns([2]))
        self.convQ = convQ
        self.convK = convK
        self.convV = convV
    def forward(self, x):
        Q = self.convQ(x)
        K = self.convK(x)
        V = self.convV(x)
        g = self.gamma.sigmoid()
        return self.att_fn(Q, K, V, reg=g)

class FHingeLoss01(nn.Module):
    r'''input:(score, hope={0, 1}) -> max(0, 1-ty)'''
    def forward(self, score, hope):
        yposneg = 2.0*(hope-0.5)
        onetns = torch.ones_like(score)
        return torch.mean(F.relu(onetns-score*yposneg))
class FVector2Pixel(nn.Module):
    def forward(self, x):
        B = x.size(0)
        return x.view(B, -1, 1, 1)
