import torch
try:
    device = torch.device('cuda:0')
    g = torch.cuda.manual_seed(1234567890)
except:
    device = torch.device('cpu')
    g = torch.manual_seed(1234567890)

x = torch.normal(0., 1., generator=g, size=(8, 8), dtype=torch.float32, device=device)

from torch.nn.functional import interpolate
x = interpolate(x[None, None], scale_factor=32, mode='bilinear', align_corners=False)
x.add_(torch.empty_like(x).normal_(generator=g).mul_(0.1))

from copy import deepcopy
from torch.nn import Parameter
from torch.optim import SGD
x_opt = Parameter(deepcopy(x), requires_grad=True)

opt = SGD([x_opt], lr=1e3)
from smooth import Smoothness, smoothness
smoothness = Smoothness(kernel_size=2).to(device)

from tqdm import trange

t = trange(1000)
for _ in t:
    y = -smoothness(x_opt)
    y.backward()
    t.set_description(f'{y.item():.3e}')
    
    opt.step()
    opt.zero_grad()

from matplotlib import pyplot as plt
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(x[0, 0].cpu())
ax[0].set_title('original')
ax[1].imshow(x_opt[0, 0].detach().cpu())
ax[1].set_title('optimized')
plt.savefig('example.png', dpi=200)
plt.show()
