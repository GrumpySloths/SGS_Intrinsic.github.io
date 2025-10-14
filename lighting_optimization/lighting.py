import numpy as np
import torch
from torch import nn
from einops import einsum, rearrange
from arguments.config_r3dg import OptimizationParams

# ====================== INTENSITY ======================

class Constant(nn.Module):
    def __init__(self,
                 value,
                 exp_val=True):
        super().__init__()
        self.value = nn.Parameter(torch.tensor(value, dtype=torch.float32))
        self.exp_val = exp_val

    def forward(self, direction):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return val.unsqueeze(0).expand_as(direction)

    def reg_loss(self):
        val = self.value
        if self.exp_val:
            val = torch.exp(val)
        return torch.sum(val)


# 这里实际上是用一个全局的SG来表示环境光照
class MultipleSphericalGaussians(nn.Module):
    def __init__(self,
                 sg_col=6,
                 sg_row=2,
                 ch=3,
                 single_color=False,
                 w_lamb_reg=0):
        super().__init__()

        self.sg_col = sg_col
        self.sg_row = sg_row
        self.SGNum = self.sg_col * self.sg_row  #12

        self.single_color = single_color
        self.COLORNum = self.SGNum

        self.w_lamb_reg = w_lamb_reg

        self.ch = ch

        self.nearest_dist_sqr = None

        self.weight, self.theta, self.phi, self.lamb = self.init_sg_grid()

        is_enabled = torch.tensor(True)
        self.register_buffer('is_enabled', is_enabled)

    def training_setup(self):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=5e-2, eps=1e-15,weight_decay=False)
    
    def step(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def capture(self):
        captured_list = [
            self.parameters(),
            self.optimizer.state_dict(),
        ]

        return captured_list
    
    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        model_state, opt_state, first_iter = torch.load(checkpoint_path)
        self.model.load_state_dict(model_state)
        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception as e:
                print("Not loading optimizer state_dict!", e)
        return first_iter
    
    def set_requires_grad(self, requires_grad: bool = False):
        """
        Freeze or unfreeze all parameters in this module.

        Args:
            requires_grad (bool): If False, freeze all parameters. If True, unfreeze all parameters.
        """
        for param in self.parameters():
            param.requires_grad = requires_grad
        
    def init_sg_grid(self):
        phiCenter = ((np.arange(self.sg_col) + 0.5) / self.sg_col - 0.5) * np.pi * 2
        thetaCenter = (np.arange(self.sg_row) + 0.5) / self.sg_row * np.pi / 2.0

        phiCenter, thetaCenter = np.meshgrid(phiCenter, thetaCenter)

        thetaCenter = thetaCenter.reshape(self.SGNum, 1).astype(np.float32)
        thetaCenter = torch.from_numpy(thetaCenter).expand([self.SGNum, 1])

        phiCenter = phiCenter.reshape(self.SGNum, 1).astype(np.float32)
        phiCenter = torch.from_numpy(phiCenter).expand([self.SGNum, 1])

        thetaRange = (np.pi / 2 / self.sg_row) * 1.5
        phiRange = (2 * np.pi / self.sg_col) * 1.5

        self.register_buffer('thetaCenter', thetaCenter)
        self.register_buffer('phiCenter', phiCenter)
        self.register_buffer('thetaRange', torch.tensor(thetaRange))
        self.register_buffer('phiRange', torch.tensor(phiRange))

        weight = nn.Parameter(torch.ones((self.COLORNum, self.ch), dtype=torch.float32) * (0))
        theta = nn.Parameter(torch.zeros((self.SGNum, 1), dtype=torch.float32))
        phi = nn.Parameter(torch.zeros((self.SGNum, 1), dtype=torch.float32))
        lamb = nn.Parameter(torch.log(torch.ones(self.SGNum, 1) * np.pi / self.sg_row))

        return weight, theta, phi, lamb

    def deparameterize(self):
        theta = self.thetaRange * torch.tanh(self.theta) + self.thetaCenter

        phi = self.phiRange * torch.tanh(self.phi) + self.phiCenter
        lamb = self.deparameterize_lamb()

        weight = self.deparameterize_weight()

        return weight, theta, phi, lamb

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def deparameterize_lamb(self):
        lamb = torch.exp(self.lamb)
        return lamb

    def get_axis(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.cos(phi)
        axisY = torch.sin(theta) * torch.sin(phi)
        axisZ = torch.cos(theta)
        axis = torch.cat([axisX, axisY, axisZ], dim=1)

        return axis

    # def forward(self, direction):
    #     if self.is_enabled:
    #         weight, theta, phi, lamb = self.deparameterize()

    #         axis = self.get_axis(theta, phi)

    #         cos_angle = einsum(direction, axis, 'b c, sg c -> b sg')
    #         cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
    #         lamb = rearrange(lamb, 'sg 1 -> 1 sg 1')
    #         weight = rearrange(weight, 'sg c -> 1 sg c')
    #         sg_val = weight * torch.exp(lamb * (cos_angle - 1))
    #         sg_val = torch.sum(sg_val, dim=1) #[76800,3]

    #         return sg_val
    #     else:
    #         return torch.zeros_like(direction)

    def forward(self, direction, iteration=None, noise_std=30, noise_decay_iteration=10000):
        if self.is_enabled:
            weight, theta, phi, lamb = self.deparameterize()
            axis = self.get_axis(theta, phi)
            cos_angle = einsum(direction, axis, 'b c, sg c -> b sg')
            cos_angle = rearrange(cos_angle, 'b sg -> b sg 1')
            lamb = rearrange(lamb, 'sg 1 -> 1 sg 1')
            weight = rearrange(weight, 'sg c -> 1 sg c')
            sg_val = weight * torch.exp(lamb * (cos_angle - 1))
            sg_val = torch.sum(sg_val, dim=1) #[76800,3]

            # 训练初期加噪声扰动
            if (iteration is not None) and (iteration < noise_decay_iteration):
                std = noise_std * (1 - iteration / noise_decay_iteration)
                noise = torch.randn_like(sg_val) * std
                sg_val = sg_val + noise

            return sg_val
        else:
            return torch.zeros_like(direction)

    def reg_loss(self):
        if self.is_enabled:
            val = self.deparameterize_weight()
            val = torch.sum(val)

            if self.w_lamb_reg > 0:
                lamb_val = self.deparameterize_lamb()
                lamb_val = torch.sum(lamb_val)
                val += lamb_val * self.w_lamb_reg

            return val
        else:
            return torch.tensor(0, device=self.weight.device, dtype=torch.float32)
    
    @property
    def spp(self):
        return 1

    def sample_direction(self, vpos, normal):
        '''
            Sample directions from the light positions to the view positions.
            vpos:(3,h,w)
            normal:(3,h,w)
            return:(1,3,h,w)
        '''
        # (1, 3, h, w)
        return normal.unsqueeze(0)

    def pdf_direction(self, vpos, direction):
        # (1, 1, h, w)
        # return torch.ones_like(vpos[:, :, :1, ...])
        return torch.ones_like(vpos[None,:1,:,:])


# ====================== EMISSIVE LIGHTING ======================


class FusedSGGridPointLighting(nn.Module):
    def __init__(self,
                 num_lights=[3,3,3],  #[6,6,6]
                 bound=1.5,
                 sg_col=6,
                 sg_row=2,
                 ch=3,
                 single_color=False):
        super().__init__()

        self.num_lights = num_lights #[6,6,6]

        self.position = nn.Parameter(self.generate_grid_3d(np.array(num_lights), bound)) #这里的position也是可优化参数，表示光源的位置

        # Value init
        self.sg_col = sg_col
        self.sg_row = sg_row
        self.SGNum = self.sg_col * self.sg_row  #12

        self.single_color = single_color #false
        self.COLORNum = self.SGNum

        self.ch = ch

        self.nearest_dist_sqr = None
        #一个疑问就是这里的weight,theta,phi,lamb实际代指什么呢?这里实际代指的是每个点光源的SG参数，一个点光源通过多个SG来表示
        self.weight, self.theta, self.phi, self.lamb = self.init_sg_grid()

        is_enabled = torch.ones(self.spp, dtype=torch.bool)
        self.register_buffer('is_enabled', is_enabled)

    def training_setup(self,training_args: OptimizationParams):

        self.optimizer = torch.optim.Adam(self.parameters(), lr=training_args.pointlight_lr, eps=1e-15,weight_decay=False)
    
    def step(self, max_norm=1.0):
        # torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm)
        # for group in self.optimizer.param_groups:
        #     for param in group['params']:
        #         if param.grad is not None and torch.isnan(param.grad).any():
        #             raise RuntimeError("NaN detected in optimizer gradients")
        self.optimizer.step()
        self.optimizer.zero_grad()
    
    def capture(self):
        captured_list = [
            self.parameters(),
            self.optimizer.state_dict(),
        ]

        return captured_list

    def create_from_ckpt(self, checkpoint_path, restore_optimizer=False):
        model_state, opt_state, first_iter = torch.load(checkpoint_path)
        self.load_state_dict(model_state)
        if restore_optimizer:
            try:
                self.optimizer.load_state_dict(opt_state)
            except Exception as e:
                print("Not loading optimizer state_dict!", e)
        return first_iter

    def save_ckpt(self, checkpoint_path, current_iter=None):
        """
        保存模型和优化器的状态到 checkpoint_path。

        Args:
            checkpoint_path (str): 保存文件路径
            current_iter (int, optional): 当前训练轮次或步数
        """
        model_state = self.state_dict()
        opt_state = self.optimizer.state_dict() if hasattr(self, 'optimizer') else None
        # 保存当前轮次，便于恢复训练
        ckpt = (model_state, opt_state, current_iter)
        torch.save(ckpt, checkpoint_path)
    
    def init_sg_grid(self):
        phiCenter = ((np.arange(self.sg_col) + 0.5) / self.sg_col - 0.5) * np.pi * 2
        thetaCenter = (np.arange(self.sg_row) + 0.5) / self.sg_row * np.pi / 2.0

        phiCenter, thetaCenter = np.meshgrid(phiCenter, thetaCenter)

        thetaCenter = thetaCenter.reshape(1, self.SGNum, 1).astype(np.float32) #[1,12,1]
        thetaCenter = torch.from_numpy(thetaCenter).expand([1, self.SGNum, 1])

        phiCenter = phiCenter.reshape(1, self.SGNum, 1).astype(np.float32)
        phiCenter = torch.from_numpy(phiCenter).expand([1, self.SGNum, 1])

        thetaRange = (np.pi / 2 / self.sg_row) * 1.5
        phiRange = (2 * np.pi / self.sg_col) * 1.5

        self.register_buffer('thetaCenter', thetaCenter)  #这里注册到buffer的变量会随着整个模型一起保存，迁移和加载
        self.register_buffer('phiCenter', phiCenter)
        self.register_buffer('thetaRange', torch.tensor(thetaRange))
        self.register_buffer('phiRange', torch.tensor(phiRange))
        #这里的参数空间设置和对齐挺有意思的，值得参考和借鉴
        weight = nn.Parameter(torch.ones((self.spp, self.COLORNum, self.ch), dtype=torch.float32) * (-4)) #[216,12,3]
        theta = nn.Parameter(torch.zeros((self.spp, self.SGNum, 1), dtype=torch.float32)) #[216,12,1]
        phi = nn.Parameter(torch.zeros((self.spp, self.SGNum, 1), dtype=torch.float32)) #[216,12,1]
        lamb = nn.Parameter(torch.log(torch.ones(self.spp, self.SGNum, 1) * np.pi / self.sg_row)) #[216,12,1]

        return weight, theta, phi, lamb

    def generate_grid_3d(self, num_lights, bound):
        """
        Uniformly sample positions in 3D space within [-bound, bound]^3.

        Args:
            num_lights (list or tuple): Number of points along each axis, e.g., [6,6,6].
            bound (float): The boundary of the cube.

        Returns:
            torch.Tensor: Positions of shape (prod(num_lights), 3).
        """
        x = np.linspace(-bound, bound, num_lights[0], dtype=np.float32)
        y = np.linspace(-bound, bound, num_lights[1], dtype=np.float32)
        z = np.linspace(-bound, bound, num_lights[2], dtype=np.float32)
        grid = np.stack(np.meshgrid(x, y, z, indexing='ij'), axis=-1).reshape(-1, 3)
        return torch.from_numpy(grid)
    
    @property
    def spp(self):
        if isinstance(self.num_lights, int):
            return self.num_lights
        elif isinstance(self.num_lights, (tuple, list)):
            return int(np.prod(self.num_lights))
        else:
            raise NotImplementedError()

    def sample_direction(self, vpos, normal):
        '''
            Sample directions from the light positions to the view positions.
            vpos:(3,h,w)
            normal:(3,h,w)
            return:(spp,3,h,w)
        '''
        # (spp, 3, h, w),表示由每个pixel所对应的空间点指向光源的方向,position:(spp,3)
        if torch.isnan(self.position).any():
            raise RuntimeError("NaN detected in tensor position")
        return torch.nn.functional.normalize(self.position[:,:, None, None] - vpos[None], dim=1)
        # return torch.nn.functional.normalize(self.position[None, :, :, None, None] - vpos, dim=2)

    def pdf_direction(self, vpos, direction):
        # vpos: (3, h, w), self.position: (nlights, 3)
        # Compute squared distance from each light to each pixel
        # Output: (nlights, 1, h, w)
        vpos_exp = vpos.unsqueeze(0)  # (1, 3, h, w)
        pos_exp = self.position.unsqueeze(-1).unsqueeze(-1)  # (nlights, 3, 1, 1)
        dist_sqr = torch.sum((pos_exp - vpos_exp) ** 2, dim=1, keepdim=True)  # (nlights, 1, h, w)
        # Record nearest distance squared for each light
        self.nearest_dist_sqr = dist_sqr.view(self.spp, -1).min(dim=1).values

        return dist_sqr

    def deparameterize(self):
        theta = self.thetaRange * torch.tanh(self.theta) + self.thetaCenter

        phi = self.phiRange * torch.tanh(self.phi) + self.phiCenter
        lamb = self.deparameterize_lamb()

        weight = self.deparameterize_weight()

        return weight, theta, phi, lamb

    def deparameterize_weight(self):
        weight = torch.exp(self.weight)
        return weight

    def deparameterize_lamb(self):
        lamb = torch.exp(self.lamb)
        return lamb

    def get_axis(self, theta, phi):
        # Get axis
        axisX = torch.sin(theta) * torch.cos(phi)
        axisY = torch.sin(theta) * torch.sin(phi)
        axisZ = torch.cos(theta)
        axis = torch.cat([axisX, axisY, axisZ], dim=2)
        return axis

    def forward(self, direction):
        weight, theta, phi, lamb = self.deparameterize()
        #这里的get_axis本质上是将球面坐标转换为直角坐标
        axis = self.get_axis(theta, phi) #[48,12,3]
        #direction.shape:[48,76800,3],这里每个pixel到每个光源都要计算一次距离
        cos_angle = einsum(direction, axis, 'l b c, l sg c -> l b sg')
        cos_angle = rearrange(cos_angle, 'l b sg -> l b sg 1')
        lamb = rearrange(lamb, 'l sg 1 -> l 1 sg 1')
        weight = rearrange(weight, 'l sg c -> l 1 sg c')
        sg_val = weight * torch.exp(lamb * (cos_angle - 1)) #[48,76800,12,3]
        sg_val = torch.sum(sg_val, dim=2) #[48,76800,3]

        # Mask disabled lights
        sg_val = rearrange(self.is_enabled, 'l -> l 1 1') * sg_val

        # # Sum over the lights
        # sg_val = torch.sum(sg_val, dim=0)
        return sg_val

    def light_random_sample(self, direction, disable_prob=0.5, device=None, intensity_jitter=0.3, color_jitter=0.2):
        """
        Forward pass with a random mask to disable some point lights,
        and randomly perturb lighting intensity and color.

        Args:
            direction: (l, b, c) input directions.
            disable_prob: probability to disable each light (float in [0,1]).
            device: torch device for mask tensor (optional).
            intensity_jitter: max relative change for intensity (float, e.g. 0.3 for ±30%).
            color_jitter: max absolute change for color channels (float, e.g. 0.2 for ±0.2).

        Returns:
            sg_val: (l, b, c) output values with randomly masked and jittered lights.
            mask: (l,) the random boolean mask used.
        """
        weight, theta, phi, lamb = self.deparameterize()
        axis = self.get_axis(theta, phi)  # [l, sg, 3]
        cos_angle = einsum(direction, axis, 'l b c, l sg c -> l b sg')
        cos_angle = rearrange(cos_angle, 'l b sg -> l b sg 1')
        lamb = rearrange(lamb, 'l sg 1 -> l 1 sg 1')
        weight = rearrange(weight, 'l sg c -> l 1 sg c')
        sg_val = weight * torch.exp(lamb * (cos_angle - 1))  # [l, b, sg, c]
        sg_val = torch.sum(sg_val, dim=2)  # [l, b, c]

        l = direction.shape[0]
        if device is None:
            device = direction.device

        # Generate random mask for lights
        mask = (torch.rand(l, device=device) > disable_prob).to(dtype=sg_val.dtype)
        mask = rearrange(mask, 'l -> l 1 1')

        # Generate random intensity jitter per light
        intensity_scale = 1.0 + (torch.rand(l, device=device) * 2 - 1) * intensity_jitter  # [l]
        intensity_scale = rearrange(intensity_scale, 'l -> l 1 1')

        # Generate random color jitter per light (per channel)
        color_scale = 1.0 + (torch.rand(l, 1, sg_val.shape[2], device=device) * 2 - 1) * color_jitter  # [l,1,c]
        # Broadcast to [l, b, c]
        color_scale = color_scale.expand(-1, sg_val.shape[1], -1)

        sg_val = sg_val * intensity_scale * color_scale
        sg_val = mask * sg_val

        return sg_val, mask.squeeze()
    
    def val_reg_loss(self):
        val = self.deparameterize_weight()

        # Mask disabled lights
        val = rearrange(self.is_enabled, 'l -> l 1 1') * val

        val = torch.mean(val) * 3
        return val

    def pos_reg_loss(self):
        pos = 1 / self.nearest_dist_sqr.clamp(min=1e-6)

        # Mask disabled lights
        pos = rearrange(self.is_enabled, 'l -> l 1 1') * pos

        return torch.mean(pos)


# ====================== COMPOSITION ======================


class ComposeLighting(nn.Module):
    def __init__(self, lightings):
        super().__init__()
        self.lightings = nn.ModuleDict(lightings)

    @property
    def spp(self):
        return sum((lighting.spp for lighting in self.lighting_values))

    @property
    def sub_spps(self):
        return [lighting.spp for lighting in self.lighting_values]

    @property
    def lighting_values(self):
        return self.lightings.values()

    def position_init(self, vpos, normal, image):
        """
        Initialize the position of the lightings
        :param vpos:
        :param normal:
        :param image:
        :return:
        """
        for lighting in self.lighting_values:
            if hasattr(lighting, "position_init"):
                lighting.position_init(vpos=vpos, normal=normal, image=image)

    def sample_direction(self, vpos, normal):
        """
        Sample directions for each light sources separately
        :param vpos: BS x SPP x 3 x H x W
        :param normal: BS x SPP x 3 x H x W
        :return:
        """
        return torch.cat([lighting.sample_direction(vpos=vpos, normal=normal) for lighting in self.lighting_values], dim=1)

    def pdf_direction(self, vpos, direction):
        """
        Sample directions for each light sources separately
        :param vpos: BS x SPP x 3 x H x W
        :return:
        """
        return torch.cat([lighting.pdf_direction(vpos=vpos, direction=dir) for lighting, dir in zip(self.lighting_values, torch.split(direction, self.sub_spps, dim=1))], dim=1)

    def forward(self, direction):
        """
        Each direction goes for each light sources separately
        :param direction: N x 3
        :return:
        """   #self.sub_spps:[48,1]
        return torch.cat([lighting(direction=dir) for lighting, dir in zip(self.lighting_values, torch.split(direction, self.sub_spps))], dim=0)

    def val_reg_loss(self):
        """
        Calculate the regularization loss for each light sources separately
        :return:
        """
        val_regs = torch.stack([lighting.val_reg_loss() for lighting in self.lighting_values], dim=0)
        return torch.sum(torch.tensor(self.sub_spps, device=val_regs.device) * val_regs) / self.spp

    def pos_reg_loss(self):
        """
        Calculate the regularization loss for each light sources separately
        :return:
        """
        pos_regs = torch.stack([lighting.pos_reg_loss() for lighting in self.lighting_values], dim=0)
        return torch.sum(torch.tensor(self.sub_spps, device=pos_regs.device) * pos_regs) / self.spp

