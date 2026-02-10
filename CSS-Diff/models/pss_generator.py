
import torch
import torch.nn as nn
import yaml
from .PSS_modules.denoising import generalized_steps
from .PSS_modules.diffusion import get_beta_schedule
from .PSS_modules.ckpt_util import get_ckpt_path
from .PSS_modules.ema import EMAHelper
from .PSS_modules.utils import dict2namespace
from .PSS_modules.diffusion import Model
from .PSS_modules.q_sample import q_sample  # 加入正向扩散

class PSSGenerator(nn.Module):
    def __init__(self, config=None, ckpt_path=None, device=None):
        super().__init__()
        
        self.config = config
        self.device = torch.device(device)
        self.model = Model(self.config, self.device).to(self.device)
        self.model.eval()

        # Load pretrained checkpoint
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location=self.device)
            if 'model' in state_dict:
                state_dict = state_dict['model']
            self.model.load_state_dict(state_dict, strict=True)

        self.betas = get_beta_schedule(
            beta_schedule=self.config['diffusion']['beta_schedule'],
            beta_start=self.config['diffusion']['beta_start'],
            beta_end=self.config['diffusion']['beta_end'],
            num_diffusion_timesteps=self.config['diffusion']['num_diffusion_timesteps'],
        )
        self.betas = torch.tensor(self.betas).float().to(self.device)

    # def forward(self, x, seq=None, eta=0.0, trainable=True, cond=None):
    #     if seq is None:
    #         seq = list(range(0, self.config['diffusion']['num_diffusion_timesteps'], 10))
    
    #     if trainable:
    #         xs, x0_preds = generalized_steps(
    #             x, seq, self.model, self.betas, eta=eta, cond=cond
    #         )
    #     else:
    #         with torch.no_grad():
    #             xs, x0_preds = generalized_steps(
    #                 x, seq, self.model, self.betas, eta=eta, cond=cond
    #             )
    #     return x0_preds[-1]
    def forward(self, x, seq=None, eta=0.0, trainable=True, cond=None):
        if seq is None:
            seq = list(range(0, self.config['diffusion']['num_diffusion_timesteps'], 10))
        
        # Ensure x and cond are on self.device
        x = x.to(self.device)
        if cond is not None:
            cond = cond.to(self.device)
        
        if trainable:
            xs, x0_preds = generalized_steps(
                x, seq, self.model, self.betas, eta=eta, cond=cond
            )
        else:
            with torch.no_grad():
                xs, x0_preds = generalized_steps(
                    x, seq, self.model, self.betas, eta=eta, cond=cond
                )
        return x0_preds[-1]

    def q_sample(self, x0, t):
        return q_sample(x0, t, self.betas)
