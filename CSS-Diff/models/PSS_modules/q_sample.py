
import torch

def q_sample(x0, t, betas):
    """
    正向扩散：从原始图像 x0 加噪生成 xt
    参数:
        x0: [B, C, H, W] 原始图像，范围应为 [-1, 1]
        t: [B] 时间步整数张量
        betas: [T] beta 序列
    返回:
        xt: 加噪后的图像
    """
    alphas = 1.0 - betas
    alphas_cumprod = alphas.cumprod(dim=0)

    # 获取当前时间步的 alpha_t
    a_t = alphas_cumprod[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1]

    noise = torch.randn_like(x0)
    xt = (a_t.sqrt() * x0 + (1.0 - a_t).sqrt() * noise)
    return xt, noise
