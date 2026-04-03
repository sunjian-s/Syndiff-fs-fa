import torch
import numpy as np


_MODELS = {} # 这是一个全局字典，用作“注册表”，存储所有模型的名称和类


def register_model(cls=None, *, name=None):
  """
  一个“装饰器” (Decorator)，用于自动注册模型类。
  
  使用方法：
  @register_model(name='my_model')
  class MyModel(nn.Module):
    ...
  """

  def _register(cls):
    # 默认使用类名作为名称
    if name is None:
      local_name = cls.__name__
    else:
      local_name = name
    if local_name in _MODELS:
      raise ValueError(f'错误：名为 {local_name} 的模型已被注册')
    
    # 将类 (cls) 存入全局注册表 _MODELS
    _MODELS[local_name] = cls
    return cls

  if cls is None:
    return _register # 如果 @register_model(name=...) 这样被调用
  else:
    return _register(cls) # 如果 @register_model 这样被调用


def get_model(name):
  """根据名称从全局注册表中获取模型类"""
  return _MODELS[name]


def get_sigmas(config):
  """
  【噪声调度表 - 方案1：SMLD (Score Matching with Langevin Dynamics)】
  
  从配置中获取 SMLD 模型的噪声水平 (sigmas)。
  它会创建一个从 sigma_max 到 sigma_min 的“指数”间隔（在 log 空间中是线性的）。
  """
  # 例如：np.log(100) 到 np.log(0.01)
  sigmas = np.exp(
    np.linspace(np.log(config.model.sigma_max), np.log(config.model.sigma_min), config.model.num_scales))

  return sigmas


def get_ddpm_params(config):
  """
  【噪声调度表 - 方案2：DDPM (Denoising Diffusion Probabilistic Models)】
  
  获取原始 DDPM 论文中使用的所有关键系数 (betas, alphas)。
  """
  num_diffusion_timesteps = 1000 # DDPM 默认的总步数
  # 注意：这里的 beta 计算方式与我们之前见过的不同，它假设总步数是1000
  beta_start = config.model.beta_min / config.model.num_scales
  beta_end = config.model.beta_max / config.model.num_scales
  # 创建一个从 beta_start 到 beta_end 线性增长的 beta 数组
  betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)

  alphas = 1. - betas # alpha_t = 1 - beta_t
  alphas_cumprod = np.cumprod(alphas, axis=0) # alpha_bar_t (累积乘积)
  sqrt_alphas_cumprod = np.sqrt(alphas_cumprod) # sqrt(alpha_bar_t)
  sqrt_1m_alphas_cumprod = np.sqrt(1. - alphas_cumprod) # sqrt(1 - alpha_bar_t)

  # 返回一个包含所有预计算系数的字典
  return {
    'betas': betas,
    'alphas': alphas,
    'alphas_cumprod': alphas_cumprod,
    'sqrt_alphas_cumprod': sqrt_alphas_cumprod,
    'sqrt_1m_alphas_cumprod': sqrt_1m_alphas_cumprod,
    'beta_min': beta_start * (num_diffusion_timesteps - 1),
    'beta_max': beta_end * (num_diffusion_timesteps - 1),
    'num_diffusion_timesteps': num_diffusion_timesteps
  }


def create_model(config):
  """
  【模型创建工厂】
  
  根据配置文件 (config) 创建、初始化并封装模型。
  """
  model_name = config.model.name # 1. 从配置中获取模型名称 (例如 'ncsnpp')
  score_model = get_model(model_name)(config) # 2. 使用 get_model() 从注册表中获取类，并实例化
  score_model = score_model.to(config.device) # 3. 将模型移动到 GPU
  
  # 4. 【关键】使用 DataParallel 将模型封装为“多GPU并行”模式
  #     这是一种比 DDP 简单，但在某些情况下效率较低的多卡方案
  score_model = torch.nn.DataParallel(score_model)
  return score_model


def get_model_fn(model, train=False):
  """
  创建一个“模型函数” (model_fn)，用于统一训练和评估的调用接口。
  这是一个“包装器” (wrapper)。

  Args:
    model: 传入的模型 (已经被 DataParallel 封装)
    train: 是训练模式 (True) 还是评估模式 (False)

  Returns:
    一个函数 (model_fn)
  """

  def model_fn(x, labels):
    """
    这个内部函数才是实际被调用的函数。

    Args:
      x: 输入数据 (例如 图像)
      labels: 条件变量 (例如 时间步 t)
    """
    if not train:
      model.eval() # 如果是评估，设置为 .eval() 模式 (关闭 Dropout 等)
      return model(x, labels)
    else:
      model.train() # 如果是训练，设置为 .train() 模式
      return model(x, labels)

  return model_fn # 返回这个内部函数


def to_flattened_numpy(x):
  """辅助函数：将 PyTorch 张量 (Tensor) 转换为 1D 的 NumPy 数组"""
  # .detach() -> 脱离计算图
  # .cpu() -> 移动到 CPU
  # .numpy() -> 转换为 NumPy
  # .reshape((-1,)) -> 扁平化为 1 维
  return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
  """辅助函数：将 1D 的 NumPy 数组 'x' 转换回指定 'shape' 的 PyTorch 张量"""
  return torch.from_numpy(x.reshape(shape))