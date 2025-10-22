import torch
from dataclasses import dataclass

@dataclass
class Config:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # 训练参数
    num_epochs: int = 3
    learning_rate: float = 1e-5
    batch_size: int = 2  
    eval_batch_size: int = 10
    
    # PAG参数
    num_samples: int = 3
    max_correction_rounds: int = 2 
    confidence_threshold: float = 0.8
    
    # A*-PO参数 - 基于论文实现
    alpha: float = 0.7
    beta: float = 0.01
    temperature: float = 0.1
    advantage_scale: float = 1.0
    offline_weight: float = 0.3
    advantage_lambda: float = 0.95
    gamma: float = 0.99
    entropy_coeff: float = 0.01
    value_coeff: float = 0.1
    
    # 技术参数
    max_length: int = 384
    max_train_samples: int = 100  # 限制样本数
    max_test_samples: int = 50
    gradient_clip: float = 1.0
    
    # 生成参数
    gen_temperature: float = 0.7
    gen_top_p: float = 0.9
    gen_max_length: int = 150
    
    def __post_init__(self):
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32