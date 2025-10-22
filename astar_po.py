import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
from config import Config

class ValueHead(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.value_network = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() == 3:
            last_hidden = hidden_states[:, -1, :]
        else:
            last_hidden = hidden_states
        return self.value_network(last_hidden)

class AStarPO:
    def __init__(self, model, tokenizer, optimizer, config: Config):
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.config = config
        self.device = model.device
        
        # 值函数头用于优势估计
        hidden_size = model.config.hidden_size
        self.value_head = ValueHead(hidden_size).to(self.device)
        
        # 值函数优化器
        self.value_optimizer = torch.optim.AdamW(
            self.value_head.parameters(), 
            lr=config.learning_rate
        )
    
    def compute_advantage(self, rewards: List[float]) -> torch.Tensor:
        advantages = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        
        if advantages.std() > 0:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages * self.config.advantage_scale
    
    def a_star_po_loss(self, online_log_probs: torch.Tensor, 
                      online_advantages: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        
        reinforce_loss = -torch.mean(online_log_probs * online_advantages)
        
        # 熵正则化
        entropy = -torch.mean(torch.exp(online_log_probs) * online_log_probs)
        entropy_bonus = -self.config.entropy_coeff * entropy
        
        # KL惩罚
        kl_penalty = torch.mean(online_log_probs ** 2) * self.config.beta
        
        # 纯在线损失（移除离线部分）
        total_loss = reinforce_loss + entropy_bonus + kl_penalty
        
        loss_info = {
            'total_loss': total_loss.item(),
            'reinforce_loss': reinforce_loss.item(),
            'entropy': entropy.item(),
            'kl_penalty': kl_penalty.item(),
            'advantage_mean': online_advantages.mean().item()
        }
        
        return total_loss, loss_info
    
    def update_policy(self, online_batch: List[Dict]) -> Optional[Dict]:
        """策略更新 - 纯在线版本"""
        if not online_batch:
            return None
            
        try:
            # 准备在线数据（来自PAG自我纠正）
            online_log_probs = torch.stack([d['log_prob'] for d in online_batch])
            online_rewards = torch.tensor([d['reward'] for d in online_batch], device=self.device)
            
            # 计算优势
            online_advantages = self.compute_advantage(online_rewards.cpu().tolist())
            
            # 计算损失
            loss, loss_info = self.a_star_po_loss(online_log_probs, online_advantages)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip)
            self.optimizer.step()
            
            return loss_info
            
        except Exception as e:
            print(f"A*-PO fail: {e}")
            return None