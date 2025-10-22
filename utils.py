import torch
import torch.nn.functional as F
import re
from typing import List, Dict, Any

def compare_answers(pred: str, true: str) -> bool:
    """比较答案 - 自己实现"""
    try:
        pred_clean = re.sub(r'[^\d.-]', '', pred)
        true_clean = re.sub(r'[^\d.-]', '', true)
        
        if not pred_clean or not true_clean:
            return False
        
        pred_num = float(pred_clean)
        true_num = float(true_clean)
        
        return abs(pred_num - true_num) < 1e-6
    except:
        return False

def compute_sequence_log_prob(model, tokenizer, input_ids: torch.Tensor, 
                            generated_ids: torch.Tensor, device: torch.device) -> torch.Tensor:
    """计算序列对数概率 - 自己实现"""
    try:
        input_len = input_ids.shape[1]
        total_len = generated_ids.shape[1]
        
        if total_len <= input_len:
            return torch.tensor(0.0, device=device)
        
        with torch.no_grad():
            outputs = model(input_ids=generated_ids)
            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)
            
            shifted_ids = generated_ids[:, 1:]
            gathered_log_probs = torch.gather(
                log_probs[:, :-1, :], 2, shifted_ids.unsqueeze(2)
            ).squeeze(2)
            
            if input_len < gathered_log_probs.shape[1]:
                gen_log_probs = gathered_log_probs[:, input_len-1:]
                return gen_log_probs.mean()
            else:
                return torch.tensor(0.0, device=device)
    except Exception as e:
        print(f"序列对数概率计算失败: {e}")
        return torch.tensor(0.0, device=device)

def tokenize_text(tokenizer, text: str, max_length: int, device: torch.device) -> Dict[str, torch.Tensor]:
    """文本tokenization - 自己实现"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    )
    return {k: v.to(device) for k, v in inputs.items()}