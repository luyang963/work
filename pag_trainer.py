import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Optional, Tuple
import re
from tqdm import tqdm
import json

from config import Config
from astar_po import AStarPO
from data_loader import MathDataLoader
from utils import tokenize_text, compute_sequence_log_prob, compare_answers

class PolicyAsVerifier:
    """Policy as Generative Verifier - Core component of PAG paper"""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
    
    def verify_solutions(self, problem: str, candidates: List[str]) -> Tuple[List[float], str]:
        """Generative verifier - evaluate candidate solutions"""
        try:
            verification_prompt = self._build_verification_prompt(problem, candidates)
            inputs = tokenize_text(self.tokenizer, verification_prompt, 1024, self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs['input_ids'],
                    attention_mask=inputs['attention_mask'],
                    max_length=inputs['input_ids'].shape[1] + 50,
                    do_sample=False,
                    temperature=0.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
            
            verification_text = self.tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            
            # Parse verification results
            scores = self._parse_verification_scores(verification_text, len(candidates))
            reasoning = self._extract_reasoning(verification_text)
            
            return scores, reasoning
            
        except Exception as e:
            print(f"Verifier failed: {e}")
            return [0.0] * len(candidates), "Verification failed"
    
    def _build_verification_prompt(self, problem: str, candidates: List[str]) -> str:
        """Build verification prompt"""
        prompt = f"""You are a math solution verifier. Please carefully analyze the following problem and candidate solutions, evaluating the correctness of each solution.

Problem: {problem}

Candidate solutions:
"""
        for i, candidate in enumerate(candidates):
            clean_candidate = candidate.replace(problem, "").strip()
            if len(clean_candidate) > 100:
                clean_candidate = clean_candidate[:100] + "..."
            prompt += f"\nSolution {i+1}:\n{clean_candidate}\n"
        
        prompt += """
Please answer in the following format:
Analysis: [Your reasoning process]
Scores: [Give scores for each solution in order, between 0.0 and 1.0, separated by commas]
Best solution: [Number]

Please answer now:"""
        
        return prompt
    
    def _parse_verification_scores(self, verification_text: str, num_candidates: int) -> List[float]:
        """Parse verification scores"""
        scores = [0.0] * num_candidates
        
        # Find scores section
        score_match = re.search(r'Scores\s*:\s*([0-9.,\s]+)', verification_text)
        if score_match:
            score_text = score_match.group(1)
            numbers = re.findall(r'[0-9.]+', score_text)
            if numbers:
                for i, num in enumerate(numbers[:num_candidates]):
                    try:
                        scores[i] = min(max(float(num), 0.0), 1.0)
                    except:
                        scores[i] = 0.0
        
        # If no scores found, try to find best solution
        if max(scores) == 0.0:
            best_match = re.search(r'Best solution\s*:\s*(\d+)', verification_text)
            if best_match:
                best_idx = int(best_match.group(1)) - 1
                if 0 <= best_idx < num_candidates:
                    scores[best_idx] = 1.0
        
        return scores
    
    def _extract_reasoning(self, verification_text: str) -> str:
        """Extract reasoning process"""
        reasoning_match = re.search(r'Analysis\s*:\s*(.+?)(?=Scores|Best solution|$)', verification_text, re.DOTALL)
        if reasoning_match:
            return reasoning_match.group(1).strip()
        return "No detailed reasoning"

class PagTrainer:
    """Pure PAG trainer - only uses self-correction data, no offline data"""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        print(f"Loading model: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            torch_dtype=config.torch_dtype,
            device_map="auto" if torch.cuda.is_available() else None
        )
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize components - pure online version
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.a_star_po = AStarPO(self.model, self.tokenizer, self.optimizer, config)
        self.verifier = PolicyAsVerifier(self.model, self.tokenizer, self.device)
        
        # Load data
        data_loader = MathDataLoader(config)
        self.train_data, self.test_data = data_loader.load_math_data()
        
        # Training progress tracking
        self.train_progress = {
            'epoch': 0,
            'step': 0,
            'best_accuracy': 0.0,
            'loss_history': [],
            'accuracy_history': []
        }
        
        print("PAG trainer initialization completed! (Pure online version)")
    
    def generate_with_log_probs(self, prompt: str, num_samples: int = 1) -> tuple:
        """Generate answers and compute log probabilities"""
        try:
            inputs = tokenize_text(self.tokenizer, prompt, self.config.max_length, self.device)
            
            all_sequences = []
            all_log_probs = []
            
            for _ in range(num_samples):
                with torch.no_grad():
                    outputs = self.model.generate(
                        inputs['input_ids'],
                        attention_mask=inputs['attention_mask'],
                        max_length=inputs['input_ids'].shape[1] + self.config.gen_max_length,
                        do_sample=True,
                        temperature=self.config.gen_temperature,
                        top_p=self.config.gen_top_p,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        return_dict_in_generate=True,
                        output_scores=True
                    )
                    
                    generated_ids = outputs.sequences
                    log_prob = compute_sequence_log_prob(
                        self.model, self.tokenizer, 
                        inputs['input_ids'], generated_ids, self.device
                    )
                    
                    all_sequences.append(generated_ids)
                    all_log_probs.append(log_prob)
            
            return all_sequences, all_log_probs
            
        except Exception as e:
            print(f"Generation failed: {e}")
            return [], []
    
    def self_correction_cycle(self, problem: str, reference_solution: str) -> List[Dict]:
        """Pure PAG self-correction cycle - no offline data"""
        online_data = []
        
        for round_idx in range(self.config.max_correction_rounds):
            print(f"  Correction round {round_idx + 1}/{self.config.max_correction_rounds}")
            
            # Generate candidate answers
            prompt = f"Problem: {problem}\nSolution:"
            sequences, log_probs = self.generate_with_log_probs(prompt, self.config.num_samples)
            
            if not sequences:
                print("    Failed to generate answers, skipping this round")
                continue
                
            # Decode candidate answers
            candidates = []
            for seq in sequences:
                text = self.tokenizer.decode(seq[0], skip_special_tokens=True)
                if problem in text:
                    text = text.replace(problem, "").replace("Solution:", "").strip()
                text = re.sub(r'Problem:.*?Solution:', '', text).strip()
                candidates.append(text)
            
            if not candidates:
                print("    No valid candidate answers, skipping verification")
                continue
                
            # Use generative verifier for evaluation
            rewards, reasoning = self.verifier.verify_solutions(problem, candidates)
            print(f"    Verifier reasoning: {reasoning[:80]}...")
            print(f"    Reward distribution: {rewards}")
            
            # Collect training data (pure online data)
            for i, (log_prob, reward, candidate) in enumerate(zip(log_probs, rewards, candidates)):
                online_data.append({
                    'log_prob': log_prob,
                    'reward': reward,
                    'text': candidate,
                    'problem': problem,
                    'round': round_idx
                })
                
                print(f"    Candidate {i+1}: {candidate[:50]}... | Reward: {reward:.2f}")
            
            # High confidence stopping condition
            if max(rewards) >= self.config.confidence_threshold:
                print("    ✓ Found high-confidence answer, stopping correction")
                break
                
            # Pure online policy update
            if len(online_data) >= 2:
                recent_batch = online_data[-2:]
                loss_info = self.a_star_po.update_policy(recent_batch)
                if loss_info:
                    self.train_progress['step'] += 1
                    self.train_progress['loss_history'].append(loss_info['total_loss'])
                    print(f"    A*-PO update - loss: {loss_info['total_loss']:.4f}")
        
        return online_data
    
    def train_epoch(self, epoch_num: int) -> float:
        """Train one epoch - pure online version"""
        self.model.train()
        total_loss = 0.0
        update_count = 0
        
        if not self.train_data:
            print("❌ No training data, skipping epoch")
            return 0.0
            
        num_batches = min(len(self.train_data) // self.config.batch_size, 
                         self.config.max_train_samples // self.config.batch_size)
        
        print(f"Epoch {epoch_num}: Training {num_batches} batches")
        
        progress_bar = tqdm(range(num_batches), desc=f"Epoch {epoch_num}")
        
        for batch_idx in progress_bar:
            batch_loss = 0.0
            batch_updates = 0
            
            # Process batch
            start_idx = batch_idx * self.config.batch_size
            end_idx = min(start_idx + self.config.batch_size, len(self.train_data))
            batch_items = self.train_data[start_idx:end_idx]
            
            for item in batch_items:
                # Pure PAG self-correction, generate online data
                online_data = self.self_correction_cycle(item['problem'], item['solution'])
                
                if online_data and len(online_data) >= 2:
                    recent_data = online_data[-2:]
                    loss_info = self.a_star_po.update_policy(recent_data)
                    
                    if loss_info:
                        batch_loss += loss_info['total_loss']
                        batch_updates += 1
            
            if batch_updates > 0:
                avg_batch_loss = batch_loss / batch_updates
                total_loss += avg_batch_loss
                update_count += 1
                
                progress_bar.set_postfix({
                    'loss': f'{avg_batch_loss:.4f}',
                    'updates': update_count
                })
        
        avg_epoch_loss = total_loss / max(update_count, 1)
        self.train_progress['epoch'] = epoch_num
        print(f"Epoch {epoch_num} completed - average loss: {avg_epoch_loss:.4f}")
        return avg_epoch_loss