from collections.abc import Iterable
import math
import os
import sys
import typing
import json
import time
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from numpy.lib.format import open_memmap

from cs336_basics.bpe_tokenizer import Tokenizer, bpe_tokenizer, serialize_merges, serialize_vocab
from cs336_basics.optimizer import AdamW
from cs336_basics.data_loader import get_batch
from cs336_basics.transformer import TransformerLM

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]):
    state = {
        'model_state': model.state_dict(),
        'optim_state': optimizer.state_dict(),
        'iterations': iteration
    }
    torch.save(state, out)

def load_checkpoint(src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model: torch.nn.Module, optimizer: torch.optim.Optimizer) -> int:
    state = torch.load(src)
    model.load_state_dict(state['model_state'])
    optimizer.load_state_dict(state['optim_state'])
    return state['iterations']

def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute a numerically stable cross entropy loss

    Args
      logits: (... batch vocab_size)
    """
    log_probs = nn.functional.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return nll.mean()

def get_lr_cosine_schedule(t: int, max_lr: float, min_lr: float, warm_up_iter: int, cosine_iter: int) -> float:
    if t < warm_up_iter:
        return t * max_lr / warm_up_iter
    if t < cosine_iter:
        return min_lr + 0.5 * (1 + math.cos(math.pi * (t - warm_up_iter) / (cosine_iter - warm_up_iter)))*(max_lr - min_lr)
    return min_lr

def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    # Compute global L2 norm across all gradients
    total_sq = None
    params_list = list(parameters)
    for p in params_list:
        grad = p.grad if isinstance(p, torch.nn.Parameter) else None
        if grad is None:
            continue
        grad_norm = grad.norm(2)
        if total_sq is None:
            total_sq = grad_norm.pow(2)
        else:
            total_sq = total_sq + grad_norm.pow(2)
    if total_sq is None:
        return
    total_norm = total_sq.sqrt()
    clip_coef = max_l2_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in params_list:
            grad = p.grad if isinstance(p, torch.nn.Parameter) else None
            if grad is None:
                continue
            grad.mul_(clip_coef)


def train_model():
    # Transformer parameters
    vocab_size = 10000
    context_length = 256
    d_model = 512
    d_ff = 1344
    theta = 10000
    num_layers = 4
    num_heads = 16

    transformer = TransformerLM(vocab_size, d_model, context_length, num_layers, num_heads, d_ff, theta)

    # Optimizer parameters
    learning_rate = 1e-3
    betas = (0.9, 0.999)
    eps = 1e-8
    optimizer = AdamW(transformer.parameters())

def _tokenize_file(input_path: str, tokenizer: Tokenizer) -> str:
    data_dir = get_data_dir_path()
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(data_dir, f"{base_name}.tokens.npy")
    if os.path.exists(output_path):
        return output_path

    # Two-pass streaming: count, then write via memmap to keep RAM low
    _log('Trainer', f"Counting tokens for: {input_path}")
    total_tokens = 0
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for _ in tokenizer.encode_iterable(f):
            total_tokens += 1
    _log('Trainer', f"Total tokens: {total_tokens}")

    _log('Trainer', f"Writing tokens to {output_path} using memmap")
    mm = open_memmap(output_path, mode='w+', dtype=np.uint16, shape=(total_tokens,))
    idx = 0
    report_every = max(1_000_000, total_tokens // 20)
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        for tid in tokenizer.encode_iterable(f):
            mm[idx] = np.uint16(tid)
            idx += 1
            if idx % report_every == 0:
                _log('Trainer', f"Tokenized {idx}/{total_tokens} tokens ({idx / max(total_tokens,1):.0%})")
    del mm  # ensure flush
    _log('Trainer', f"Finished tokenization to {output_path}")
    return output_path


def tokenize_training_data(config: dict, tokenizer: Tokenizer):
    data_dir = get_data_dir_path()

    input_path = config['data_file']
    # Name output based on input filename for clarity
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(data_dir, f"{base_name}.tokens.npy")
    
    # If tokenized data is already present, skip re-tokenization
    if os.path.exists(output_path):
        return output_path
    
    # Simple approach: read all text into memory and encode at once
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()

    token_ids = tokenizer.encode(text)
    tokens_np = np.asarray(token_ids, dtype=np.uint16)
    np.save(output_path, tokens_np)
    return output_path


def get_vocab(config: dict) -> tuple[str, str]:
    # Check for precomputed BPE files in the repository's data directory
    data_dir = get_data_dir_path()
    bpe_vocab_path = os.path.join(data_dir, 'bpe_vocab.json')
    bpe_merges_path = os.path.join(data_dir, 'bpe_merges.txt')

    has_bpe_vocab = os.path.exists(bpe_vocab_path)
    has_bpe_merges = os.path.exists(bpe_merges_path)

    if not has_bpe_merges or not has_bpe_vocab or config['trainer_config']['recreate_vocab']:
        # Read input path strictly from trainer_config
        tcfg = config['trainer_config']
        input_path = tcfg['data_file']
        vocab_size = config['tokenizer_config']['vocab_size']
        special_tokens = config['tokenizer_config']['special_tokens']
        vocab, merges = bpe_tokenizer(input_path, vocab_size, special_tokens)
        serialize_merges(merges, bpe_merges_path)
        serialize_vocab(vocab, bpe_vocab_path)
    return bpe_vocab_path, bpe_merges_path

def get_data_dir_path() -> str:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(repo_root, 'data')
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)
    return data_path

def _select_device() -> str:
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _log(component: str, message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] [{component}] {message}")


@torch.no_grad()
def _evaluate(model: torch.nn.Module, val_tokens: np.ndarray, batch_size: int, context_length: int, device: str, eval_iters: int) -> float:
    model.eval()
    running = 0.0
    for _ in range(eval_iters):
        x, y = get_batch(val_tokens, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        running += float(loss.item())
    model.train()
    return running / eval_iters


def _maybe_init_wandb(cfg: dict) -> typing.Any:
    wandb_cfg = cfg.get('wandb', {}) or {}
    enabled = bool(wandb_cfg.get('enabled', False))
    if not enabled:
        return None
    try:
        import wandb
        wandb.init(
            project=wandb_cfg.get('project', 'cs336-a1'),
            name=wandb_cfg.get('run_name'),
            config=cfg,
        )
        _log('Trainer', 'Initialized Weights & Biases logging')
        return wandb
    except Exception:
        _log('Trainer', 'Failed to initialize Weights & Biases logging; proceeding without it')
        return None


def _find_latest_checkpoint(output_dir: str) -> str | None:
    try:
        if not os.path.isdir(output_dir):
            return None
        candidates = []
        for name in os.listdir(output_dir):
            if not name.startswith("ckpt_step_") or not name.endswith(".pt"):
                continue
            middle = name[len("ckpt_step_"):-len(".pt")]
            try:
                step = int(middle)
                candidates.append((step, os.path.join(output_dir, name)))
            except ValueError:
                continue
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]
    except Exception:
        return None


def _get_eos_token_id(tokenizer: Tokenizer) -> int | None:
    try:
        eos = None
        for t in tokenizer.special_tokens:
            if t.strip() == "<|endoftext|>":
                eos = t
                break
        if eos is None and tokenizer.special_tokens:
            eos = tokenizer.special_tokens[0]
        if eos is None:
            return None
        token_bytes = eos.encode("utf-8")
        return tokenizer.vocab_reverse.get(token_bytes)
    except Exception:
        return None


def eval_interactive(config: dict) -> None:
    device = _select_device()
    _log('Trainer', f"Using device: {device}")

    # Tokenizer setup
    _log('Trainer', 'Preparing tokenizer vocab and merges')
    vocab_path, merges_path = get_vocab(config)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, config['tokenizer_config']['special_tokens'])

    # Model config
    mcfg = config.get('model_config', {})
    vocab_size = int(mcfg.get('vocab_size', 10000))
    context_length = int(mcfg.get('context_length', 256))
    d_model = int(mcfg.get('d_model', 512))
    d_ff = int(mcfg.get('d_ff', 1344))
    theta = float(mcfg.get('theta', 10000))
    num_layers = int(mcfg.get('num_layers', 4))
    num_heads = int(mcfg.get('num_heads', 16))

    _log('Trainer', f"Initializing model for eval: layers={num_layers}, heads={num_heads}, d_model={d_model}, d_ff={d_ff}")
    model = TransformerLM(vocab_size, d_model, context_length, num_layers, num_heads, d_ff, theta)
    model.to(device)

    # Load checkpoint: prefer explicit eval.checkpoint_path, else training.resume_path, else latest in resolved training.output_dir
    ecfg = config.get('eval', {}) or {}
    tcfg = config.get('training', {}) or {}
    data_dir = get_data_dir_path()
    raw_out = tcfg.get('output_dir')
    default_out = os.path.join(data_dir, 'checkpoints')
    resolved_out = os.path.join(data_dir, str(raw_out).lstrip(os.sep)) if raw_out else default_out
    ckpt_path = ecfg.get('checkpoint_path') or tcfg.get('resume_path') or _find_latest_checkpoint(resolved_out)
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            state = torch.load(ckpt_path, map_location='cpu')
            model.load_state_dict(state['model_state'])
            _log('Trainer', f"Loaded checkpoint for eval: {ckpt_path}")
        except Exception as e:
            _log('Trainer', f"Failed to load checkpoint '{ckpt_path}': {e}")
    else:
        _log('Trainer', 'No checkpoint found; evaluating with randomly initialized weights')

    # Eval sampling parameters
    temperature = float(ecfg.get('temperature', 0.7))
    top_p = ecfg.get('top_p')
    top_p = float(top_p) if top_p is not None else None
    max_new_tokens = int(ecfg.get('max_new_tokens', 128))
    eos_token_id = _get_eos_token_id(tokenizer)

    _log('Trainer', 'Entering interactive eval mode. Type \'/exit\' or \'/quit\' to leave.')
    try:
        while True:
            try:
                user = input('> ').strip()
            except EOFError:
                break
            if user == '':
                continue
            if user.lower() in {"/exit", "/quit", "exit", "quit"}:
                break

            prefix_ids = tokenizer.encode(user)
            if not prefix_ids:
                print("")
                continue

            prefix_tensor = torch.tensor(prefix_ids, dtype=torch.long, device=device).unsqueeze(0)
            with torch.no_grad():
                out_ids = model.generate(
                    prefix_tensor,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    eos_token_id=eos_token_id,
                )
            # Decode only the generated continuation
            gen_only = out_ids[0].tolist()[len(prefix_ids):]
            text = tokenizer.decode(gen_only)
            print(text)
    except KeyboardInterrupt:
        pass
    _log('Trainer', 'Exiting interactive eval mode')


def train(config: dict):
    device = _select_device()
    _log('Trainer', f"Using device: {device}")

    # Tokenizer and tokenized datasets
    _log('Trainer', 'Preparing tokenizer vocab and merges')
    vocab_path, merges_path = get_vocab(config)
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, config['tokenizer_config']['special_tokens'])

    # Prefer explicit train_file; else use trainer_config.data_file
    tcfg_all = config['trainer_config']
    train_text_path = config.get('train_file', tcfg_all['data_file'])
    valid_text_path = config.get('valid_file')
    _log('Trainer', f"Tokenizing training corpus: {train_text_path}")
    train_tokens_path = _tokenize_file(train_text_path, tokenizer)
    val_tokens_path = None
    if valid_text_path:
        _log('Trainer', f"Tokenizing validation corpus: {valid_text_path}")
        val_tokens_path = _tokenize_file(valid_text_path, tokenizer)

    # Memory-map token arrays
    _log('Trainer', f"Memory-mapping token files")
    train_tokens = np.load(train_tokens_path, mmap_mode='r')
    val_tokens = np.load(val_tokens_path, mmap_mode='r') if val_tokens_path else None

    # Model config
    mcfg = config.get('model_config', {})
    vocab_size = int(mcfg.get('vocab_size', 10000))
    context_length = int(mcfg.get('context_length', 256))
    d_model = int(mcfg.get('d_model', 512))
    d_ff = int(mcfg.get('d_ff', 1344))
    theta = float(mcfg.get('theta', 10000))
    num_layers = int(mcfg.get('num_layers', 4))
    num_heads = int(mcfg.get('num_heads', 16))

    _log('Trainer', f"Initializing model: layers={num_layers}, heads={num_heads}, d_model={d_model}, d_ff={d_ff}")
    model = TransformerLM(vocab_size, d_model, context_length, num_layers, num_heads, d_ff, theta)
    model.to(device)

    # Optimizer config
    ocfg = config.get('optim_config', {})
    learning_rate = float(ocfg.get('learning_rate', 3e-4))
    betas = tuple(ocfg.get('betas', (0.9, 0.999)))
    eps = float(ocfg.get('eps', 1e-8))
    weight_decay = float(ocfg.get('weight_decay', 0.0))
    optimizer = AdamW(model.parameters(), lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay)
    _log('Trainer', f"Initialized optimizer AdamW: lr={learning_rate:.3e}, betas={betas}, eps={eps}, wd={weight_decay}")

    # LR schedule
    scfg = config.get('lr_schedule', {})
    use_schedule = scfg.get('type', 'constant') == 'cosine'
    warmup_iters = int(scfg.get('warmup_iters', 0))
    cosine_iters = int(scfg.get('cosine_iters', 0))
    min_lr = float(scfg.get('min_lr', learning_rate))
    max_lr = float(scfg.get('max_lr', learning_rate))
    if use_schedule:
        _log('Trainer', f"Using cosine LR schedule: warmup={warmup_iters}, cosine_iters={cosine_iters}, min_lr={min_lr:.3e}, max_lr={max_lr:.3e}")
    else:
        _log('Trainer', f"Using constant LR: {learning_rate:.3e}")

    # Training config
    tcfg = config.get('training', {})
    batch_size = int(tcfg.get('batch_size', 32))
    max_steps = tcfg.get('max_steps')
    total_tokens_target = tcfg.get('total_tokens')
    if max_steps is None and total_tokens_target is not None:
        denom = batch_size * context_length
        max_steps = max(1, int(total_tokens_target // denom))
    if max_steps is None:
        max_steps = 1000
    log_interval = int(tcfg.get('log_interval', 50))
    eval_interval = int(tcfg.get('eval_interval', 200))
    eval_iters = int(tcfg.get('eval_iters', 50))
    ckpt_interval = int(tcfg.get('checkpoint_interval', 500))
    grad_clip = float(tcfg.get('grad_clip', 1.0))
    data_dir = get_data_dir_path()
    raw_output_dir = tcfg.get('output_dir')
    if raw_output_dir:
        output_dir = os.path.join(data_dir, str(raw_output_dir).lstrip(os.sep))
    else:
        output_dir = os.path.join(data_dir, 'checkpoints')
    os.makedirs(output_dir, exist_ok=True)
    _log('Trainer', f"Output directory: {output_dir}")

    # Resume
    start_step = 0
    resume_path = tcfg.get('resume_path')
    if resume_path and os.path.exists(resume_path):
        try:
            start_step = load_checkpoint(resume_path, model, optimizer)
            _log('Trainer', f"Resumed from checkpoint: {resume_path} at step {start_step}")
        except Exception as e:
            _log('Trainer', f"Failed to resume from checkpoint: {e}")

    wandb = _maybe_init_wandb(config)

    model.train()
    start_time = time.time()
    last_log_time = start_time
    running_loss = 0.0

    _log('Trainer', f"Starting training: batch_size={batch_size}, context_length={context_length}, max_steps={max_steps}")

    for step in range(start_step, max_steps):
        if use_schedule:
            lr = get_lr_cosine_schedule(step, max_lr, min_lr, warmup_iters, cosine_iters)
            for g in optimizer.param_groups:
                g['lr'] = lr
        else:
            lr = learning_rate

        inputs, targets = get_batch(train_tokens, batch_size, context_length, device)
        logits = model(inputs)
        loss = cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            gradient_clipping(model.parameters(), grad_clip)
        optimizer.step()

        running_loss += float(loss.item())

        if (step + 1) % log_interval == 0 or step == start_step:
            now = time.time()
            iter_time = now - last_log_time
            last_log_time = now
            avg_loss = running_loss / log_interval
            tokens_per_iter = batch_size * context_length
            toks_per_s = tokens_per_iter * log_interval / max(iter_time, 1e-9)
            msg = f"step {step+1}/{max_steps} | loss {avg_loss:.4f} | lr {lr:.3e} | {toks_per_s:.0f} tok/s"
            _log('Trainer', msg)
            if wandb is not None:
                wandb.log({
                    'train/loss': avg_loss,
                    'train/lr': lr,
                    'speed/tokens_per_s': toks_per_s,
                    'progress/step': step + 1,
                    'progress/wall_time_s': now - start_time,
                }, step=step+1)
            running_loss = 0.0

        if val_tokens is not None and ((step + 1) % eval_interval == 0):
            val_loss = _evaluate(model, val_tokens, batch_size, context_length, device, eval_iters)
            _log('Trainer', f"eval step {step+1}: val_loss {val_loss:.4f}")
            if wandb is not None:
                wandb.log({'eval/loss': val_loss}, step=step+1)

        if (step + 1) % ckpt_interval == 0 or (step + 1) == max_steps:
            ckpt_path = os.path.join(output_dir, f"ckpt_step_{step+1}.pt")
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            _log('Trainer', f"Saved checkpoint to {ckpt_path}")



if __name__ == '__main__':
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        # Default to repo config if not provided
        config_file = os.path.join(os.path.dirname(__file__), 'run_config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)
    # Mode resides in trainer_config.mode
    _tcfg = config['trainer_config']
    mode = (_tcfg.get('mode') or 'train').lower()
    if mode == 'eval':
        eval_interactive(config)
    else:
        train(config)





