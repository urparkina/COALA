import torch
from torch import nn
from peft.tuners.tuners_utils import check_target_module_exists
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import ProcessPoolExecutor, as_completed
import sys
from coala.COALA import COALA_Layer
import math
from collections import defaultdict
import heapq
import re

def get_numbers(name: str):
    return list(map(int, re.findall(r'\d+', name)))


def compactify(arr, compression_condition=lambda m, n: n > m):
    if not arr:
        return arr

    m = arr[0].shape[0]
    n = sum(t.shape[1] for t in arr)


    if compression_condition(m, n):
        Y = torch.cat(arr, dim=1)
        orig_dtype = Y.dtype

        _, R = torch.linalg.qr(Y.T.to(torch.float32).cuda(), mode='r')
        R = R.cpu()
        R = R.to(orig_dtype)  
        return [R.T]

    return arr

def get_layer(model, name):
    layer = model
    for attr in name.split("."):
        layer = getattr(layer, attr)
    return layer


def set_layer(model, name, layer):
    attrs, name = name.rsplit(".", 1)
    model = get_layer(model, attrs)
    setattr(model, name, layer)


def _get_total_parameters(model):
    return sum(p.numel() for p in model.parameters())


def process_layer(name, module, coala_config, logs=None):
    if not check_target_module_exists(coala_config, name):
        return (name, None)

    if not isinstance(module, nn.Linear):
        return (name, None)

    out_f, in_f = module.weight.shape
    X = None
    if hasattr(coala_config, 'dic'):
        X = torch.cat(coala_config.dic[name], dim=1)
    kwargs = {
        'ratio': coala_config.ratio[name],
        'compress_strategy': coala_config.compress_strategy,
        'fp16': coala_config.fp16,
        'X': X,
        'params': coala_config.params,
    } 

    coala_layer = COALA_Layer(
        module,
        in_features=in_f,
        out_features=out_f,
        **kwargs
    )

        
    return (name, coala_layer)

def calculate_theoretical_error(name, module, coala_config, logs=None, ranks=None):
    if not check_target_module_exists(coala_config, name):
        return (name, None)

    if not isinstance(module, nn.Linear):
        return (name, None)

    if coala_config.compress_strategy == 'empty':
        return (name, 2)
    
    if not coala_config.adaptive_rank:
        return (name, 2)
    
    X = torch.cat(coala_config.dic[name], dim=1)
    T = module.weight @ X.cuda()
    s = torch.linalg.svdvals(T)
    s = s**2
    s /= torch.sum(s)
    
    m, n = module.weight.shape
    r = math.floor(coala_config.ratio * m * n / (m + n))
    r = min(r, m, n)
    return (name, (s, r, m, n))


def heterogeneous_compression(errors, cfg):
    groups = defaultdict(list)
    targets = cfg.target_modules
    if cfg.adaptive_rank == False or cfg.compress_strategy == "empty":
        return {name: cfg.ratio for name in errors}

        

    for name, err in errors.items():
        for t in targets:
            if t in name:
                loss = torch.sum(err[0][err[1]:]**2)
                groups[t].append((name, loss))
                break
                
    
    res   = {}
    ratio = cfg.ratio
    for items in groups.values():
        if not items:
            continue
        inv_logs = [1 / math.log(err) for _, err in items]
        
        s        = sum(inv_logs)
        k        = (1 - ratio) * len(items)
        
        for (name, _), inv in zip(items, inv_logs):
            res[name] = 1 - (k * inv / s)
            
    return res
        


def inject_coala(coala_config, model, logs=None, ranks=None):
    model_adapter = model

    for param_name, param in model_adapter.named_parameters():
        param.requires_grad = False
        
    total_params_before = _get_total_parameters(model)
    
    ##### FINDING ERRORS
    futures = {}
    theoretical_errors = {}
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        for name, module in model_adapter.named_modules():
            future = executor.submit(calculate_theoretical_error, name, module, coala_config, logs, ranks)
            futures[future] = name
        
    for future in as_completed(futures):
            name, err = future.result()
            if err is not None:
                theoretical_errors[name] = err
    
    
    ##### CALCULATE RATIONS
    if ranks != None:
        coala_config.ratio = ranks
    else:
        coala_config.ratio = heterogeneous_compression(theoretical_errors, coala_config)
    
    ##### COMPRESS MATRIX
    futures = {}
    with ThreadPoolExecutor(max_workers=1) as executor:
        for name, module in model_adapter.named_modules():
            future = executor.submit(process_layer, name, module, coala_config, logs)
            futures[future] = name

        for future in as_completed(futures):
            name, coala_layer = future.result()
            if coala_layer is not None:
                set_layer(model_adapter, name, coala_layer)
                print(f"Compress {name:20} layer", file=sys.stderr)

    total_params_after = _get_total_parameters(model)
    if coala_config.compress_strategy != 'emptys':
        print(f'Before: {total_params_before}\nAfter:  {total_params_after}\nCompress Ratio (%): {total_params_after / total_params_before * 100}')
        print(f'Before: {total_params_before}\nAfter:  {total_params_after}\nCompress Ratio (%): {total_params_after / total_params_before * 100}', file=sys.stderr)
    return coala_config.ratio, model_adapter


def prepare_get_samples(model, coala_config):
    coala_config.dic = {}
    hooks = []
    
    for name, module in model.named_modules():
        if check_target_module_exists(coala_config, name):
            hooks.append(module.register_forward_pre_hook(_calculate(name, coala_config.dic, coala_config.params['accumulate'])))

    return model, hooks
            
            
def _calculate(name, dic, fl=0):
        def hook(model, input):
            X = input[0].cpu()
            X = X.permute(2, 1, 0).reshape(X.shape[2], X.shape[0] * X.shape[1])
            prev = dic.get(name)
            if prev != None:
                dic[name].append(X)
            else:
                dic[name] = [X]
            del X
            if fl:
                dic[name] = compactify(dic[name], lambda m, n: n > fl * m)
 

        return hook
    
    
def after_get_samples(model, coala_config, hooks):
    for h in hooks:
        h.remove()
    for name, module in model.named_modules():
        if check_target_module_exists(coala_config, name):
            X = torch.cat(coala_config.dic[name], dim=1)
            print(name, module.weight.shape, X.shape, file=sys.stderr)
            # if name == 'model.layers.4.self_attn.q_proj' or name == 'model.layers.4.self_attn.k_proj' or name == 'model.layers.4.self_attn.v_proj':
            #     torch.save(module.weight.cpu(), name + '.W.pt')
            #     torch.save(X.cpu(), name + '.X.pt')
            coala_config.dic[name] = compactify(coala_config.dic[name], lambda m, n: n > m)

    
    