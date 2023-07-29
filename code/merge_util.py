import re
import torch

def filter_modules_by_regex(base_module, include_patterns, include_type):
    '''
    find a subset 
    '''
    modules = {}
    for name, module in base_module.named_modules():
        valid_name = not include_patterns or any(
            [re.match(patt, name) for patt in include_patterns]
        )
        valid_type = not include_type or any(
            [isinstance(module, md_cls) for md_cls in include_type]
        )
        if valid_type and valid_name:
            modules[name] = module
    return modules


def get_submodule(m, name):
    ns = name.split(".")
    r = m
    for n in ns:
        r = getattr(r, n)
    return r

# def filter_statedict_by_regex(statedict, include_patterns, ):
#     '''
#     find a subset 
#     '''
#     modules = {}
#     for name, value in statedict.items():
#         valid_name = not include_patterns or any(
#             [re.match(patt, name) for patt in include_patterns]
#         )
#         valid_type = not include_type or any(
#             [isinstance(value, md_cls) for md_cls in include_type]
#         )
#         if valid_type and valid_name:
#             modules[name] = module
#     return modules



def filter_param_regex(param_names, include_regex=None, exclude_regex=None ):
    param_name_to_merge=[]
    for name in param_names:
        valid1 = (exclude_regex is None) or (not any([re.match(patt, name) for patt in exclude_regex]))
        valid2 = (include_regex is None) or (any ([re.match(patt, name) for patt in include_regex])) 
        if valid1 and valid2:
            param_name_to_merge.append(name)
    return param_name_to_merge


# def filter_params_to_merge(param_names, exclude_param_regex):
#     params_to_merge = []
#     for name in param_names:
#         valid = not any([re.match(patt, name) for patt in exclude_param_regex])
#         if valid:
#             params_to_merge.append(name)
#     return params_to_merge



def to_diag(cov_mat):
    mask = torch.diag(torch.ones(cov_mat.size(0))).to(cov_mat.device)
    diag_cov_mat = mask * cov_mat
    return diag_cov_mat

def reduce_non_diag(cov_mat, a):
    diag_weight = torch.diag(torch.ones(cov_mat.size(0)) - a).to(cov_mat.device)
    non_diag_weight = torch.zeros_like(diag_weight).fill_(a)
    weight = diag_weight + non_diag_weight
    ret = cov_mat * weight
    return ret