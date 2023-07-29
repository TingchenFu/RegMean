from merge_util import filter_modules_by_regex, filter_params_to_merge,to_diag,reduce_non_diag
import torch
import re

def regmean(statedicts, activations,exclude_param_regex_list=[],coeff=None,regmean_diag=False,regmean_reduce_nondiag=-1.0):
    '''
    Args:
        statedicts:  a list of the state dict of checkpoint to be merged
        activations: a list of dict, with the length of list the number of model 
        params: the linear parameter to be merged
    Return:
        the merged checkpoint
    '''
    avg_params = {}
    n_model = len(activations)
    # param is a dict of list
    params = {}
    for state_dict in statedicts:
        n2p = state_dict
        #{k: v for k, v in local_model.named_parameters()}
        merge_param_names = filter_params_to_merge(
            [n for n in n2p], exclude_param_regex_list
        )
        for n in merge_param_names:
            try:
                params[n].append(n2p[n])
            except:
                params[n] = n2p[n]

    # special treatments for linear weight params
    for name in params.keys():
        valid_for_regmean = not any(
            [
                re.match(patt, name)
                for patt in exclude_param_regex_list
            ]
        )
        module_name = name[:-len(".weight")]
        if name.endswith(".weight") and valid_for_regmean and module_name in activations[0].keys():
            gram_m_ws, grams = [], []
            
            is_conv = "meta_info" in activations[0] and module_name in activations[
                0
            ]["meta_info"].get("conv1d", [])

            for model_id, model_activition in enumerate(activations):
                param_activation = model_activition[module_name]

                if regmean_diag:
                    param_activation = to_diag(param_activation)

                if regmean_reduce_nondiag >= 0:
                    param_activation = reduce_non_diag(
                        param_activation, a=regmean_reduce_nondiag
                    )

                model_param = params[name][model_id]
                if coeff is not None:
                    coeff = (
                        coeff[model_id] * n_model
                    )  # according to formula
                    param_activation = param_activation * coeff

                if is_conv:
                    gram_m_ws.append(torch.matmul(param_activation, model_param))
                else:
                    gram_m_ws.append(
                        torch.matmul(param_activation, model_param.transpose(0, 1))
                    )

                grams.append(param_activation)
            sum_cov = sum(grams)  # sum over model
            sum_gram_m_ws = sum(gram_m_ws)
            sum_cov_inv = torch.inverse(sum_cov)
            wt = torch.matmul(sum_cov_inv, sum_gram_m_ws)
            if is_conv:
                w = wt
            else:
                w = wt.transpose(0, 1)
            avg_params[name] = w
        else:
            if coeff is None:
                avg_params[name] = torch.stack(params[name], 0).mean(0)
            else:
                params = torch.stack(params[name], 0)
                coeff = coeff
                if not torch.is_tensor(coeff):
                    coeff = torch.FloatTensor(coeff)
                coeff = coeff.view(-1, *[1 for _ in range(params.dim() - 1)]).to(
                    params.device
                )
                avg_params[name] = (params * coeff).sum(0)
    return avg_params