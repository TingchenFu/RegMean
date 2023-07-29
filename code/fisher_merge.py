from merge_util import filter_params_to_merge
import torch
def fisher_merge(statedicts,  fisher_weights, coeff, fisher_smooth=1e-10, exclude_param_regex_list=None):   
    '''
    Args:
        statedicts: a list of state dict
        fisher_weights: a list of dict. every element in a list corresponding to a local model.
        coeff: a list of n mdoel.
    '''
    
    # parame is dict of list. the value for every key is a list.
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
    avg_params = {}
    fisher_norms, concat_norm = None, None
    # if self.merger_config.fisher_normalize is not None:
    #     fisher_norms, concat_norm = self.get_fisher_norm_coeff(
    #         all_params, fisher_weights
    #     )

    for param_name,parame_values in params.items():
        param_values = torch.stack(parame_values)  # [N, *]
        fisher = (
            torch.stack([x[param_name] for x in fisher_weights])
            + fisher_smooth
        )  # [N, *]

        if isinstance(coeff, list):
            coeff = torch.tensor(coeff)
        coeff = coeff.view(-1, *[1 for _ in range(parame_values.dim() - 1)]).to(
            parame_values.device
        )

        # if self.merger_config.fisher_normalize:
        #     if self.merger_config.fisher_normalize == "param":
        #         fisher_norm = fisher_norms[n]
        #     elif self.merger_config.fisher_normalize == "model":
        #         fisher_norm = concat_norm
        #     else:
        #         raise NotImplementedError
        #     fisher_norm_coeff = 1.0 / (
        #         fisher_norm + self.merger_config.fisher_smooth
        #     )  # [N]
        #     fisher_norm_coeff = fisher_norm_coeff / fisher_norm_coeff.sum()
        #     fisher_norm_coeff = fisher_norm_coeff.view(
        #         -1, *[1 for _ in range(params.dim() - 1)]
        #     )
        #     coeff = coeff * fisher_norm_coeff

        sum_p = param_values * fisher * coeff
        sum_p = sum_p.sum(0)

        denom = (fisher * coeff).sum(0)

        avg_p = sum_p / denom
        avg_params[n] = avg_p
    return avg_params