import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from code.merge_util import filter_modules_by_regex
from tqdm import tqdm

def compute_activation(model, train_dataloader, activation_step,):
    '''
    core function
    compute the inner production in linear module for a single model on a single train_dataloader
    input: the single model to be merged, the dataloader it was trained on and the activation step to approximate the hidden activation
    output: the hidden information for a single model
    '''
    #train_dataloader = self.get_train_dataloader()
    covs = {}
    xn = {}

    def get_grams(name):
        def hook(module, input, output):
            """
            Note: adhere to signature of hook functions
            """
            #(bs, seq_length,hidden)
            x = input[0].detach()  # $[b,t,h]
            x = x.view(-1, x.size(-1))
            xtx = torch.matmul(x.transpose(0, 1), x)  # [h,h]
            if name not in covs:
                covs[name] = xtx / x.size(0)
                xn[name] = x.size(0)
            else:
                covs[name] = (covs[name] * xn[name] + xtx) / (x.size(0) + xn[name])
                xn[name] += x.size(0)
        return hook
    
    linear_modules = filter_modules_by_regex(
        model, None, [nn.Linear, nn.Conv1d, Conv1D]
    )
    print("Linear modules: {}".format(linear_modules))
    handles = []
    for name, module in linear_modules.items():
        handle = module.register_forward_hook(get_grams(name))
        handles.append(handle)

    # mark cov modules as special
    covs["meta_info"] = {
        "conv1d": [
            n
            for n, m in filter_modules_by_regex(
                model, None, [nn.Conv1d, Conv1D]
            ).items()
        ]
    }

    if activation_step <0:
        activation_step = len(train_dataloader)
    for step, inputs in tqdm(
        enumerate(train_dataloader), total=activation_step, desc="Computing gram matrix"
    ):
        if step == activation_step:
            break
        # print(inputs['labels']
        _ = model(**inputs)

    for handle in handles:
        handle.remove()

    return covs



def collect_squared_gradients(self, model):
    n2fisher = {n: p.grad.detach() ** 2 for n, p in model.named_parameters()}
    return n2fisher

def compute_fisher(model, train_dataloader, fisher_step, fisher_variant):
    fisher = {}
    assert fisher_variant in ['hard','soft','empirical']

    if fisher_step < 0:
        fisher_step = len(train_dataloader)

    for step, inputs in tqdm(
        enumerate(train_dataloader), total=fisher_step, desc="Computing fisher"
    ):
        model_output= model(**inputs)
        logit = model_output[1]
        if logit.size(-1) == 1 or fisher_variant == 'empirical':
            # is regression task. can only compute empiricial fisher
            # assume f(x; theta) is gaussian with fixed var. log likelihood proportional to || f(x) - y ||^2
            loss = model_output['loss']
            model.zero_grad()
            loss.backward()
            #b_n2fisher = self.collect_squared_gradients(model)
        elif fisher_variant == "hard":
            log_probs = torch.log_softmax(logit, -1)
            _, target_labels = logit.max(-1)
            nll_loss = F.nll_loss(log_probs, target_labels)
            model.zero_grad()
            nll_loss.backward()
            #b_n2fisher = self.collect_squared_gradients(model)
        else:
            assert fisher_variant == "soft"
            probs = torch.softmax(logit, -1).detach()  # [b,c]
            log_probs = torch.log_softmax(logit, -1)
            num_labels = probs.size(-1)
            nll_losses = []
            for label in range(num_labels):
                target = (
                    torch.full(probs.size()[:-1], label).long().to(probs.device)
                )
                nll_loss = F.nll_loss(log_probs, target, reduction="none")
                nll_losses.append(nll_loss)
            nll_losses = torch.stack(nll_losses, -1)  # [b,c]
            weighted_nll_losses = probs * nll_losses
            mean_nll_loss = weighted_nll_losses.sum(-1).mean()
            model.zero_grad()
            mean_nll_loss.backward()
            #b_n2fisher = self.collect_squared_gradients(model)

        n2fisher = {n: p.grad.detach() ** 2 for n, p in model.named_parameters()}
        for n, f in n2fisher.items():
            try:
                fisher[n]+=f
            except:
                fisher[n]=f
    fisher = {n: f/fisher_step for n,f in fisher }
    return fisher