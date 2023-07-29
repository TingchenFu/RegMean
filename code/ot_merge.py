import torch
import ot
import numpy as np
from collections import OrderedDict
from merge_util import filter_param_regex,get_submodule

class GroundMetric:
    """
    Ground Metric object for Wasserstein computations:

    """

    def __init__(
        self,
        ground_metric,
        ground_metric_normalize,
        ground_metric_eff,
        clip_min,
        clip_max,
        clip_gm,
        reg,
        squared,
        dist_normalize,
        activation_histograms,
        act_num_samples,
    ):
        self.ground_metric_type = ground_metric
        self.ground_metric_normalize = ground_metric_normalize
        self.mem_eff = ground_metric_eff
        self.reg = reg
        self.squared = squared
        self.clip_min = clip_min
        self.clip_max = clip_max
        self.clip_gm = clip_gm
        self.dist_normalize = dist_normalize
        self.activation_histogarms= activation_histograms
        self.act_num_samples = act_num_samples


    def _clip(self, ground_metric_matrix):
        percent_clipped = (
            float(
                (ground_metric_matrix >= self.reg * self.clip_max)
                .long()
                .sum()
                .data
            )
            / ground_metric_matrix.numel()
        ) * 100
        print("percent_clipped is (assumes clip_min = 0) ", percent_clipped)
        # will keep the M' = M/reg in range clip_min and clip_max
        ground_metric_matrix.clamp_(
            min=self.reg * self.clip_min, max=self.reg * self.clip_max
        )
        return ground_metric_matrix

    def _normalize(self, ground_metric_matrix):

        if self.ground_metric_normalize == "log":
            ground_metric_matrix = torch.log1p(ground_metric_matrix)
        elif self.ground_metric_normalize == "max":
            print(
                "Normalizing by max of ground metric and which is ",
                ground_metric_matrix.max(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.max()
        elif self.ground_metric_normalize == "median":
            print(
                "Normalizing by median of ground metric and which is ",
                ground_metric_matrix.median(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.median()
        elif self.ground_metric_normalize == "mean":
            print(
                "Normalizing by mean of ground metric and which is ",
                ground_metric_matrix.mean(),
            )
            ground_metric_matrix = ground_metric_matrix / ground_metric_matrix.mean()
        elif self.ground_metric_normalize == "none":
            return ground_metric_matrix
        else:
            raise NotImplementedError

        return ground_metric_matrix

    def _sanity_check(self, ground_metric_matrix):
        assert not (ground_metric_matrix < 0).any()
        assert not (torch.isnan(ground_metric_matrix).any())

    def _cost_matrix_xy(self, x, y, p=2, squared=True):
        # TODO: Use this to guarantee reproducibility of previous results and then move onto better way
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(1)
        y_lin = y.unsqueeze(0)
        c = torch.sum((torch.abs(x_col - y_lin)) ** p, 2)
        if not squared:
            print("dont leave off the squaring of the ground metric")
            c = c ** (1 / 2)
        # print(c.size())
        if self.dist_normalize:
            assert NotImplementedError
        return c

    def _pairwise_distances(self, x, y=None, squared=True):
        """
        Source: https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
        Input: x is a Nxd matrix
               y is an optional Mxd matirx
        Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
                if y is not given then use 'y=x'.
        i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
        """
        x_norm = (x**2).sum(1).view(-1, 1)
        if y is not None:
            y_t = torch.transpose(y, 0, 1)
            y_norm = (y**2).sum(1).view(1, -1)
        else:
            y_t = torch.transpose(x, 0, 1)
            y_norm = x_norm.view(1, -1)

        dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
        # Ensure diagonal is zero if x=y
        dist = torch.clamp(dist, min=0.0)
        if self.activation_histograms and self.dist_normalize:
            dist = dist / self.act_num_samples
            print("Divide squared distances by the num samples")

        if not squared:
            print("dont leave off the squaring of the ground metric")
            dist = dist ** (1 / 2)

        return dist

    def _get_euclidean(self, coordinates, other_coordinates=None):
        # TODO: Replace by torch.pdist (which is said to be much more memory efficient)

        if other_coordinates is None:
            matrix = torch.norm(
                coordinates.view(coordinates.shape[0], 1, coordinates.shape[1])
                - coordinates,
                p=2,
                dim=2,
            )
        else:
            if self.mem_eff:
                matrix = self._pairwise_distances(
                    coordinates, other_coordinates, squared=self.squared
                )
            else:
                matrix = self._cost_matrix_xy(
                    coordinates, other_coordinates, squared=self.squared
                )

        return matrix

    def _normed_vecs(self, vecs, eps=1e-9):
        norms = torch.norm(vecs, dim=-1, keepdim=True)
        # print("stats of vecs are: mean {}, min {}, max {}, std {}".format(
        #    norms.mean(), norms.min(), norms.max(), norms.std()
        # ))
        return vecs / (norms + eps)

    def _get_cosine(self, coordinates, other_coordinates=None):
        if other_coordinates is None:
            matrix = coordinates / torch.norm(coordinates, dim=1, keepdim=True)
            matrix = 1 - matrix @ matrix.t()
        else:
            matrix = 1 - torch.div(
                coordinates @ other_coordinates.t(),
                torch.norm(coordinates, dim=1).view(-1, 1)
                @ torch.norm(other_coordinates, dim=1).view(1, -1),
            )
        return matrix.clamp_(min=0)

    def get_metric(self, coordinates, other_coordinates=None):
        get_metric_map = {
            "euclidean": self._get_euclidean,
            "cosine": self._get_cosine,
        }
        return get_metric_map[self.ground_metric_type](coordinates, other_coordinates)

    def process(self, coordinates, other_coordinates=None):
        coordinates = self._normed_vecs(coordinates)
        if other_coordinates is not None:
            other_coordinates = self._normed_vecs(other_coordinates)

        ground_metric_matrix = self.get_metric(coordinates, other_coordinates)
        self._sanity_check(ground_metric_matrix)
        ground_metric_matrix = self._normalize(ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)
        if self.clip_gm:
            ground_metric_matrix = self._clip(ground_metric_matrix)
        self._sanity_check(ground_metric_matrix)
        return ground_metric_matrix




def ot_merge(statedicts,
        include_regex,
        exact=True,
        reg=0.01,
        correction=True,
        proper_marginals=False,
        past_correction=True):
    '''
    note that here we can still use statedict and directly incidate the model type
    '''   
    # we assume the input and outputs of the ffn layers are aligned
    # local modules is a list of dict. with each dict maps the name to class name
    assert len(statedicts) == 2 # it only supports merge two models.
    src_param=OrderedDict()
    tgt_param=OrderedDict()
    
    
    param_names_to_align = filter_param_regex(list(statedicts[0].keys()),include_regex,None)
    #target_state_dict = statedicts[0]
    for param_name in param_names_to_align:
        src_param[param_name]=statedicts[0][param_name]
        tgt_param[param_name]=statedicts[1][param_name]
    aligned_param = tgt_param



    # for model_id, state_dict in enumerate(statedicts):
    #     name2module = filter_modules_by_regex(
    #         local_model, ot_filter_regex, include_type=None
    #     )
    #     n2p = filter_statedict_by_regex(state_dict,ot_filter_regex)
    #     local_modules.append(name2module)

    # tgt_name2module = filter_modules_by_regex(target_model, ot_filter_regex, include_type=None)

    # for ffn_name in tgt_name2module:
    #     ffn_modules = [x[ffn_name] for x in local_modules]
    #     tgt_ffn_module = tgt_name2module[ffn_name]
    #     ffn_module_match(ffn_modules, tgt_ffn_module, ot_pattern)


    eps = 1e-10
    # layers = [
    #     [get_submodule(x, ot_pattern.ot_lin1) for x in ffn_modules]
    #     + [get_submodule(target_ffn_module, ot_pattern.ot_lin1)],
    #     [get_submodule(x, ot_pattern.ot_lin2) for x in ffn_modules]
    #     + [get_submodule(target_ffn_module, ot_pattern.ot_lin2)],
    # ]
    ground_metric_object = GroundMetric()
    T_var = None

    for layer_id, param_name in enumerate(list(src_param.keys())):
        src_weight = src_param[param_name]
        tgt_weight = tgt_param[param_name]

        # w_a = lina.weight.data
        # w_b = linb.weight.data
        # w_tgt = tgt_lin.weight.data

        mu_card, nu_card = src_weight.shape[0], tgt_weight.shape[0]

        if layer_id == 0:
            M = ground_metric_object.process(src_weight, tgt_weight).to(src_weight.device)
            aligned_wt = src_weight
        else:
            aligned_wt = torch.matmul(src_weight, T_var).to(src_weight.device)
            M = ground_metric_object.process(aligned_wt, tgt_weight)

        mu = np.ones(mu_card) / mu_card
        nu = np.ones(nu_card) / nu_card

        cpuM = M.data.cpu().numpy()
        if exact:
            T = ot.emd(mu, nu, cpuM)
        else:
            T = ot.bregman.sinkhorn(mu, nu, cpuM, reg=reg)

        T_var = torch.from_numpy(T).float().to(src_weight.device)

        # if self.ot_params.debug:
        #     logging.info("The trace of T is {}".format(T_var.trace()))

        if correction:
            if not proper_marginals:
                # think of it as m x 1, scaling weights for m linear combinations of points in X
                marginals = torch.ones(T_var.shape[0]) / T_var.shape[0]
                marginals.to(src_weight.device)
                marginals = torch.diag(1.0 / (marginals + eps)).to(
                    src_weight.device
                )  # take inverse
                T_var = torch.matmul(T_var, marginals)
            else:
                marginals_beta = T_var.t() @ torch.ones(
                    T_var.shape[0], dtype=T_var.dtype
                )

                marginals = 1 / (marginals_beta + eps)
                print("shape of inverse marginals beta is ", marginals_beta.shape)
                print("inverse marginals beta is ", marginals_beta)

                T_var = T_var * marginals
                # i.e., how a neuron of 2nd model is constituted by the neurons of 1st model
                # this should all be ones, and number equal to number of neurons in 2nd model
                print(T_var.sum(dim=0))

        if past_correction:
            matched_w_a = torch.matmul(
                T_var.transpose(0, 1), aligned_wt.reshape(aligned_wt.shape[0], -1)
            )
        else:
            matched_w_a = torch.matmul(
                T_var.transpose(0, 1), src_weight.view(src_weight.shape[0], -1)
            )

        aligned_param.copy_(matched_w_a)

    return aligned_param