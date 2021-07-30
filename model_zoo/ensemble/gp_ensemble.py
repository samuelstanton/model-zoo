# import torch
#
# from model_zoo.ensemble import BaseEnsemble
# from model_zoo.regression.dkl_svgp import DeepFeatureSVGP
#
#
# class GPEnsemble(BaseEnsemble):
#     """ Ensemble of SVGP + DKL regression models
#     """
#     def __init__(self, input_dim, target_dim, num_components,
#                  num_elites, submodule_params):
#         """
#         Args:
#             input_dim (int)
#             target_dim (int)
#             num_components (int)
#             num_elites (int)
#             submodule_params (dict): kwargs to pass to GP constructor)
#         """
#         super().__init__(input_dim, target_dim, num_components, num_elites)
#         components = [DeepFeatureSVGP(
#             input_dim=input_dim,
#             label_dim=target_dim,
#             **submodule_params
#         ) for _ in range(num_components)]
#         self.components = torch.nn.ModuleList(components)
