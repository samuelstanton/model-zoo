from torch.nn import ModuleList

from model_zoo.ensemble import BaseEnsemble
from model_zoo.regression import FCRegression


class FCEnsemble(BaseEnsemble):
    """ Ensemble of fully-connected neural net regression models
    """
    def __init__(self, input_dim, target_dim, num_components,
                 num_elites, submodule_params):
        """
        Args:
            input_dim (int)
            target_dim (int)
            num_components (int)
            num_elites (int)
            submodule_params (dict): kwargs to pass to FC constructor)
        """
        super().__init__(input_dim, target_dim, num_components, num_elites)
        components = [
            FCRegression(
                input_dim,
                target_dim,
                batch_norm=False,
                **submodule_params
            ) for _ in range(num_components)
        ]
        self.components = ModuleList(components)
