import hypothesis
import hypothesis.nn
import torch

from .base import BaseCriterion
from .base import BaseRatioEstimator



# class LikelihoodToEvidenceCriterion(BaseCriterion):
#
#     DENOMINATOR = "inputs|outputs"
#
#     def __init__(self,
#         estimator,
#         batch_size=hypothesis.default.batch_size,
#         logits=False):
#
#         super(LikelihoodToEvidenceCriterion, self).__init__(
#             batch_size=batch_size,
#             denominator=LikelihoodToEvidenceCriterion.DENOMINATOR,
#             estimator=estimator,
#             logits=logits)

class LikelihoodToEvidenceCriterion(torch.nn.Module):
    DENOMINATOR = "inputs|outputs"

    def __init__(self, estimator, denominator=DENOMINATOR,
                 batch_size=hypothesis.default.batch_size, logits=False):
        super().__init__()
        if logits:
            self.criterion = torch.nn.BCEWtihLogitsLoss()
            self._forward = self._forward_with_logits

        else:
            self.criterion = torch.nn.BCELoss()
            self._forward = self._forward_without_logits

        self.batch_size = batch_size
        self.estimator = estimator
        self.independent_random_variables = self._derive_independent_random_variables(denominator)
        self.ones = torch.ones(self.batch_size, 1)
        self.random_variables = self._derive_random_variables(denominator)
        self.zeros = torch.zeros(self.batch_size, 1)

    def _derive_random_variables(self, denominator):
        random_variables = denominator.replace(hypothesis.default.dependent_delimiter, " ") \
            .replace(hypothesis.default.independent_delimiter, " ") \
            .split(" ")
        random_variables.sort()

        return random_variables

    def _derive_independent_random_variables(self, denominator):
        groups = denominator.split(hypothesis.default.independent_delimiter)
        for index in range(len(groups)):
            groups[index] = groups[index].split(hypothesis.default.dependent_delimiter)

        return groups

    def _forward_without_logits(self, **kwargs):
        y_dependent, _ = self.estimator(**kwargs)

        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)

            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.

        y_independent, _ = self.estimator(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)

        return loss

    def _forward_with_logits(self, **kwargs):
        y_dependent = self.estimator.log_ratio(**kwargs)

        for group in self.independent_random_variables:
            random_indices = torch.randperm(self.batch_size)
            for variable in group:
                kwargs[variable] = kwargs[variable][random_indices] # Make variable independent.

        y_independent = self.estimator.log_ratio(**kwargs)
        loss = self.criterion(y_dependent, self.ones) + self.criterion(y_independent, self.zeros)

        return loss

    def variables(self):
        return self.random_variables

    def independent_variables(self):
        return self.independent_random_variables

    def to(self, device):
        self.criterion = self.criterion.to(device)
        self.ones = self.ones.to(device)
        self.zeros = self.zeros.to(device)

        return self

    def forward(self, **kwargs):
        return self._forward(**kwargs)



class BaseLikelihoodToEvidenceRatioEstimator(BaseRatioEstimator):

    def __init__(self):
        super(BaseLikelihoodToEvidenceRatioEstimator, self).__init__()

    def forward(self, inputs, outputs):
        log_ratios = self.log_ratio(inputs=inputs, outputs=outputs)

        return log_ratios.sigmoid(), log_ratios

    def log_ratio(self, inputs, outputs):
        raise NotImplementedError
