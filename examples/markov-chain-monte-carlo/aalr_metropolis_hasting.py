# Markov chain Monte Carlo: Amortized Approximate Likelihood Ratios
import hypothesis
import torch
import numpy as np
import matplotlib.pyplot as plt

from hypothesis.simulation import Simulator
from torch.distributions.uniform import Uniform
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceRatioEstimatorMLP as RatioEstimator

# TRAINING
from hypothesis.util.data import SimulatorDataset
from hypothesis.nn.amortized_ratio_estimation import LikelihoodToEvidenceCriterion as Criterion
from hypothesis.visualization.util import make_square

class NormalSimulator(Simulator):

    def __init__(self):
        super(NormalSimulator, self).__init__()

    def forward(self, inputs):
        inputs = inputs.view(-1, 1)
        return torch.randn(inputs.size(0), 1) + inputs


simulator = NormalSimulator()


# PRIOR
prior = Uniform(-30, 30)


# Architecture definitions
activation = torch.nn.ELU
layers = [64, 64, 64]
inputs_shape = (1,)
outputs_shape = (1,)

# Allocation
ratio_estimator = RatioEstimator(
    activation=activation,
    shape_inputs=inputs_shape,
    shape_outputs=outputs_shape,
    layers=layers)

ratio_estimator = ratio_estimator.to(hypothesis.accelerator)
ratio_estimator = ratio_estimator.train()

# TRAINING
batch_size = 256
epochs = 25

dataset = SimulatorDataset(simulator, prior)
# criterion = Criterion(estimator=ratio_estimator, batch_size=batch_size).to(hypothesis.accelerator)
criterion = Criterion(estimator=ratio_estimator, batch_size=batch_size).to('cuda')

optimizer = torch.optim.Adam(ratio_estimator.parameters())

losses = []
for epoch in range(epochs):
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    num_batches = len(data_loader)
    data_loader = iter(data_loader)

    for batch_index in range(num_batches):
        optimizer.zero_grad()
        inputs, outputs = next(data_loader)
        inputs = inputs.to(hypothesis.accelerator)
        outputs = outputs.to(hypothesis.accelerator)
        loss = criterion(inputs=inputs, outputs=outputs)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

losses = np.array(losses)
plt.plot(np.log(losses), lw=2, color="black")
plt.minorticks_on()
plt.xlabel("Gradient updates")
plt.ylabel("Logarithmic loss")
make_square(plt.gca())
plt.show()

ratio_estimator = ratio_estimator.cpu()
ratio_estimator.eval()

## POSTERIOR INFERENCE
