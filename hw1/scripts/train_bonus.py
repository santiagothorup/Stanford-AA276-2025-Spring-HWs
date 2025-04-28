"""
train_pendulum_bonus.py
Train a neural CBF-QP controller for the 2-state, 1-input inverted pendulum
defined in inverted_pendulum.py *and* the BONUS branch of part1.py.
"""

from argparse import ArgumentParser
import os, sys
import torch
import torch.multiprocessing
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

# ------------------------------------------------------------------ #
#  library imports                                                   #
# ------------------------------------------------------------------ #
from neural_clbf.controllers import NeuralCBFController
from neural_clbf.datamodules.episodic_datamodule import EpisodicDataModule
from neural_clbf.experiments import ExperimentSuite

# add repo root so we can import part1 and inverted_pendulum
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from part1 import safe_mask, failure_mask          # noqa: E402
from neural_clbf.systems import InvertedPendulum     # noqa: E402

torch.multiprocessing.set_sharing_strategy("file_system")

# ------------------------------------------------------------------ #
#  basic hyper-parameters                                            #
# ------------------------------------------------------------------ #
controller_period = 0.05     # [s]
simulation_dt     = 0.01
batch_size        = 64

# ------------------------------------------------------------------ #
#  create system                                                     #
# ------------------------------------------------------------------ #
nominal_params = dict(m=1.0, L=1.0, b=0.05)
scenarios      = [nominal_params]                 # you can add variations

dynamics_model = InvertedPendulum(
    nominal_params,
    dt=simulation_dt,
    controller_dt=controller_period,
    scenarios=scenarios,
)

# ------------------------------------------------------------------ #
#  DataModule                                                        #
# ------------------------------------------------------------------ #
initial_conditions = [
    (-0.3, 0.3),      # θ  (rad)
    (-1.0, 1.0),      # θ̇ (rad/s)
]

data_module = EpisodicDataModule(
    dynamics_model,
    initial_conditions,
    trajectories_per_episode=0,
    trajectory_length=1,
    fixed_samples=100_000,
    max_points=3_000_000,
    val_split=0.01,
    batch_size=1024,
)

# ------------------------------------------------------------------ #
#  (optional) experiment suite                                       #
# ------------------------------------------------------------------ #
experiment_suite = ExperimentSuite([])

# ------------------------------------------------------------------ #
#  Neural CBF-QP controller                                          #
# ------------------------------------------------------------------ #
cbf_controller = NeuralCBFController(
    dynamics_model,
    scenarios,
    data_module,
    experiment_suite=experiment_suite,
    control_indices=[0],          # single torque input
    cbf_hidden_layers=3,
    cbf_hidden_size=256,
    cbf_lambda=0.3,
    cbf_relaxation_penalty=1e3,
    controller_period=controller_period,
    primal_learning_rate=1e-4,
    scale_parameter=1.0,
    learn_shape_epochs=1,
    use_relu=True,
    disable_gurobi=True,
)

# ------------------------------------------------------------------ #
#  Lightning boilerplate                                             #
# ------------------------------------------------------------------ #
parser = ArgumentParser()
parser = pl.Trainer.add_argparse_args(parser)
args   = parser.parse_args()
args.gpus = 1                         # set to 0 for CPU training

logger  = pl_loggers.TensorBoardLogger("outputs", name="bonus_pendulum")

trainer = pl.Trainer.from_argparse_args(
    args,
    logger=logger,
    reload_dataloaders_every_epoch=True,
    max_epochs=51,
)

# ------------------------------------------------------------------ #
#  Train                                                             #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    trainer.fit(cbf_controller)
