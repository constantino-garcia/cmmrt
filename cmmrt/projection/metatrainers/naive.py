import inspect

import torch

from cmmrt.projection.projection_tasks import ProjectionsTasks
from cmmrt.projection.metatrainers.base import MetaTrainer
from cmmrt.projection.models.projector.Projector import Projector
from cmmrt.utils.train.torchutils import get_default_device


def smape(x, y):
    """Symmetric Mean Absolute Percentage Error (SMAPE)"""
    with torch.no_grad():
        l1_norm = torch.abs if x.ndim == 0 else lambda x: torch.linalg.norm(x, ord=1)
        return l1_norm(x - y) / torch.max(
            torch.tensor(1e-10, requires_grad=False),
            (l1_norm(x) + l1_norm(y)) / 2.
        ).item()


class NaiveMetaTrainer(MetaTrainer):
    def __init__(self, model: Projector, optimizer, inner_epochs=1, outer_epochs=100, tolerance=5e-3,
                 device=get_default_device(), verbose=10):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")
        for arg, val in values.items():
            setattr(self, arg, val)

    def metatrain(self, tasks: ProjectionsTasks, **kwargs):
        tasks_dl = torch.utils.data.DataLoader(tasks, batch_size=1, shuffle=True)

        if self.device == 'cuda':
            self.model.cuda()

        if self.outer_epochs < 0:
            check_convergence = True
            max_epochs = 100000
        else:
            check_convergence = False
            max_epochs = self.outer_epochs

        def get_model_params():
            with torch.no_grad():
                return [torch.clone(par.squeeze()) for name, par in self.model.named_parameters() if
                        'mlp' not in name]

        last_params = get_model_params()
        print("Param length:", len(last_params))

        iterator = range(1, max_epochs)
        for epoch in iterator:
            epoch_loss = 0
            beta = min([1, 1e-3 + 1. / max(1, max_epochs // 2) * epoch])
            for x, y, _ in tasks_dl:
                x = x.flatten()
                y = y.flatten()

                if self.device == 'cuda':
                    x = x.cuda()
                    y = y.cuda()
                for _ in range(self.inner_epochs):
                    loss = self.model.train_step(x, y, self.optimizer, beta=beta, **kwargs)
                epoch_loss += loss

            if epoch % self.verbose == 0:
                print('[%d] - Loss: %.3f' % (
                    epoch, epoch_loss.item() / len(tasks_dl)
                ))
            if check_convergence:
                new_params = get_model_params()
                max_smape = torch.max(
                    torch.stack([smape(new_par, last_par) for new_par, last_par in zip(new_params, last_params)])
                )
                if max_smape < self.tolerance:
                    print(f"Model has converged! Stopping training at epoch {epoch}")
                    break
                last_params = get_model_params()

        return self.model, self.optimizer
