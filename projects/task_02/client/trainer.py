"""
Local training logic for federated learning clients.

Returns (weights_bytes, num_samples, avg_loss) so the caller can include
the loss in the SUBMIT payload for dashboard monitoring.
"""
import io
import logging
from typing import Tuple

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


class LocalTrainer:
    def __init__(
        self,
        client_id: int,
        num_clients: int,
        epochs: int = 1,
        lr: float = 0.01,
        batch_size: int = 64,
    ) -> None:
        self.client_id = client_id
        self.num_clients = num_clients
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size

    def train(
        self,
        arch_bytes: bytes,
        weights_bytes: bytes,
    ) -> Tuple[bytes, int, float]:
        """
        Load model (TorchScript, no class needed), apply global weights,
        run local SGD, return (updated_weights_bytes, num_samples, avg_loss).
        """
        # Load architecture — no import of MNISTMLP needed
        model = torch.jit.load(io.BytesIO(arch_bytes))

        # Apply current global weights
        global_sd = torch.load(io.BytesIO(weights_bytes), weights_only=False)
        model.load_state_dict(global_sd)
        model.train()

        from models.mnist_mlp import get_data_partition
        loader = get_data_partition(
            self.client_id, self.num_clients,
            train=True, batch_size=self.batch_size,
        )

        optimizer = torch.optim.SGD(list(model.parameters()), lr=self.lr, momentum=0.9)
        criterion = nn.NLLLoss()

        total_samples = 0
        total_loss = 0.0

        for epoch in range(self.epochs):
            for images, labels in loader:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                n = labels.size(0)
                total_samples += n
                total_loss += loss.item() * n

        avg_loss = round(total_loss / max(total_samples, 1), 6)
        log.info(
            "client %d: %d epoch(s), %d samples, avg loss %.4f",
            self.client_id, self.epochs, total_samples, avg_loss,
        )

        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getvalue(), total_samples, avg_loss
