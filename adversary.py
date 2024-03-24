import torch


class AdversaryBase:
    def __init__(self, adv_lr, adv_max_norm, adv_norm_type="fro"):
        self.adv_lr = adv_lr
        self.adv_max_norm = adv_max_norm
        self.adv_norm_type = adv_norm_type
        self.delta = None

    def init(self, input_temp, input_mask=None):
        self.delta = torch.zeros_like(input_temp)
        if input_mask is not None:
            self.delta = self.delta * input_mask

    def update(self):
        raise NotImplementedError

    def requires_grad(self):
        if self.delta is not None:
            self.delta.requires_grad_(True)


class AdversaryForEmbedding(AdversaryBase):
    def __init__(self, adv_lr, adv_max_norm, adv_norm_type="fro"):
        super().__init__(adv_lr, adv_max_norm, adv_norm_type)

    def update(self):
        delta_grad = self.delta.grad.clone().detach()

        if self.adv_norm_type == "inf":
            grad_norm = torch.norm(delta_grad.view(delta_grad.shape[0], -1), dim=-1, p=float("inf"))
            grad_norm = torch.clamp(grad_norm, min=1e-8).view(-1, 1, 1)
            self.delta = (self.delta + self.adv_lr * delta_grad / grad_norm).detach()
            if self.adv_max_norm > 0:
                self.delta = torch.clamp(self.delta, -self.adv_max_norm, self.adv_max_norm).detach()
        elif self.adv_norm_type == "fro":
            grad_norm = torch.norm(delta_grad.view(delta_grad.shape[0], -1), dim=-1, p="fro")
            grad_norm = torch.clamp(grad_norm, min=1e-8).view(-1, 1, 1)
            self.delta = (self.delta + self.adv_lr * delta_grad / grad_norm).detach()
            if self.adv_max_norm > 0:
                delta_norm = torch.norm(self.delta.view(self.delta.shape[0], -1), dim=-1, p="fro").detach()
                clip_mask = (delta_norm > self.adv_max_norm).to(self.delta)
                clip_weights = (self.adv_max_norm / delta_norm * clip_mask + (1 - clip_mask))
                self.delta = (self.delta * clip_weights.view(-1, 1, 1)).detach()
