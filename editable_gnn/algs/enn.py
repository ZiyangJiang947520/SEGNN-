import ipdb
import os
import sys
import torch
import torch.nn as nn
import higher
import torch.nn.functional as F



def fomaml_callback(all_grads):
    return [g.detach() if g is not None else None for g in all_grads]


class EditableModel(nn.Module):
    def __init__(self, model, config, model_constructor):
        super().__init__()

        self.model = model
        self.config = config
        self.model_constructor = model_constructor

    def forward(self, *inputs, **kwargs):
        return self.model(*inputs, **kwargs)

    def outer_parameters(self):
        return self.parameters()


class ENN(EditableModel):
    def __init__(self, model, config, model_constructor, edit_lrs=None):
        super().__init__(model, config, model_constructor)

        if edit_lrs is None:
            edit_lrs = nn.Parameter(torch.tensor([config.edit_lr]))
        self.edit_lrs = edit_lrs

        self.grad_callback = fomaml_callback if config.first_order else lambda x: x


    def outer_parameters(self):
        return super().outer_parameters()


    def get_state_dict(self):
        return self.state_dict()


    def edit(self, x, adj_t, edit_batch, loss_op):
        label = edit_batch['label']
        opt = torch.optim.SGD([{"params": p, "lr": self.config.edit_lr}
                               for (n, p) in self.model.named_parameters()])
        with torch.enable_grad(), higher.innerloop_ctx(
                self.model,
                opt,
                override={'lr': list(self.edit_lrs)},
                copy_initial_weights=False,
                track_higher_grads=self.training,
                in_place=True
        ) as (fmodel, diffopt):
            fmodel.eval()
            for edit_step in range(self.config.n_edit_steps):
                output = fmodel(x, adj_t)[edit_batch['idx']]
                if len(label.shape) == 2:
                    label = label.squeeze()
                loss = loss_op(output, label)
                diffopt.step(loss, grad_callback=self.grad_callback)

        model_edited = fmodel
        model_edited.train(self.training)

        return ENN(model_edited, self.config, self.model_constructor, edit_lrs=self.edit_lrs), {}
