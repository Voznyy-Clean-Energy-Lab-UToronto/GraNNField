"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

class Hook:
    @property
    def state_dict(self):
        return {}

    @state_dict.setter
    def state_dict(self, state_dict):
        pass

    def on_train_begin(self, trainer):
        pass

    def on_train_ends(self, trainer):
        pass

    def on_train_failed(self, trainer):
        pass

    def on_epoch_begin(self, trainer):
        pass

    def on_batch_begin(self, trainer, train_batch):
        pass

    def on_batch_end(self, trainer, train_batch, result, loss):
        pass

    def on_validation_begin(self, trainer):
        pass

    def on_validation_batch_begin(self, trainer):
        pass

    def on_validation_batch_end(self, trainer, val_batch, val_result):
        pass

    def on_validation_end(self, trainer, val_loss):
        pass

    def on_epoch_end(self, trainer):
        pass
