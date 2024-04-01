"""
Salatan Duangdangchote
Clean Energy Lab, University of Toronto Scarborough

salatandua/grannfield: https://github.com/salatandua/grannfield

---

This code has been modified from the original version at
atomistic-machine-learning/schnetpack: https://github.com/atomistic-machine-learning/schnetpack
"""

import numpy as np
import torch


class Metric:
    def __init__(self, target, model_output=None, name=None):
        self.target = target
        self.model_output = target if model_output is None else model_output

        if name is None:
            self.name = self.__class__.__name__
        else:
            self.name = name

    def add_batch(self, batch, result):
        """Add a batch to calculate the metric on"""
        raise NotImplementedError

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        raise NotImplementedError

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        pass


class ModelBias(Metric):
    def __init__(self,
                 target,
                 model_output=None,
                 name=None,):
        name = "Bias_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(ModelBias, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.l2loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l2loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        return y - yp

    def add_batch(self, batch, result):
        y = batch[self.target]
        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1)).detach().cpu().data.numpy()
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class MeanSquaredError(Metric):
    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
    ):
        name = "MSE_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(MeanSquaredError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.bias_correction = bias_correction

        self.l2loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l2loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]

        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
                    # print(result.shape)
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.l2loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.l2loss / self.n_entries


class RootMeanSquaredError(MeanSquaredError):
    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
    ):
        name = "RMSE_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(RootMeanSquaredError, self).__init__(
            target=target,
            model_output=model_output,
            bias_correction=bias_correction,
            name=name,
        )

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return np.sqrt(self.l2loss / self.n_entries)


class MeanAbsoluteError(Metric):
    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
    ):
        self.name = "MAE_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=self.name,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]

        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
                    # print(result.shape)
            else:
                result = result[self.model_output]
            yp = result
        diff = self._get_diff(y, yp)
        self.l1loss += torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries


class XAxisMeanAbsoluteError(MeanAbsoluteError):
    def __init__(
        self,
        target,
        model_output,
        bias_correction=None,
        name=None,
    ):
        name = "X_Axis_MAE_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target][:, 0]
        yp = result[self.model_output][:, 0]

        diff = self._get_diff(y, yp)

        self.l1loss += (
            torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        )
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries


class YAxisMeanAbsoluteError(MeanAbsoluteError):
    def __init__(
            self,
            target,
            model_output,
            bias_correction=None,
            name=None,
    ):
        name = "Y_Axis_MAE_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target][:, 1]
        yp = result[self.model_output][:, 1]

        diff = self._get_diff(y, yp)

        self.l1loss += (
            torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        )
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries


class ZAxisMeanAbsoluteError(MeanAbsoluteError):
    def __init__(
            self,
            target,
            model_output,
            bias_correction=None,
            name=None,
    ):
        name = "Z_Axis_MAE_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(MeanAbsoluteError, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.bias_correction = bias_correction

        self.l1loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.l1loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = y - yp
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target][:, 2]
        yp = result[self.model_output][:, 2]

        diff = self._get_diff(y, yp)

        self.l1loss += (
            torch.sum(torch.abs(diff).view(-1), 0).detach().cpu().data.numpy()
        )
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        """Aggregate metric over all previously added batches."""
        return self.l1loss / self.n_entries


class CosineSimilarity(Metric):
    def __init__(
        self,
        target,
        model_output=None,
        bias_correction=None,
        name=None,
    ):
        name = "CosineSimilarity_" + target if name is None else name
        model_output = target if model_output is None else model_output
        super(CosineSimilarity, self).__init__(
            target=target,
            model_output=model_output,
            name=name,
        )

        self.bias_correction = bias_correction

        self.loss = 0.0
        self.n_entries = 0.0

    def reset(self):
        """Reset metric attributes after aggregation to collect new batches."""
        self.loss = 0.0
        self.n_entries = 0.0

    def _get_diff(self, y, yp):
        diff = torch.cosine_similarity(y, yp, dim=0)
        if self.bias_correction is not None:
            diff += self.bias_correction
        return diff

    def add_batch(self, batch, result):
        y = batch[self.target]

        if self.model_output is None:
            yp = result
        else:
            if type(self.model_output) is list:
                for idx in self.model_output:
                    result = result[idx]
                    # print(result.shape)
            else:
                result = result[self.model_output]
            yp = result

        diff = self._get_diff(y, yp)
        self.loss += torch.sum(diff.view(-1) ** 2).detach().cpu().data.numpy()
        self.n_entries += np.prod(y.shape)

    def aggregate(self):
        return self.loss / self.n_entries