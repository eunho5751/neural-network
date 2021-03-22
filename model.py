from collections import OrderedDict
import numpy as np

class Model:
    def __init__(self, loss, optimizer, metrics):
        self.layers = OrderedDict()
        self.optimizable_layers = []
        self.layer_index = {}
        self.loss = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def add(self, layer):
        if layer.is_optimizable():
            self.optimizable_layers.insert(0, layer)

        name = type(layer).__name__
        if name not in self.layer_index:
            self.layer_index[name] = 0
        else:
            self.layer_index[name] += 1
        self.layers[name + str(self.layer_index[name])] = layer

    def train(self, x, y, batch_size, epoch_size):
        steps_per_epoch = int(np.ceil(x.shape[0] / batch_size))
        for i in range(epoch_size):
            x_train = x.copy()
            y_train = y.copy()

            total_loss = 0
            total_metrics = 0
            for j in range(steps_per_epoch):
                batches_per_step = np.min([x_train.shape[0], batch_size])
                batch_mask = np.random.choice(x_train.shape[0], batches_per_step, replace=False)
                x_batch = x_train[batch_mask]
                y_batch = y_train[batch_mask]
                x_train = np.delete(x_train, batch_mask, axis=0)
                y_train = np.delete(y_train, batch_mask, axis=0)

                # train
                y_pred = self.__forward(x_batch)
                loss = self.loss.forward(y_pred, y_batch)
                self.__backward()

                # evaluate
                metrics = self.metrics.result(y_batch, y_pred)
                total_loss += loss
                total_metrics += metrics

            # evaluate
            avg_loss = total_loss / steps_per_epoch
            avg_metrics = total_metrics / steps_per_epoch

            print(f'Epoch {i+1}/{epoch_size}')
            self.__print_eval(avg_loss, avg_metrics)

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def evaluate(self, x, y, batch_size):
        x_eval = x.copy()
        y_eval = y.copy()

        total_loss = 0
        total_metrics = 0
        steps = int(np.ceil(x.shape[0] / batch_size))
        for i in range(steps):
            batches_per_step = np.min([x_eval.shape[0], batch_size])
            batch_mask = np.random.choice(x_eval.shape[0], batches_per_step, replace=False)
            x_batch = x_eval[batch_mask]
            y_batch = y_eval[batch_mask]
            x_eval = np.delete(x_eval, batch_mask, axis=0)
            y_eval = np.delete(y_eval, batch_mask, axis=0)

            pred = self.predict(x_batch)
            loss = self.loss.forward(pred, y_batch)
            metrics = self.metrics.result(y_batch, pred)
            total_loss += loss
            total_metrics += metrics

        avg_loss = total_loss / steps
        avg_metrics = total_metrics / steps
        self.__print_eval(avg_loss, avg_metrics)
        return avg_loss, avg_metrics

    def __forward(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def __backward(self):
        layers = list(self.layers.values())
        layers.reverse()

        dout = self.loss.backward()
        for layer in layers:
            dout = layer.backward(dout)
        self.optimizer.update(self.optimizable_layers)

    def __print_eval(self, loss, metrics):
        print(f'loss: {loss:0.4f} / metrics: {metrics:0.4f}')

