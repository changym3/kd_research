from torchmetrics.functional import accuracy



class Evaluator():
    def __init__(self, eval_metrics=['acc']) -> None:
        self._eval_metrics = eval_metrics

    def _eval_acc(self, y_pred, y_true):
        return float(accuracy(y_pred, y_true))

    def eval(self, y_pred, y_true):
        eval_res = {}
        if 'acc' in self._eval_metrics:
            eval_res['acc'] = self._eval_acc(y_pred, y_true)
        return eval_res
