import numpy as np

class SegmentationMetrics:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def update(self, preds, labels):
        """
        preds: numpy array of shape (B, H, W)
        labels: numpy array of shape (B, H, W)
        """
        for pred, label in zip(preds, labels):
            self.conf_matrix += self._confusion_matrix(label, pred)

    def _confusion_matrix(self, label, pred):
        mask = (label >= 0) & (label < self.num_classes)
        combined = self.num_classes * label[mask].astype(int) + pred[mask].astype(int)
        conf_matrix = np.bincount(
            combined,
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return conf_matrix

    def get_scores(self):
        with np.errstate(divide='ignore', invalid='ignore'):
            acc = np.diag(self.conf_matrix).sum() / np.maximum(self.conf_matrix.sum(), 1)
            acc_cls = np.diag(self.conf_matrix) / np.maximum(self.conf_matrix.sum(axis=1), 1)
            acc_cls = np.nanmean(acc_cls)
            iu = np.diag(self.conf_matrix) / (
                np.maximum(
                    self.conf_matrix.sum(axis=1) + self.conf_matrix.sum(axis=0) - np.diag(self.conf_matrix),
                    1
                )
            )
            mean_iu = np.nanmean(iu)
            freq = self.conf_matrix.sum(axis=1) / np.maximum(self.conf_matrix.sum(), 1)
            fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()

        return {
            "Pixel Accuracy": acc,
            "Mean Accuracy": acc_cls,
            "Mean IoU": mean_iu,
            "FWIoU": fwavacc,
            "IoU per Class": iu,
        }

    def print_scores(self):
        scores = self.get_scores()
        for k, v in scores.items():
            if isinstance(v, np.ndarray):
                continue
            print(f"{k}: {v:.4f}")
