import numpy as np

try:
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ModuleNotFoundError:
    def mean_squared_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean((y_true - y_pred) ** 2))

    def mean_absolute_error(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(np.abs(y_true - y_pred)))

    def roc_auc_score(y_true, y_score):
        y_true = np.asarray(y_true, dtype=np.int64)
        y_score = np.asarray(y_score, dtype=np.float64)
        pos = y_true == 1
        neg = y_true == 0
        n_pos = int(np.sum(pos))
        n_neg = int(np.sum(neg))
        if n_pos == 0 or n_neg == 0:
            raise ValueError("roc_auc_score is undefined with one class.")

        order = np.argsort(y_score, kind="mergesort")
        sorted_scores = y_score[order]
        ranks = np.empty_like(sorted_scores, dtype=np.float64)
        start = 0
        while start < len(sorted_scores):
            end = start + 1
            while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
                end += 1
            ranks[start:end] = (start + end + 1) / 2.0
            start = end

        original_ranks = np.empty_like(ranks)
        original_ranks[order] = ranks
        rank_sum_pos = float(np.sum(original_ranks[pos]))
        return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    def accuracy_score(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        return float(np.mean(y_true == y_pred))

    def recall_score(y_true, y_pred, zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fn = float(np.sum((y_true == 1) & (y_pred == 0)))
        denom = tp + fn
        return float(zero_division) if denom == 0 else tp / denom

    def precision_score(y_true, y_pred, zero_division=0):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        tp = float(np.sum((y_true == 1) & (y_pred == 1)))
        fp = float(np.sum((y_true == 0) & (y_pred == 1)))
        denom = tp + fp
        return float(zero_division) if denom == 0 else tp / denom

    def f1_score(y_true, y_pred, zero_division=0):
        precision = precision_score(y_true, y_pred, zero_division=zero_division)
        recall = recall_score(y_true, y_pred, zero_division=zero_division)
        denom = precision + recall
        return float(zero_division) if denom == 0 else 2.0 * precision * recall / denom

    def confusion_matrix(y_true, y_pred, labels=None):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if labels is None:
            labels = np.unique(np.concatenate([y_true, y_pred]))
        matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
        label_to_idx = {label: idx for idx, label in enumerate(labels)}
        for yt, yp in zip(y_true, y_pred):
            if yt in label_to_idx and yp in label_to_idx:
                matrix[label_to_idx[yt], label_to_idx[yp]] += 1
        return matrix


class ClinicalEvaluator:
    def __init__(
        self,
        threshold=65.0,
        duration_steps=30,
        prob_threshold=0.5,
        *,
        original_duration_steps=None,
        downsample_rate=1,
    ):
        self.threshold = float(threshold)
        self.prob_threshold = float(prob_threshold)
        self.original_duration_steps = int(original_duration_steps or duration_steps)
        self.downsample_rate = max(int(downsample_rate), 1)
        self.duration = max(1, int(np.ceil(self.original_duration_steps / self.downsample_rate)))

    def _detect_event_in_single_sequence(self, signal):
        below_threshold = np.asarray(signal) < self.threshold
        count = 0
        for flag in below_threshold:
            count = count + 1 if flag else 0
            if count >= self.duration:
                return 1
        return 0

    def detect_predicted_event_algorithm(self, forecast_samples):
        prob_curve = np.mean(np.asarray(forecast_samples) < self.threshold, axis=0)
        count = 0
        for flag in prob_curve >= self.prob_threshold:
            count = count + 1 if flag else 0
            if count >= self.duration:
                return 1
        return 0

    def calculate_sustained_risk_probability(self, forecast_samples):
        prob_curve = np.mean(np.asarray(forecast_samples) < self.threshold, axis=0)
        if prob_curve.size == 0:
            return 0.0
        if prob_curve.size < self.duration:
            return float(np.min(prob_curve))
        return float(
            max(
                np.min(prob_curve[start : start + self.duration])
                for start in range(prob_curve.size - self.duration + 1)
            )
        )

    def calculate_sample_risk_probability(self, forecast_samples):
        return self.calculate_sustained_risk_probability(forecast_samples)

    def calculate_deterministic_risk_probability(self, signal):
        below = (np.asarray(signal) < self.threshold).astype(np.float32)
        if below.size == 0:
            return 0.0
        if below.size < self.duration:
            return float(np.mean(below))
        return float(
            max(
                np.mean(below[start : start + self.duration])
                for start in range(below.size - self.duration + 1)
            )
        )

    def calculate_metrics(self, predictions_mean, predictions_samples, ground_truths):
        y_pred = np.concatenate(predictions_mean).ravel()
        y_true = np.concatenate(ground_truths).ravel()

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = float(np.sqrt(mse))

        y_labels = []
        y_preds_cls = []
        y_probs = []

        for idx, ground_truth in enumerate(ground_truths):
            gt_label = self._detect_event_in_single_sequence(ground_truth)
            y_labels.append(gt_label)

            if predictions_samples is None:
                pred_label = self._detect_event_in_single_sequence(predictions_mean[idx])
                prob_score = self.calculate_deterministic_risk_probability(predictions_mean[idx])
            else:
                sample_array = np.asarray(predictions_samples[idx])
                if sample_array.ndim >= 2 and sample_array.shape[0] > 1:
                    pred_label = self.detect_predicted_event_algorithm(sample_array)
                    prob_score = self.calculate_sustained_risk_probability(sample_array)
                else:
                    pred_label = self._detect_event_in_single_sequence(predictions_mean[idx])
                    prob_score = self.calculate_deterministic_risk_probability(predictions_mean[idx])

            y_preds_cls.append(pred_label)
            y_probs.append(prob_score)

        y_labels = np.asarray(y_labels)
        y_preds_cls = np.asarray(y_preds_cls)
        y_probs = np.asarray(y_probs)

        auc = roc_auc_score(y_labels, y_probs) if len(np.unique(y_labels)) > 1 else 0.5
        acc = accuracy_score(y_labels, y_preds_cls)
        recall = recall_score(y_labels, y_preds_cls, zero_division=0)
        precision = precision_score(y_labels, y_preds_cls, zero_division=0)
        f1 = f1_score(y_labels, y_preds_cls, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_labels, y_preds_cls, labels=[0, 1]).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        return {
            "MSE": mse,
            "MAE": mae,
            "RMSE": rmse,
            "AUC": auc,
            "F1": f1,
            "Accuracy (%)": acc * 100,
            "Recall (%)": recall * 100,
            "Precision (%)": precision * 100,
            "Specificity (%)": specificity * 100,
        }
