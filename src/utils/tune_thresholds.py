from sklearn.metrics import roc_curve, RocCurveDisplay, classification_report, precision_recall_curve, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import TunedThresholdClassifierCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib



class ClassificationThresholdTuner():
    def __init__(self, model_estimator, split:tuple, scoring: list[str] = ["roc_auc", "f1"], cv: float = .2):
        """
        model: model or pipeline in sklearn api syntax that needs optimized. Is untrained, but instantiated with parameters
        split: tuple of X_train, X_test, y_train, y_test in that order. Can be generated from load_data.get_split_from_file
        """
        self.X_train, self.X_test, self.y_train, self.y_test = split

        self.label_encoder = LabelEncoder()

        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)
    

        self.available_scores = []
        self.models = {}
        for score in scoring:

            model = TunedThresholdClassifierCV(model_estimator, scoring=score, cv=cv, store_cv_results=True)

            model.fit(self.X_train, self.y_train)

            self.models[score] = model
            self.available_scores.append(score)

    def __str__(self) -> str:
        out_string = ""

        for key, model in self.models.items():
            out_string += f"Scoring: {key}, Best Threshold: {model.best_threshold_}, Training Score: {model.best_score_}\n"

        return out_string

    def print_classification_reports(self):
        """
        Loops through every tuned model, generates predictions using their 
        optimized thresholds, and prints a comprehensive classification report.
        """
        if not self.models:
            print("No models available to report.")
            return

        # Get the original string names of your target classes (e.g., ['no', 'yes'])
        target_names = self.label_encoder.classes_

        for score_name, tuned_model in self.models.items():
            print("\n" + "="*60)
            print(f" CLASSIFICATION REPORT FOR MODEL OPTIMIZED BY: [{score_name.upper()}]")
            print(f" Optimized Threshold Boundary: {tuned_model.best_threshold_:.4f}")
            print("="*60)

            # 1. Use .predict() to get predictions using the optimized threshold
            numeric_preds = tuned_model.predict(self.X_test)

            # 2. Decode the numeric predictions back to original string labels ('no'/'yes')
            string_preds = self.label_encoder.inverse_transform(numeric_preds)
            string_true = self.label_encoder.inverse_transform(self.y_test)

            # 3. Generate and print the report
            report = classification_report(
                string_true, 
                string_preds, 
                target_names=target_names
            )
            print(report)

    def get_roc_curve(self):

        if "roc_auc" not in self.available_scores:
            raise ValueError("roc_auc not in available_scores")

        tuned_model = self.models["roc_auc"]

        roc_auc = tuned_model.best_score_
        best_thresh = tuned_model.best_threshold_

        y_probs = tuned_model.predict_proba(self.X_test)[:,1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_probs, pos_label=1)

        gmeans = np.sqrt(tpr * (1 - fpr))
        index = np.argmax(gmeans)

        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(
            self.y_test, 
            y_probs, 
            name=f"ROC curve", 
            # color="darkorange", 
            ax=ax
        )
        ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess (AUC = 0.50)")

        # Plot the specific optimized threshold point marker
        ax.scatter(fpr[index], tpr[index], marker='o', color='darkorange', s=100, zorder=5, 
                label=f'Optimal Threshold ({best_thresh:.2f})')

        # Quick styling
        ax.set_title('Receiver Operating Characteristic (ROC) Curve')
        ax.legend(loc="lower right")
        ax.grid(True, linestyle=':', alpha=0.6)
        plt.show()

    def get_precision_recall_curve(self, metric_key: str = "f1"):
        """
        Plots the Precision-Recall curve for a selected tuned model.
        Plots a marker at the model's internally optimized threshold point.
        
        metric_key: The dictionary key of the model to use (e.g., 'f1' or 'roc_auc')
        """
        if metric_key not in self.models:
            raise ValueError(f"'{metric_key}' model not found. Available keys: {self.available_scores}")

        tuned_model = self.models[metric_key]
        best_thresh = tuned_model.best_threshold_

        # 1. Extract positive class probabilities for the test set
        y_probs = tuned_model.predict_proba(self.X_test)[:, 1]

        # 2. Compute precision and recall array elements across all thresholds
        precision, recall, thresholds = precision_recall_curve(self.y_test, y_probs, pos_label=1)

        # 3. Locate the index of your model's internally optimized threshold
        # Find the threshold in the array that is closest to our best_thresh
        ix = np.argmin(np.abs(thresholds - best_thresh))
        if ix >= len(thresholds): ix = len(thresholds) - 1

        # 4. Generate the plot layout
        fig, ax = plt.subplots(figsize=(8, 6))

        # Automatically plot the PR Curve line (calculates Average Precision - AP score)
        PrecisionRecallDisplay.from_predictions(
            self.y_test,
            y_probs,
            color="teal",
            lw=2,
            ax=ax
        )

        # 5. Plot a marker at your model's optimized decision threshold location
        ax.scatter(
            recall[ix], 
            precision[ix], 
            marker='o', 
            color='darkorange', 
            s=200, 
            zorder=5, 
            label=f"Optimized Threshold ({best_thresh:.2f})"
        )

        # 6. Customize chart styles
        ax.set_title(f"Precision-Recall Curve (Optimized by {metric_key.upper()})")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc="lower left")
        
        plt.show()


    def get_confusion_matrix(self, score: list[str] | None = None):

        if score is None:
            score = self.available_scores

        for scoring_metric, model in self.models.items():
            if scoring_metric not in score:
                continue


            # 1. Use .predict() to get predictions using the optimized threshold
            numeric_preds = model.predict(self.X_test)

            # 2. Decode the numeric predictions back to original string labels ('no'/'yes')
            string_preds = self.label_encoder.inverse_transform(numeric_preds)
            string_true = self.label_encoder.inverse_transform(self.y_test)

            # 3. Compute confusion matrixes
            cm = confusion_matrix(string_true, string_preds, labels=['no', 'yes'])

            # 4. Create the side-by-side plot
            fig, ax = plt.subplots(figsize=(14, 5))

            # Plot Confusion Matrix

            optimized_threshold = model.best_threshold_
            title_string = (
            f"Confusion Matrix (Optimized via {scoring_metric.upper()})\n"
            f"Decision Threshold: {optimized_threshold:.4f}"
            )

            disp_gmean = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['no', 'yes'])
            disp_gmean.plot(ax=ax, cmap='Blues', values_format='d')
            ax.set_title(title_string, fontsize= 12, fontweight = "bold", pad = 10)

            plt.tight_layout()
            plt.show()

    def save_model(self, metric_key: str, filepath_prefix: str, final: bool = False, **kwargs):
        """
        Serializes a specific threshold-optimized model alongside its metadata 
        artifacts. If final=True, the model is retrained on all available data 
        (train + test) before being saved.
        
        Parameters:
        - metric_key: The dictionary key of the model optimization to save (e.g., 'f1' or 'roc_auc')
        - filepath_prefix: The base directory or file path prefix where artifacts should be saved
        - final: If True, retrains the underlying threshold classifier on the combined train and test sets
        """
        if metric_key not in self.models:
            raise ValueError(f"'{metric_key}' model not found. Available keys: {self.available_scores}")

        # Extract the model template/instance configured for this specific metric
        tuned_model = self.models[metric_key]

        # Handle final retraining if requested
        if final:
            print(f"\n[INFO] 'final' flag set to True. Combining train and test sets for final retraining...")
            
            # Combine feature matrices and encoded target arrays
            X_all = np.concatenate([self.X_train, self.X_test], axis=0) if hasattr(self.X_train, 'ndim') else self.X_train.append(self.X_test)
            y_all = np.concatenate([self.y_train, self.y_test], axis=0)

            # Re-fit the TunedThresholdClassifierCV model on the complete dataset
            tuned_model.fit(X_all, y_all)
            print(f"[INFO] Retraining complete. New optimal threshold found: {tuned_model.best_threshold_:.4f}")

        # Construct the payload containing the model and updated metadata artifacts
        artifacts = {
            "model": tuned_model,
            "best_threshold": tuned_model.best_threshold_,
            "best_score": tuned_model.best_score_,
            "metric_optimized": metric_key,
            "label_encoder": self.label_encoder,
            "classes": self.label_encoder.classes_.tolist(),
            "is_final_production_model": final
        }


        # Serialize everything into a single joblib file
        save_path = rf"../../src/models/saved_models/{filepath_prefix}.joblib"
        joblib.dump(artifacts, save_path)
        print(f"Successfully serialized pipeline to: {save_path}")

