
from pathlib import Path
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.base import clone
import naiveautoml
from naiveautoml.evaluators import SplitBasedEvaluator
import matplotlib
try:
    matplotlib.use('Qt5Agg')
except ImportError:
    # No Qt bindings available (e.g. a headless container) - fall back to a
    # non-interactive backend so figures can still be saved to file.
    matplotlib.use('Agg')
import os
os.environ["TABPFN_TOKEN"] = "tabpfn_sk_nLiLECwi51aaI_CVnS6yOYxsiVx9D80RCVCiMO_a7LM"


logging.basicConfig(level=logging.INFO)

from FeatureSelection import FeatureSelection
from experiment_folds import ExperimentFolds
import utils
from classifiers import classifier_config


class _FixedFoldSplitter:
    """
    Minimal splitter wrapping precomputed (train_index, test_index)
    position arrays, so naiveautoml's SplitBasedEvaluator can evaluate
    every candidate pipeline against our fixed leave-one-experiment-out
    folds instead of a random k-fold/mccv split.
    """
    def __init__(self, fold_index_pairs):
        self.fold_index_pairs = fold_index_pairs
        self.n_splits = len(fold_index_pairs)

    def split(self, X, y=None, groups=None):
        for train_idx, test_idx in self.fold_index_pairs:
            yield train_idx, test_idx


class GasClassification:
    """
    Orchestrates training/evaluation on top of self.folds (ExperimentFolds):
    experiment/window bookkeeping and train/val/test/CV-fold construction.
    Resampling/scaling/imputation and the combined
    load_and_process_data_for_classification pipeline live as plain
    functions in utils.py.
    """
    def __init__(self, experiments_file=None):
        self.folds = ExperimentFolds(experiments_file=experiments_file)
        self.classifier_name = None

    def _build_classifier(self, classifier_name):
        automl_results_path = None
        if classifier_name == "AutoML":
            automl_results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path']) / "NaiveAutoML"
        return classifier_config(classifier_name, automl_results_path=automl_results_path)

    def save_best_metrics(self, clf, data, feature_subset=None):
        """
        Score `clf` on val/train/test, write a combined classification
        report + confusion matrices to results_path/<classifier_name>/, and
        dump the fitted classifier there too.
        """
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path']) / self.classifier_name
        results_path.mkdir(parents=True, exist_ok=True)

        X_train = data["train"]["X"][feature_subset] if feature_subset else data["train"]["X"]
        X_val = data["val"]["X"][feature_subset] if feature_subset else data["val"]["X"]
        X_test = data["test"]["X"][feature_subset] if feature_subset else data["test"]["X"]

        y_pred_val = clf.predict(X_val)
        y_pred_test = clf.predict(X_test)

        score_test = clf.score(X_test, data["test"]["y"])
        score_train = clf.score(X_train, data["train"]["y"])
        score_val = clf.score(X_val, data["val"]["y"])

        report = classification_report(data["val"]["y"], y_pred_val, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        test_row = pd.DataFrame(
            {'precision': [score_test], 'recall': [score_test], 'f1-score': [score_test],
             'support': [len(data["test"]["y"])]}, index=['test_accuracy'])
        report_df = pd.concat([report_df, test_row], axis=0)
        train_row = pd.DataFrame(
            {'precision': [score_train], 'recall': [score_train], 'f1-score': [score_train],
             'support': [len(data["train"]["y"])]}, index=['train_accuracy'])
        report_df = pd.concat([report_df, train_row], axis=0)

        cm_val_df = pd.DataFrame(confusion_matrix(data["val"]["y"], y_pred_val))
        cm_test_df = pd.DataFrame(confusion_matrix(data["test"]["y"], y_pred_test))
        cm = pd.concat([cm_val_df, cm_test_df], axis=0, keys=['val', 'test'])

        report_df.to_csv(results_path / f"{self.classifier_name}_best_classification_report.csv", index=True)
        cm.to_csv(results_path / f"{self.classifier_name}_confusion_matrix.csv", index=True)
        dump(clf, results_path / f"{self.classifier_name}_best_classifier.joblib")
        if feature_subset:
            pd.Series(feature_subset, name="Selected Features").to_csv(
                results_path / f"{self.classifier_name}_best_selected_features.csv", index=False)

        logging.info(f"Saved metrics/classifier to {results_path}")
        return {'train': score_train, 'val': score_val, 'test': score_test}

    def auto_ml(self, train=False, save=False, undersample=False, smote=True, adasyn=False,
                use_experiment_folds=True, target='class'):
        """
        undersample/smote/adasyn are mutually exclusive resampling options for
        the training data before the AutoML search sees it - ignored when
        use_experiment_folds=True.

        use_experiment_folds=True makes naiveautoml evaluate every candidate
        pipeline against the leave-one-experiment-out-per-gas folds from
        make_experiment_cv_folds (via a custom evaluation_fun) instead of its
        default random split - the search runs once and picks the pipeline
        that performs best across real held-out experiments, rather than
        training a new classifier per fold.
        """
        metric = "f1_macro"
        logging.info("Starting Naive AutoML")
        self.classifier_name = "NaiveAutoML"
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path']) / self.classifier_name
        results_path.mkdir(parents=True, exist_ok=True)

        fold_index_pairs = None
        X_test = y_test = None
        if use_experiment_folds:
            X_all, y_all, fold_index_pairs, X_test, y_test = self.folds._build_experiment_fold_indices(target=target)
            data_init = {'train': {'X': X_all, 'y': y_all}}
        else:
            data_init, groups = utils.load_and_process_data_for_classification(
                self.folds, apply_smote=smote, apply_adasyn=adasyn, scale=True, apply_undersample=undersample
            )
        logging.info(np.unique(data_init["train"]["y"], return_counts=True))

        if train:
            evaluation_fun = None
            if use_experiment_folds:
                evaluation_fun = SplitBasedEvaluator(
                    task_type='classification', splitter=_FixedFoldSplitter(fold_index_pairs),
                    logger_name='naml.evaluator',
                )

            naml = naiveautoml.NaiveAutoML(
                scoring=metric, passive_scorings= ["accuracy", "neg_log_loss"], show_progress=True, max_hpo_iterations=10,  # 100 before
                evaluation_fun=evaluation_fun,
                kwargs_as={"excluded_components": {
                    "feature-pre-processor": ["GenericUnivariateSelect"],
                    # "learner": ["RandomForestClassifier", "ExtraTreesClassifier"],
                }},
            )  # , timeout_candidate=20)  # , timeout_overall=11, timeout_candidate=11)
            naml.fit(data_init["train"]["X"], data_init["train"]["y"])
            logging.info(f"Leaderboard (head): {naml.leaderboard.head(10)} \n"
                         f"############# \n"
                         f"naml.chosen_model: {naml.chosen_model} \n")
            pd.Series(data_init["train"]["X"].columns, name="Selected Features").to_csv(
                results_path / f"{self.classifier_name}_best_selected_features.csv", index=False)

            lb = naml.leaderboard.copy()
            best_idx = lb[metric].astype(float).idxmax()
            best_row = lb.loc[best_idx]
            best_pipeline = copy.deepcopy(best_row["pipeline"])
            # best_pipeline.fit(data_init["train"]["X"], data_init["train"]["y"])
            dump(best_pipeline, results_path / f"{self.classifier_name}_best_classifier.joblib")  # note: untrained, important to load for feature selection
            naml.leaderboard.head(50).to_csv(results_path / f"{self.classifier_name}_leaderboard.csv", index=False)

        if save:
            if use_experiment_folds:
                best_pipeline_loaded = load(results_path / f"{self.classifier_name}_best_classifier.joblib")
                fold_scores = []
                for train_idx, test_idx in fold_index_pairs:
                    pl = clone(best_pipeline_loaded)
                    pl.fit(data_init["train"]["X"].iloc[train_idx], data_init["train"]["y"].iloc[train_idx])
                    fold_scores.append(
                        pl.score(data_init["train"]["X"].iloc[test_idx], data_init["train"]["y"].iloc[test_idx])
                    )
                fold_scores = pd.Series(fold_scores, name='accuracy')
                fold_scores.index.name = 'fold'
                logging.info(f"Cross-validated dev-fold accuracy (chosen pipeline): {fold_scores.tolist()} \n"
                             f"Mean +/- std: {fold_scores.mean():.4f} +/- {fold_scores.std():.4f}")
                fold_scores.to_csv(results_path / f"{self.classifier_name}_experiment_fold_scores.csv")

                final_clf = clone(best_pipeline_loaded)
                final_clf.fit(data_init["train"]["X"], data_init["train"]["y"])
                out = results_path / f"{self.classifier_name}_final_classifier.joblib"
                dump(final_clf, out)

                # Final one-shot check on the reserved test experiments -
                # never used by the search or any dev fold.
                held_out_test_score = final_clf.score(X_test, y_test)
                logging.info(f"Held-out test accuracy (final classifier, never seen during search): "
                             f"{held_out_test_score:.4f}")
                pd.Series({'held_out_test_accuracy': held_out_test_score}).to_csv(
                    results_path / f"{self.classifier_name}_held_out_test_score.csv"
                )
                logging.info(f"Saved final classifier (refit on all dev data) to {out}")
            else:
                clf = load(results_path / f"{self.classifier_name}_best_classifier.joblib")
                clf.fit(data_init["train"]["X"], data_init["train"]["y"])
                train_score = clf.score(data_init["train"]["X"], data_init["train"]["y"])
                val_score = clf.score(data_init["val"]["X"], data_init["val"]["y"])
                test_score = clf.score(data_init["test"]["X"], data_init["test"]["y"])

                logging.info(f"Classifier: {clf['learner'].get_params()} \n"
                             f"Train score: {train_score}, Validation score: {val_score}, Test score: {test_score} \n")
                logging.info(f"Pipeline params (deep): {clf.get_params(deep=True)}")

                self.save_best_metrics(clf, data_init, feature_subset=None)

    def train_classifier(self, classifier_name="HistGradBoost", target='class', show=True, save=True,
                          undersample=False, smote=False, adasyn=True, fold=0, feature_subset_path=None,
                          feature_column=None, n_features=None):
        """
        Train a classifier (configured via classifier_config). fold selects
        which split(s) to use:
        - None (default): the flat splits_path/{train,val,test}.csv from
          make_data_set - trains one classifier.
        - an int (e.g. 3): that single fold from make_experiment_cv_folds
          (splits_path/fold_3/...) - trains one classifier on that fold.
        - "all": every fold_* directory under splits_path, as written by
          make_experiment_cv_folds - trains one classifier per fold, then
          reports per-fold scores, mean+-std accuracy across folds, and one
          confusion matrix per split summed across all folds.

        feature_subset_path, if given, points to a ranked-features CSV
        (e.g. mrmr_ranked_features.csv, multivariate_ranked_features.csv, or
        a single-column "<classifier>_best_selected_features.csv" saved by
        save_best_metrics) listing feature names to train/evaluate on -
        every other feature is dropped before fitting. Names not present in
        the loaded data are ignored.

        feature_column selects which column of feature_subset_path to read
        (e.g. "cmim", "mutual_info", "mrmr") - defaults to the first column
        if not given (or if the file only has one column, as with
        best_selected_features.csv). Ignored if feature_subset_path is None.

        n_features, if given, keeps only the first n_features names from
        that column (after dropping names not in the loaded data) - i.e. the
        top-n_features entries of that ranked list. Ignored if
        feature_subset_path is None.
        """
        if fold == "all":
            splits_path = self.folds.resolve_config_path(self.folds.config_paths['splits_path'])
            fold_indices = sorted(int(p.name.split('_')[1]) for p in splits_path.glob('fold_*') if p.is_dir())
            if not fold_indices:
                raise FileNotFoundError(f"No fold_* directories found in {splits_path} - "
                                         f"run make_experiment_cv_folds first.")

            fold_scores = []
            summed_cms = None
            class_labels = None
            for f in fold_indices:
                print(f"\n=== Fold {f} ===")
                clf, scores, cms = self._train_classifier_single(
                    classifier_name=classifier_name, target=target, show=False, save=save,
                    undersample=undersample, smote=smote, adasyn=adasyn, fold=f,
                    feature_subset_path=feature_subset_path, feature_column=feature_column,
                    n_features=n_features,
                )
                fold_scores.append(scores)
                if summed_cms is None:
                    summed_cms = {name: cm.copy() for name, cm in cms.items()}
                    class_labels = clf.classes_
                else:
                    for name in summed_cms:
                        summed_cms[name] += cms[name]

            scores_df = pd.DataFrame(fold_scores, index=fold_indices)
            scores_df.index.name = 'fold'
            print("\n=== Per-fold accuracy ===")
            print(scores_df)
            print("\n=== Mean +/- std accuracy across folds ===")
            print(pd.concat([scores_df.mean().rename('mean'), scores_df.std().rename('std')], axis=1))

            for name, cm in summed_cms.items():
                disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
                fig, ax = plt.subplots(figsize=(7, 6))
                disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
                ax.set_title(f"{classifier_name} — {name} confusion matrix (summed over {len(fold_indices)} folds)")
                fig.tight_layout()

                if save:
                    figures_path = self.folds.resolve_config_path(self.folds.config_paths['figures_path'])
                    figures_path.mkdir(parents=True, exist_ok=True)
                    out = figures_path / f"{classifier_name}_all_folds_{name}_confusion_matrix.png"
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")

                if show:
                    plt.show()
                plt.close(fig)

            if save:
                results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path'])
                results_path.mkdir(parents=True, exist_ok=True)
                out = results_path / f"{classifier_name}_all_folds_scores.csv"
                scores_df.to_csv(out)
                print(f"Saved {out}")

            return scores_df, summed_cms

        return self._train_classifier_single(
            classifier_name=classifier_name, target=target, show=show, save=save,
            undersample=undersample, smote=smote, adasyn=adasyn, fold=fold,
            feature_subset_path=feature_subset_path, feature_column=feature_column, n_features=n_features,
        )

    def _train_classifier_single(self, classifier_name="HistGradBoost", target='class', show=True, save=True,
                                  undersample=True, smote=False, adasyn=False, fold=None,
                                  feature_subset_path="", feature_column=None, n_features=None):
        """
        Train a classifier (configured via classifier_config), using
        load_and_process_data_for_classification for loading/scaling/
        resampling (same helper auto_ml uses) instead of calling
        load_split/min_max_scaling/undersample/smote_oversample by hand.
        smote and adasyn are mutually exclusive oversampling options (only
        meant to use one at a time). See train_classifier for what fold,
        feature_subset_path, feature_column and n_features select.
        Reports accuracy on each split, plots/saves a confusion matrix for
        every split, and - when save=True - saves metrics/the fitted
        classifier via save_best_metrics.
        """
        self.classifier_name = classifier_name if fold is None else f"{classifier_name}_fold{fold}"
        data_init, groups = utils.load_and_process_data_for_classification(
            self.folds, apply_smote=smote, apply_adasyn=adasyn, scale=True, apply_undersample=undersample,
            target=target, fold=fold
        )

        feature_subset = None
        if feature_subset_path:
            ranked_df = pd.read_csv(feature_subset_path)
            column = feature_column if feature_column is not None else ranked_df.columns[0]
            feature_subset = ranked_df[column].dropna().tolist()
            feature_subset = [f for f in feature_subset if f in data_init['train']['X'].columns]
            if n_features is not None:
                feature_subset = feature_subset[:n_features]
            print(f"Loaded {len(feature_subset)} features from {feature_subset_path} (column: {column})")
            for name in ('train', 'val', 'test'):
                data_init[name]['X'] = data_init[name]['X'][feature_subset]

        clf = self._build_classifier(classifier_name)
        clf.fit(data_init["train"]["X"], data_init["train"]["y"])

        scores = {}
        cms = {}
        for name in ('train', 'val', 'test'):
            X_split, y_split = data_init[name]["X"], data_init[name]["y"]
            y_pred = clf.predict(X_split)
            score = clf.score(X_split, y_split)
            scores[name] = score
            print(f"[{self.classifier_name}] {name.capitalize()} accuracy: {score:.4f}")
            print(classification_report(y_split, y_pred))

            cm = confusion_matrix(y_split, y_pred, labels=clf.classes_)
            cms[name] = cm
            print(f"Confusion matrix ({name}, rows=true, cols=predicted):")
            print(pd.DataFrame(cm, index=clf.classes_, columns=clf.classes_))

            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=clf.classes_)
            fig, ax = plt.subplots(figsize=(7, 6))
            disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
            ax.set_title(f"{self.classifier_name} — {name} confusion matrix")
            fig.tight_layout()

            if save:
                figures_path = self.folds.resolve_config_path(self.folds.config_paths['figures_path'])
                figures_path.mkdir(parents=True, exist_ok=True)
                out = figures_path / f"{self.classifier_name}_{name}_confusion_matrix.png"
                fig.savefig(out, dpi=150)
                print(f"Saved {out}")

            if show:
                plt.show()
            plt.close(fig)

        if save:
            self.save_best_metrics(clf, data_init, feature_subset=feature_subset)

        return clf, scores, cms

    def compute_feature_subset_accuracy(self, classifier_name="TabPFN", target='class',
                                         ranked_features_path=None, use_majority_rank_aggregation=False,
                                         max_features=100, save=True, fold=0):
        """
        For each feature-selection approach (column) in a ranked-features
        CSV, train classifier_name repeatedly using only that approach's
        top-1, top-2, ..., top-max_features features, plus one "all_features"
        baseline trained on every available feature. Tracks train/val/test
        accuracy at every step.

        Unlike the old train_classifier_feature_subset, this does not plot
        anything itself - it saves one long-format CSV (columns: source,
        approach, n_features, split, accuracy) per call, under results_path.
        Call this once per ranked-features file/approach set you want to
        compare, then use plot_feature_subset_accuracy to load any
        combination of the resulting tables onto one figure.

        Which file is read is chosen, in order of priority:
        1. ranked_features_path, if given - any ranked-features CSV, e.g.
           mrmr_ranked_features.csv or multivariate_ranked_features.csv.
        2. use_majority_rank_aggregation=True - majority_rank_ranked_features.csv
           (majority_voting + rank_aggregation, from
           FeatureSelection.combine_majority_rank_aggregation).
        3. Default: univariate_ranked_features.csv (mutual_info/anova/chi2).

        fold=None (default) reads the flat splits_path/{train,val,test}.csv
        written by make_data_set - pass a fold index (e.g. fold=0) to
        instead read splits_path/fold_<fold>/{train,val,test}.csv, as
        written by make_experiment_cv_folds, when the flat split hasn't been
        generated.
        """
        self.classifier_name = classifier_name
        data_init, groups = utils.load_and_process_data_for_classification(
            self.folds, apply_smote=False, scale=True, apply_undersample=False, target=target, fold=fold
        )

        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path'])
        if ranked_features_path is None:
            default_name = "majority_rank_ranked_features.csv" if use_majority_rank_aggregation \
                else "univariate_ranked_features.csv"
            ranked_features_path = results_path / default_name
        ranked_df = pd.read_csv(ranked_features_path, index_col=0)
        approaches = [c for c in ranked_df.columns if not c.endswith('_score')]
        source = Path(ranked_features_path).stem

        available = set(data_init['train']['X'].columns)
        rows = []

        for approach in approaches:
            ranked_list = [f for f in ranked_df[approach].dropna().tolist() if f in available]
            n_max = min(max_features, len(ranked_list))
            print(f"=== {approach} ({n_max} feature counts) ===")

            for n in range(1, n_max + 1):
                print(f"{approach}_{n}")
                subset = ranked_list[:n]
                clf = self._build_classifier(classifier_name)
                clf.fit(data_init['train']['X'][subset], data_init['train']['y'])

                for split in ('train', 'val', 'test'):
                    X_split, y_split = data_init[split]['X'][subset], data_init[split]['y']
                    score = clf.score(X_split, y_split)
                    f1 = f1_score(y_split, clf.predict(X_split), average='macro')
                    rows.append({'source': source, 'approach': approach, 'n_features': n,
                                 'split': split, 'accuracy': score, 'f1_score': f1})

            if n_max:
                last = {r['split']: r['accuracy'] for r in rows
                        if r['approach'] == approach and r['n_features'] == n_max}
                print(f"  final (n={n_max}) train/val/test accuracy: "
                      f"{last['train']:.4f} / {last['val']:.4f} / {last['test']:.4f}")

        # "all_features" baseline: trained once on every available feature,
        # stored as a single row per split (n_features = total count) -
        # plot_feature_subset_accuracy draws single-row approaches as a
        # flat reference line rather than a point.
        all_features = sorted(available)
        clf = self._build_classifier(classifier_name)
        clf.fit(data_init['train']['X'][all_features], data_init['train']['y'])
        for split in ('train', 'val', 'test'):
            X_split, y_split = data_init[split]['X'][all_features], data_init[split]['y']
            score = clf.score(X_split, y_split)
            f1 = f1_score(y_split, clf.predict(X_split), average='macro')
            rows.append({'source': source, 'approach': 'all_features', 'n_features': len(all_features),
                         'split': split, 'accuracy': score, 'f1_score': f1})

        table = pd.DataFrame(rows)

        if save:
            tables_dir = results_path / "feature_acc_lists_to_plot"
            tables_dir.mkdir(parents=True, exist_ok=True)
            out = tables_dir / f"{classifier_name}_{source}_feature_subset_accuracy.csv"
            table.to_csv(out, index=False)
            print(f"Saved {out}")

        return table

    def plot_feature_subset_accuracy(self, classifier_name="HistGradBoost",
                                      out_name=None, show=True, save=True, metric="accuracy"):
        """
        Load every CSV in results_path/feature_acc_lists_to_plot (each
        written by compute_feature_subset_accuracy) and plot metric vs
        number of features on one figure - one subplot per split, one line
        per (source, approach) combination - so any combination of
        previously computed approaches/sources can be compared together
        just by having their tables sit in that folder. Series with a
        single row (e.g. the "all_features" baseline) are drawn as a flat
        dashed reference line instead of a single point.

        metric selects which column to plot: "accuracy" (default) or
        "f1_score" - both are saved by compute_feature_subset_accuracy.
        """
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path'])
        tables_dir = results_path / "feature_acc_lists_to_plot"
        csv_files = sorted(tables_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No accuracy tables found in {tables_dir}")

        frames = [pd.read_csv(f) for f in csv_files]
        data = pd.concat(frames, ignore_index=True)
        data['series'] = data['source'] + ':' + data['approach']

        fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        # baselines (single-row series, e.g. "all_features") shouldn't
        # stretch the x-axis out to their own n_features - scale the axis
        # to the actual swept approaches instead.
        swept = data.groupby('series').filter(lambda g: g['n_features'].nunique() > 1)
        x_min, x_max = swept['n_features'].min(), swept['n_features'].max()
        for ax, split in zip(axes, ('train', 'val', 'test')):
            split_data = data[data['split'] == split]
            for series_name, group in split_data.groupby('series'):
                group = group.sort_values('n_features')
                if len(group) == 1:
                    ax.hlines(group[metric].iloc[0], x_min, x_max, linestyles='--', label=series_name)
                else:
                    ax.plot(group['n_features'], group[metric], label=series_name, marker='.')
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{split.capitalize()} {metric}")
            ax.set_ylabel(metric.replace('_', ' ').capitalize())
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Number of features")
        fig.suptitle(f"{classifier_name} {metric} vs number of features")
        fig.tight_layout()

        if save:
            figures_path = self.folds.resolve_config_path(self.folds.config_paths['figures_path'])
            figures_path.mkdir(parents=True, exist_ok=True)
            out_name = out_name or f"{classifier_name}_feature_subset_{metric}_combined.png"
            out = figures_path / out_name
            fig.savefig(out, dpi=150)
            print(f"Saved {out}")

        if show:
            plt.show()
        plt.close(fig)

        return data


if __name__ == "__main__":
    GC = GasClassification()
    #GC.folds.make_experiment_cv_folds()
    #GC.folds.make_data_set()
    for classifier in ["TabPFN"]:#["AutoML"]:#"HGB", "RF", "ETC"]:TabICL
        GC.train_classifier(classifier, feature_column="cmim")
    #GC.auto_ml(train=True, save=True)
    #GC.train_classifier_feature_subset()
    #GC.compute_feature_subset_accuracy(use_majority_rank_aggregation=False, max_features=200, save=True)
    #GC.compute_feature_subset_accuracy(ranked_features_path= "03_results/multivariate_ranked_features.csv", use_majority_rank_aggregation=False, max_features=200, save=True, )
    #GC.plot_feature_subset_accuracy(classifier_name="TabPFN",metric="accuracy")
    #data_init, groups = utils.load_and_process_data_for_classification(GC.folds, apply_smote=False, scale=True)
    #fs = FeatureSelection()
    #fs.apply_feature_selection(data_init, groups, save=True)
    #fs.aggregate_features(majority_voting=True, rank_aggregation=True, use_mrmr=True)
    #fs.apply_mrmr(data_init, None, save=True)
    # fs.apply_multivariate_feature_selection(data_init,k=10000,save=True)
