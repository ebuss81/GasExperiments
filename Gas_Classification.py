
from pathlib import Path
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.base import clone
from kneed import KneeLocator
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
                use_experiment_folds=True, target='class',
                keep_classes=['CO2_post', 'prestimulus'], drop_classes=None, gas='CO2'):
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

        keep_classes/drop_classes/gas restrict the classification problem
        the same way they do in load_and_process_data_for_classification
        (default here: binary O3-vs-baseline, keep_classes=['O3_post',
        'prestimulus'], gas='O3') - pass keep_classes=None, gas=None for
        the full multiclass problem instead. Applied consistently whichever
        branch runs: to every dev fold and the final held-out test set via
        ExperimentFolds._build_experiment_fold_indices when
        use_experiment_folds=True, or via
        utils.load_and_process_data_for_classification otherwise. Whichever
        scope is passed is folded into self.classifier_name (via
        utils.scope_suffix), so results/figures from different scopes land
        in differently-named files instead of overwriting each other.
        """
        metric = "f1_macro"
        logging.info("Starting Naive AutoML")
        self.classifier_name = "NaiveAutoML" + utils.scope_suffix(gas, keep_classes, drop_classes)
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path']) / self.classifier_name
        results_path.mkdir(parents=True, exist_ok=True)

        fold_index_pairs = None
        X_test = y_test = None
        if use_experiment_folds:
            X_all, y_all, fold_index_pairs, X_test, y_test = self.folds._build_experiment_fold_indices(
                target=target, keep_classes=keep_classes, drop_classes=drop_classes, gas=gas,
            )
            data_init = {'train': {'X': X_all, 'y': y_all}}
        else:
            data_init, groups = utils.load_and_process_data_for_classification(
                self.folds, apply_smote=smote, apply_adasyn=adasyn, scale=True, apply_undersample=undersample,
                keep_classes=keep_classes, drop_classes=drop_classes, gas=gas,
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
                scoring=metric, passive_scorings= ["accuracy", "neg_log_loss"], show_progress=True, max_hpo_iterations=100,  # 100 before
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

                final_clf = clone(best_pipeline_loaded)
                final_clf.fit(data_init["train"]["X"], data_init["train"]["y"])

                # Performance on the reserved test experiments - never used
                # by the search, any dev fold, or (since the imputer fix)
                # even preprocessing, so this is a clean generalization
                # estimate. A per-fold refit/score wouldn't be: every fold
                # in fold_index_pairs was already used by naiveautoml's own
                # search (evaluation_fun=SplitBasedEvaluator over those same
                # folds), and its aggregate score is already on the
                # leaderboard - so there's nothing to gain by recomputing
                # it here. Reported/saved the same way
                # _train_classifier_single does per split, via
                # save_best_metrics - into the same results_path/<scope>
                # folder as everything else from this run (leaderboard,
                # best_selected_features), not a separate "_final" folder.
                # This also overwrites best_classifier.joblib with the
                # trained final_clf (previously the untrained
                # search-selected pipeline) - clone() elsewhere (e.g.
                # compute_feature_subset_accuracy) strips fitted state
                # regardless, so it's still a valid unfitted template.
                held_out_data = {'X': X_test, 'y': y_test}
                final_data = {
                    'train': {'X': data_init["train"]["X"], 'y': data_init["train"]["y"]},
                    'val': held_out_data,
                    'test': held_out_data,
                }
                figures_path = self.folds.resolve_config_path(self.folds.config_paths['figures_path'])
                figures_path.mkdir(parents=True, exist_ok=True)
                for split_name in ('train', 'val'):
                    X_split, y_split = final_data[split_name]['X'], final_data[split_name]['y']
                    y_pred = final_clf.predict(X_split)
                    print(f"[{self.classifier_name}] Final classifier {split_name} accuracy: "
                          f"{final_clf.score(X_split, y_split):.4f}")
                    print(classification_report(y_split, y_pred))

                    cm = confusion_matrix(y_split, y_pred, labels=final_clf.classes_)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=final_clf.classes_)
                    fig, ax = plt.subplots(figsize=(7, 6))
                    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
                    ax.set_title(f"{self.classifier_name} — final classifier {split_name} confusion matrix")
                    fig.tight_layout()
                    out_fig = figures_path / f"{self.classifier_name}_final_{split_name}_confusion_matrix.png"
                    fig.savefig(out_fig, dpi=150)
                    print(f"Saved {out_fig}")
                    plt.close(fig)

                self.save_best_metrics(final_clf, final_data, feature_subset=None)
                logging.info(f"Saved final classifier (refit on all dev data) to "
                             f"{results_path / f'{self.classifier_name}_best_classifier.joblib'}")
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
                                  undersample=False, smote=False, adasyn=False, fold=None,
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

    def compute_feature_subset_accuracy(self, target='class',
                                         ranked_features_path=None, use_aggregated_ranking=False,
                                         max_features=100, save=True, fold=0,
                                         keep_classes=None, drop_classes=None, gas=None,
                                         use_experiment_folds=False, n_features_grid=None):
        """
        For each feature-selection approach (column) in a ranked-features
        CSV, repeatedly clone+train the AutoML-selected best classifier for
        this scope, using only that approach's top-1, top-2, ...,
        top-max_features features, plus one "all_features" baseline trained
        on every available feature. Tracks train/val/test accuracy at every
        step.

        keep_classes/drop_classes/gas select the classification scope, the
        same way they do in load_and_process_data_for_classification and
        auto_ml - both which classifier gets loaded and which ranked-
        features file gets read depend on this scope:
        - Classifier: loaded from
          results_path/NaiveAutoML<scope_suffix>/NaiveAutoML<scope_suffix>_best_classifier.joblib
          (dumped by auto_ml - untrained if only train=True has run, fit on
          all dev data if save=True has also run) and re-cloned fresh for
          every n_features step, since each step needs its own fit on a
          different feature subset.
        - Ranked features (when ranked_features_path is None): read from
          results_path/03_01_feature_selection/, with the same
          <scope_suffix> FeatureSelection's apply_* methods save with.
        Run auto_ml(train=True, ...) and the relevant FeatureSelection
        apply_* method with the same keep_classes/drop_classes/gas first,
        or this will fail to find either file.

        Unlike the old train_classifier_feature_subset, this does not plot
        anything itself - it saves one long-format CSV (columns: source,
        approach, n_features, split, accuracy, f1_score, fold) per call,
        under results_path. Call this once per ranked-features file/
        approach set you want to compare, then use
        plot_feature_subset_accuracy to load any combination of the
        resulting tables onto one figure.

        Which ranked-features file is read is chosen, in order of priority:
        1. ranked_features_path, if given - any ranked-features CSV, e.g.
           mrmr_ranked_features.csv or multivariate_ranked_features.csv.
        2. use_aggregated_ranking=True - aggregated_ranked_features.csv
           (mean/median/product across every univariate+multivariate
           method, from FeatureSelection.aggregate_features).
        3. Default: univariate_ranked_features.csv (mutual_info/anova/relief).

        fold=None (default) reads the flat splits_path/{train,val,test}.csv
        written by make_data_set - pass a fold index (e.g. fold=0) to
        instead read splits_path/fold_<fold>/{train,val,test}.csv, as
        written by make_experiment_cv_folds, when the flat split hasn't been
        generated. Ignored when use_experiment_folds=True (see below) -
        that's a different notion of "fold" from this one.

        use_experiment_folds=True switches from that single fixed train/
        val/test split to the same leave-one-experiment-out-per-gas CV
        folds auto_ml uses (ExperimentFolds._build_experiment_fold_indices)
        - the whole sweep below runs once per fold instead of once total,
        giving repeated measurements per n_features (a real spread across
        folds) rather than a single point estimate. Each fold's own
        held-out rows become that run's 'val' split; the one experiment
        per gas reserved as the final test set is reused as 'test' for
        every fold (it's the same genuinely unseen data regardless of
        fold). The saved table's 'fold' column records which run each row
        came from (None in the default single-split mode).

        n_features_grid, if given (a list of ints), sweeps exactly those
        n_features values (deduplicated, sorted, clipped to
        min(max_features, available features)) instead of every integer
        from 1 to max_features - much cheaper, and close to necessary once
        use_experiment_folds multiplies the work by the fold count.
        Defaults to a log-spaced grid ([1, 2, 3, 5, 10, 20, 30, 50, 75,
        100, 150, 200, 300, 500, 782]) when use_experiment_folds=True and
        no grid is given; with use_experiment_folds=False the dense
        range(1, n_max+1) sweep is kept unless a grid is explicitly passed.
        """
        self.classifier_name = "NaiveAutoML" + utils.scope_suffix(gas, keep_classes, drop_classes)
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path'])

        if use_experiment_folds:
            X_dev, y_dev, fold_index_pairs, X_test_final, y_test_final = self.folds._build_experiment_fold_indices(
                target=target, keep_classes=keep_classes, drop_classes=drop_classes, gas=gas,
            )
            fold_runs = [
                {'fold': i,
                 'train': {'X': X_dev.iloc[train_idx], 'y': y_dev.iloc[train_idx]},
                 'val': {'X': X_dev.iloc[test_idx], 'y': y_dev.iloc[test_idx]},
                 'test': {'X': X_test_final, 'y': y_test_final}}
                for i, (train_idx, test_idx) in enumerate(fold_index_pairs)
            ]
            if n_features_grid is None:
                n_features_grid = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150]#, 200, 300, 500, 782]
        else:
            data_init, groups = utils.load_and_process_data_for_classification(
                self.folds, apply_smote=False, scale=True, apply_undersample=False, target=target, fold=fold,
                keep_classes=keep_classes, drop_classes=drop_classes, gas=gas,
            )
            fold_runs = [{'fold': None, **data_init}]

        best_pipeline_template = load(
            results_path / self.classifier_name / f"{self.classifier_name}_best_classifier.joblib"
        )
        # Use just the final 'learner' step (naiveautoml's own pipeline
        # step name - see the clf['learner'] usage in auto_ml), not the
        # whole pipeline: the pipeline's internal feature-preprocessor was
        # tuned assuming the full feature set, and can collapse a tiny
        # externally-chosen n-feature subset down to 0 columns (crashing
        # the final estimator) - counter to the whole point of this sweep,
        # which is to see how accuracy changes with an externally chosen
        # feature count. load_and_process_data_for_classification already
        # scales/imputes upstream, so the pipeline's other preprocessing
        # steps aren't needed here either.
        best_learner_template = best_pipeline_template['learner'] if hasattr(best_pipeline_template, 'named_steps') \
            else best_pipeline_template
        # naiveautoml's search doesn't fix random_state as a tuned
        # hyperparameter, so the picked learner is often left at sklearn's
        # default (None) - meaning separate calls to this method (e.g. one
        # per ranked-features file) would each fit differently on the same
        # data purely from training-time randomness (bootstrap sampling,
        # random splits, ...), not any real data difference. Pin it so
        # every clone(best_learner_template).fit(...) below is reproducible.

        if 'random_state' in best_learner_template.get_params():
            best_learner_template.set_params(random_state=42)

        if ranked_features_path is None:
            suffix = utils.scope_suffix(gas, keep_classes, drop_classes)
            default_name = f"aggregated_ranked_features{suffix}.csv" if use_aggregated_ranking \
                else f"univariate_ranked_features{suffix}.csv"
            ranked_features_path = results_path / "03_01_feature_selection" / default_name
        ranked_df = pd.read_csv(ranked_features_path, index_col=0)
        approaches = [c for c in ranked_df.columns if not c.endswith('_score')]
        source = Path(ranked_features_path).stem

        rows = []

        for run in fold_runs:
            fold_id = run['fold']
            available = set(run['train']['X'].columns)

            for approach in approaches:
                ranked_list = [f for f in ranked_df[approach].dropna().tolist() if f in available]
                n_max = min(max_features, len(ranked_list))
                if n_features_grid is not None:
                    sweep_ns = sorted({n for n in n_features_grid if 1 <= n <= n_max})
                else:
                    sweep_ns = list(range(1, n_max + 1))
                print(f"=== fold={fold_id} {approach} ({len(sweep_ns)} feature counts) ===")

                for n in sweep_ns:
                    print(f"{approach}_{n}")
                    subset = ranked_list[:n]
                    clf = clone(best_learner_template)
                    clf.fit(run['train']['X'][subset], run['train']['y'])

                    for split in ('train', 'val', 'test'):
                        X_split, y_split = run[split]['X'][subset], run[split]['y']
                        score = clf.score(X_split, y_split)
                        f1 = f1_score(y_split, clf.predict(X_split), average='macro')
                        rows.append({'source': source, 'approach': approach, 'n_features': n,
                                     'split': split, 'accuracy': score, 'f1_score': f1, 'fold': fold_id})

                if sweep_ns:
                    last_n = sweep_ns[-1]
                    last = {r['split']: r['accuracy'] for r in rows
                            if r['approach'] == approach and r['n_features'] == last_n and r['fold'] == fold_id}
                    print(f"  final (n={last_n}) train/val/test accuracy: "
                          f"{last['train']:.4f} / {last['val']:.4f} / {last['test']:.4f}")

            # "all_features" baseline: trained once on every available
            # feature, stored as a single row per split (n_features =
            # total count) - plot_feature_subset_accuracy draws single-row
            # approaches as a flat reference line rather than a point.
            all_features = sorted(available)
            clf = clone(best_learner_template)
            clf.fit(run['train']['X'][all_features], run['train']['y'])
            for split in ('train', 'val', 'test'):
                X_split, y_split = run[split]['X'][all_features], run[split]['y']
                score = clf.score(X_split, y_split)
                f1 = f1_score(y_split, clf.predict(X_split), average='macro')
                rows.append({'source': source, 'approach': 'all_features', 'n_features': len(all_features),
                             'split': split, 'accuracy': score, 'f1_score': f1, 'fold': fold_id})

        table = pd.DataFrame(rows)

        if save:
            tables_dir = results_path / "feature_acc_lists_to_plot"
            tables_dir.mkdir(parents=True, exist_ok=True)
            out = tables_dir / f"{self.classifier_name}_{source}_feature_subset_accuracy.csv"
            table.to_csv(out, index=False)
            print(f"Saved {out}")

        return table

    def _aggregate_over_folds(self, group, column):
        """
        Collapse a (possibly multi-fold) group of rows sharing the same
        series/split down to one row per n_features: the mean of `column`
        across folds, plus a "{column}_std" column - NaN wherever there's
        only a single fold/run at that n_features (e.g. old tables from
        before compute_feature_subset_accuracy(use_experiment_folds=True)
        existed, which only ever have one row per n_features - nothing to
        take a std of). Shared by plot_feature_subset_accuracy and
        get_best_feature_subsets_metrics so both aggregate the same way.
        """
        agg = group.groupby('n_features', as_index=False)[column].agg(['mean', 'std'])
        return agg.rename(columns={'mean': column, 'std': f'{column}_std'})

    def _select_best_n_features(self, val, min_delta, method, column='accuracy', total_features=None,
                                 lambda_penalty=0.1):
        """
        Pick a "best" row from `val` (a DataFrame with 'n_features' and
        `column` columns, sorted by n_features and already smoothed if
        desired) via method='tolerance'|'knee'|'penalized' - shared by
        get_best_feature_subsets_metrics and plot_feature_subset_accuracy
        so both pick the same point the same way. See
        get_best_feature_subsets_metrics's docstring for what each method
        does.
        """
        peak = val[column].max()
        if method == 'knee' and len(val) >= 3:
            knee = KneeLocator(val['n_features'], val[column], curve='concave', direction='increasing')
            if knee.knee is not None:
                return val.loc[(val['n_features'] - knee.knee).abs().idxmin()]
        if method == 'penalized':
            # score(k) = performance(k) - lambda * (k / total_features):
            # directly trades off validation performance against feature
            # count, rather than a hard tolerance/cutoff rule. Pick the
            # n_features that maximizes this score. total_features
            # defaults to this series' own largest swept n_features if
            # the true full feature count isn't known.
            n_total = total_features or val['n_features'].max()
            score = val[column] - lambda_penalty * (val['n_features'] / n_total)
            return val.loc[score.idxmax()]
        # 'tolerance' (default), or 'knee' with too few points / no
        # detectable bend - smallest n_features within min_delta of the
        # true peak, not the raw argmax, so a negligible gain doesn't win
        # out over fewer features.
        within_tolerance = val[val[column] >= peak - min_delta]
        return within_tolerance.iloc[0]

    def plot_feature_subset_accuracy(self, out_name=None, show=True, save=True, metric="accuracy",
                                      keep_classes=None, drop_classes=None, gas=None, rolling_window=None,
                                      mark_best=False, min_delta=0.01, method='tolerance', lambda_penalty=0.5):
        """
        Load every CSV in results_path/feature_acc_lists_to_plot whose
        filename matches this scope (written by compute_feature_subset_accuracy
        for the same keep_classes/drop_classes/gas) and plot metric vs
        number of features on one figure - one subplot per split, one line
        per (source, approach) combination - so any combination of
        previously computed ranked-features files *for this scope* can be
        compared together just by having their tables sit in that folder.
        Series with a single distinct n_features value (e.g. the
        "all_features" baseline) are drawn as a flat dashed reference
        line instead of a single point.

        If the table has multiple folds (compute_feature_subset_accuracy
        was called with use_experiment_folds=True), each series' rows are
        first collapsed to one point per n_features: the mean across
        folds, plotted as the line, with a shaded +-1 std band around it.
        Old single-run tables (one row per n_features already) are
        unaffected - the "mean" is just that one value and there's no
        band to draw.

        keep_classes/drop_classes/gas must match what
        compute_feature_subset_accuracy was called with - only tables saved
        under the matching "NaiveAutoML<scope_suffix>_..." prefix are
        loaded, so tables from different scopes (e.g. CO2 vs O3 binary
        problems) never end up mixed on the same figure.

        metric selects which column to plot: "accuracy" (default) or
        "f1_score" - both are saved by compute_feature_subset_accuracy.

        rolling_window, if given (an int > 1), smooths each plotted line
        with a centered rolling mean of that many points over n_features
        (min_periods=1, so the ends aren't cut short) - useful since the
        raw accuracy-vs-n_features curve is often noisy step to step. Only
        affects what's drawn, not the returned/saved data. Single-row
        baseline series (e.g. "all_features") are left as-is - a rolling
        mean of one point is meaningless.

        mark_best, if True, marks each multi-point series' "best"
        n_features (picked from its val curve via
        get_best_feature_subsets_metrics's same min_delta/method/
        lambda_penalty logic - see that method's docstring) with a star
        on every subplot, in that series' own line color. Note the point
        is always picked from the *val* metric curve, same as
        get_best_feature_subsets_metrics, even when plotting
        metric="f1_score" - so what's marked is "best by validation
        metric", consistent across subplots.
        """
        self.classifier_name = "NaiveAutoML" + utils.scope_suffix(gas, keep_classes, drop_classes)
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path'])
        tables_dir = results_path / "feature_acc_lists_to_plot"
        csv_files = sorted(tables_dir.glob(f"{self.classifier_name}_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No accuracy tables found in {tables_dir} for scope {self.classifier_name}")

        frames = [pd.read_csv(f) for f in csv_files]
        data = pd.concat(frames, ignore_index=True)
        data['series'] = data['source'] + ':' + data['approach']

        best_n_by_series = {}
        if mark_best:
            all_features_rows = data[data['approach'] == 'all_features']
            total_features = all_features_rows['n_features'].max() if not all_features_rows.empty else None
            val_only = data[data['split'] == 'val']
            for series_name, group in val_only.groupby('series'):
                group = self._aggregate_over_folds(group, metric)
                if len(group) < 2:
                    continue  # nothing to mark on a single-point baseline
                if rolling_window and rolling_window > 1:
                    group[metric] = group[metric].rolling(window=rolling_window, min_periods=1, center=True).mean()
                best = self._select_best_n_features(group, min_delta, method, column=metric,
                                                     total_features=total_features, lambda_penalty=lambda_penalty)
                best_n_by_series[series_name] = best['n_features']

        fig, axes = plt.subplots(3, 1, figsize=(9, 12), sharex=True)
        # baselines (single-row series, e.g. "all_features") shouldn't
        # stretch the x-axis out to their own n_features - scale the axis
        # to the actual swept approaches instead.
        swept = data.groupby('series').filter(lambda g: g['n_features'].nunique() > 1)
        x_min, x_max = swept['n_features'].min(), swept['n_features'].max()
        for ax, split in zip(axes, ('train', 'val', 'test')):
            split_data = data[data['split'] == split]
            for series_name, group in split_data.groupby('series'):
                # Collapse multiple folds' rows at the same n_features into
                # one mean (+ std, for the shaded band below) - a no-op
                # for old single-run tables, which only ever have one row
                # per n_features already.
                group = self._aggregate_over_folds(group, metric)
                if len(group) == 1:
                    ax.hlines(group[metric].iloc[0], x_min, x_max, linestyles='--', label=series_name)
                else:
                    y = group[metric]
                    y_std = group[f'{metric}_std']
                    if rolling_window and rolling_window > 1:
                        y = y.rolling(window=rolling_window, min_periods=1, center=True).mean()
                        y_std = y_std.rolling(window=rolling_window, min_periods=1, center=True).mean()
                    line, = ax.plot(group['n_features'], y, label=series_name, marker='.')
                    if y_std.notna().any():
                        ax.fill_between(group['n_features'], y - y_std.fillna(0), y + y_std.fillna(0),
                                         color=line.get_color(), alpha=0.15, linewidth=0)
                    if series_name in best_n_by_series:
                        best_n = best_n_by_series[series_name]
                        match = group['n_features'] == best_n
                        if match.any():
                            ax.scatter(best_n, y.loc[match.idxmax()], marker='*', s=200,
                                       color=line.get_color(), edgecolor='black', linewidth=0.8, zorder=5)
            ax.set_xlim(x_min, x_max)
            ax.set_title(f"{split.capitalize()} {metric}")
            ax.set_ylabel(metric.replace('_', ' ').capitalize())
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Number of features")
        fig.suptitle(f"{self.classifier_name} {metric} vs number of features")
        fig.tight_layout()

        if save:
            figures_path = self.folds.resolve_config_path(self.folds.config_paths['figures_path'])
            figures_path.mkdir(parents=True, exist_ok=True)
            out_name = out_name or f"{self.classifier_name}_feature_subset_{metric}_combined.png"
            out = figures_path / out_name
            fig.savefig(out, dpi=150)
            print(f"Saved {out}")

        if show:
            plt.show()
        plt.close(fig)

        return data

    def get_best_feature_subsets_metrics(self, keep_classes=None, drop_classes=None, gas=None, min_delta=0.01,
                                          rolling_window=None, method='tolerance', lambda_penalty=0.1):
        """
        For every feature-selection approach found in
        results_path/feature_acc_lists_to_plot for this scope (written by
        compute_feature_subset_accuracy, same file-matching as
        plot_feature_subset_accuracy), pick a "best" n_features from that
        approach's (optionally smoothed) validation-accuracy-vs-n_features
        curve, and print/return that val accuracy alongside the accuracy
        on the test split *at that same n_features* - not the best test
        accuracy, which would leak information from picking n_features
        using the test set itself.

        method selects how "best" is picked from the curve:
        - 'tolerance' (default): the smallest n_features within min_delta
          (default 1 percentage point) of the true peak validation
          accuracy - avoids preferring a larger, noisier feature count for
          a negligible/within-noise accuracy gain. min_delta=0 recovers
          the raw-argmax behavior.
        - 'knee': the curve's knee/elbow point (kneed.KneeLocator,
          curve='concave', direction='increasing') - where accuracy stops
          rising and starts flattening out as n_features grows. Doesn't
          use min_delta at all. Falls back to the plain peak (argmax) if
          the series has fewer than 3 points or KneeLocator can't find a
          bend (e.g. a flat or strictly monotonic curve).
        - 'penalized': maximizes score(k) = val_accuracy(k) -
          lambda_penalty * (k / total_features), where total_features is
          the full feature count (from this scope's "all_features" row)
          - directly trades off validation performance against feature
          count in one objective, rather than a hard cutoff/tolerance
          rule. Larger lambda_penalty favors smaller feature subsets more
          aggressively; lambda_penalty=0 recovers the raw-argmax
          behavior.

        rolling_window, if given (an int > 1), smooths each series'
        validation accuracy with a centered rolling mean (min_periods=1,
        same as plot_feature_subset_accuracy) over n_features *before*
        picking "best" (either way) - reduces the risk of the raw
        step-to-step noise picking out a one-off spike, or kneed
        mistaking noise for the bend. The reported val accuracy is this
        smoothed value; the reported test accuracy is still the genuine
        (unsmoothed) test accuracy at the chosen n_features.

        keep_classes/drop_classes/gas must match what
        compute_feature_subset_accuracy was called with - only tables
        saved under the matching "NaiveAutoML<scope_suffix>_..." prefix
        are loaded.
        """
        self.classifier_name = "NaiveAutoML" + utils.scope_suffix(gas, keep_classes, drop_classes)
        results_path = self.folds.resolve_config_path(self.folds.config_paths['results_path'])
        tables_dir = results_path / "feature_acc_lists_to_plot"
        csv_files = sorted(tables_dir.glob(f"{self.classifier_name}_*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"No accuracy tables found in {tables_dir} for scope {self.classifier_name}")

        frames = [pd.read_csv(f) for f in csv_files]
        data = pd.concat(frames, ignore_index=True)
        data['series'] = data['source'] + ':' + data['approach']

        print(f"\n{'=' * 70}\nBest feature-subset results for {self.classifier_name}\n{'=' * 70}")

        all_features_rows = data[data['approach'] == 'all_features']
        total_features = all_features_rows['n_features'].max() if not all_features_rows.empty else None

        rows = []
        for series_name, group in data.groupby('series'):
            val = group[group['split'] == 'val']
            if val.empty:
                continue
            # Collapse multiple folds' val rows at the same n_features into
            # one mean before picking "best" - a no-op for old single-run
            # tables (one row per n_features already).
            val = self._aggregate_over_folds(val, 'accuracy')
            if rolling_window and rolling_window > 1:
                val['accuracy'] = val['accuracy'].rolling(window=rolling_window, min_periods=1, center=True).mean()
            peak_accuracy = val['accuracy'].max()

            best = self._select_best_n_features(val, min_delta, method, total_features=total_features,
                                                 lambda_penalty=lambda_penalty)
            best_n = best['n_features']

            # Same averaging for the reported test accuracy: with multiple
            # folds, each fold refit a different classifier but scored it
            # on the same held-out test set, so there's one test row per
            # fold at best_n to average, not just one to read off.
            test = group[(group['split'] == 'test') & (group['n_features'] == best_n)]
            test_acc = test['accuracy'].mean() if not test.empty else float('nan')
            test_std = test['accuracy'].std() if len(test) > 1 else float('nan')

            print(f"\n{series_name}\n{'-' * len(series_name)}")
            print(f"  best val accuracy  : {best['accuracy']:.4f}  (n_features={best_n}, "
                  f"true peak={peak_accuracy:.4f})")
            if pd.notna(test_std):
                print(f"  test accuracy      : {test_acc:.4f} +/- {test_std:.4f}  "
                      f"(at n_features={best_n}, across {len(test)} folds)")
            else:
                print(f"  test accuracy      : {test_acc:.4f}  (at n_features={best_n})")
            rows.append({'series': series_name, 'n_features': best_n,
                         'val_accuracy': best['accuracy'], 'test_accuracy': test_acc})

        results = pd.DataFrame(rows).sort_values('val_accuracy', ascending=False).reset_index(drop=True)

        print(f"\n{'=' * 70}\nRanked by val accuracy\n{'=' * 70}")
        for rank, row in results.iterrows():
            print(f"{rank + 1}. {row['series']}\n"
                  f"   val={row['val_accuracy']:.4f}  test={row['test_accuracy']:.4f}  n_features={row['n_features']}\n")

        return results


if __name__ == "__main__":
    GC = GasClassification()
    #GC.folds.make_experiment_cv_folds()
    #GC.folds.make_data_set()
    #for classifier in ["TabPFN"]:#["AutoML"]:#"HGB", "RF", "ETC"]:TabICL
    #    GC.train_classifier(classifier, feature_column="cmim")
    #GC.auto_ml(train=True, save=True)
    #GC.train_classifier_feature_subset()

    keep_classes_by_gas = [['CO2_post', 'prestimulus'], ['O3_post', 'prestimulus'], ['N2_post', 'prestimulus']]
    for classes, gas in zip(keep_classes_by_gas, ["CO2", "O3", "N2"]):
        #GC.auto_ml(train=True, save=True, keep_classes=classes, gas=gas)
        GC.compute_feature_subset_accuracy(use_aggregated_ranking=False, max_features=10000, save=True, keep_classes=classes, gas=gas, use_experiment_folds=True, n_features_grid=None)
        #multivariate_path = (
        #    GC.folds.resolve_config_path(GC.folds.config_paths['results_path']) / "03_01_feature_selection"
        #    / f"multivariate_ranked_features{utils.scope_suffix(gas, classes, None)}.csv"
        #)
        #GC.compute_feature_subset_accuracy(ranked_features_path=multivariate_path, max_features=2000, save=True,
        #                                    keep_classes=classes, gas=gas)
        GC.plot_feature_subset_accuracy(metric="accuracy", keep_classes=classes, gas=gas, rolling_window=1, mark_best=True, method='penalized')
        #data_init, groups = utils.load_and_process_data_for_classification(
        #    GC.folds, apply_smote=True, apply_adasyn=False, scale=True, apply_undersample=False,
        #    fold=0, keep_classes=classes, drop_classes=None, gas=gas,
        #)
        #fs = FeatureSelection()
        #fs.apply_univariate_feature_selection(, keep_classes=classes, gas=gas)
        #    data_init, groups, save=True, keep_classes=classes, gas=gas)
        #fs.apply_multivariate_feature_selection(data_init, k=200, save=True, keep_classes=classes, gas=gas)
        #fs.aggregate_features(keep_classes=classes, gas=gas)
        #GC.plot_feature_subset_accuracy(out_name="AutoML", metric="accuracy", keep_classes=classes, gas=gas)
        #GC.get_best_feature_subsets_metrics(keep_classes=classes, gas=gas, rolling_window=1)
    #fs.apply_mrmr(data_init, None, save=True)
    # fs.apply_multivariate_feature_selection(data_init,k=10000,save=True
