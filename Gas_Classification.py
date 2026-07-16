
from pathlib import Path
from datetime import datetime, timedelta
import json
import copy
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, f1_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.base import clone
from imblearn.over_sampling import SMOTE, ADASYN
import naiveautoml
from naiveautoml.evaluators import SplitBasedEvaluator
import matplotlib
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier
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
    def __init__(self, experiments_file=None):
        self.experiments_file = experiments_file or Path(__file__).with_name("gas_experiments.json")
        self.base_dir = Path(__file__).resolve().parent
        self.PNs = ['P1','P3']
        self.gases = ["CO2", "N2", "O3"]
        self.classifier_name = None
        self.read_experiment_config()
        with open(self.base_dir / "config.json", "r") as file:
            self.config = json.load(file)
        self.config_paths = self.config['paths']

    def resolve_config_path(self, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.base_dir / path

    def read_experiment_config(self):
        with open(self.experiments_file, "r") as file:
            self.experiment_config = json.load(file)
        self.experiment_start_times = self.experiment_config['experiment_time']

    def get_experiment_window_range(self, gas, index=0):
        """
        Return (experiment_name, start_datetime, end_datetime) for the
        experiment at `index` listed for `gas` in gas_experiments.json, so
        its windows can be held out entirely as an unseen-plant test set.

        CO2/N2 experiments carry an explicit start/end datetime. O3
        applications are event-triggered (start/end are "None" in the
        config), so the range is derived from that experiment's times.csv
        instead - padded by the -1h application-window offset and the
        2h10m window duration used when the windows were extracted.
        """
        experiment = self.experiment_config[gas][index]
        start_value = experiment.get('start_datetime')
        end_value = experiment.get('end_datetime')

        if start_value and start_value != "None":
            start_dt = datetime.strptime(start_value.strip(), "%Y-%m-%d %H:%M")
            end_dt = datetime.strptime(end_value.strip(), "%Y-%m-%d %H:%M")
            return experiment['name'], start_dt, end_dt

        storage_path = self.resolve_config_path(self.config_paths['storage_path'])
        times_file = Path(storage_path, gas, experiment['name'], "times.csv")
        times = pd.to_datetime(pd.read_csv(times_file)['times'])
        start_dt = times.min() - timedelta(hours=1)
        end_dt = times.max() - timedelta(hours=1) + timedelta(hours=2, minutes=10)
        return experiment['name'], start_dt, end_dt

    def get_first_experiment_window_range(self, gas):
        """Back-compat alias for get_experiment_window_range(gas, index=0)."""
        return self.get_experiment_window_range(gas, index=0)

    def _load_labeled_data(self, target='class'):
        """
        Combine the per-gas 10-minute feature tables (written by
        Preprocess.get_10min_calc_features) into one DataFrame, add the
        combined "class" target column, and compute the (gas, node,
        window_start) group key for every row. Shared by make_data_set and
        make_experiment_cv_folds.
        """
        features_path = self.resolve_config_path(self.config_paths['features_path'])

        frames = []
        for gas in self.gases:
            file = features_path / f"{gas}_10min_features.csv"
            if not file.is_file():
                print(f"No feature file for {gas}: {file}")
                continue
            frames.append(pd.read_csv(file))

        if not frames:
            raise FileNotFoundError(f"No feature files found in {features_path}")

        data = pd.concat(frames, ignore_index=True)
        # All prestimulus rows share one class regardless of gas (baseline
        # looks the same before any gas is applied); only poststimulus rows
        # are split per gas.
        data['class'] = data['gas'] + '_post'
        data.loc[data['prestimulus'], 'class'] = 'prestimulus'
        groups = data['gas'] + '|' + data['node'] + '|' + data['window_start'].astype(str)
        return data, groups

    def make_data_set(self, target='class', val_size=0.2, random_state=42):
        """
        Combine the per-gas 10-minute feature tables (written by
        Preprocess.get_10min_calc_features) into one classification dataset,
        then split it into train/val/test CSVs under splits_path.

        prestimulus is treated as a class too: a combined "class" column is
        added with one shared "prestimulus" class for all gases (baseline
        looks the same before any gas is applied) plus one "<gas>_post"
        class per gas for the poststimulus windows.

        The test set is the first experiment listed for each gas in
        gas_experiments.json (an unseen plant), identified via its
        start/end datetime range - not a random sample - so test
        performance reflects generalization to a plant never seen during
        training. The remaining windows are grouped by application (gas,
        node, window_start) before being split into train/val, so the
        prestimulus/poststimulus pair belonging to the same application
        always ends up in the same split.
        """
        splits_path = self.resolve_config_path(self.config_paths['splits_path'])
        splits_path.mkdir(parents=True, exist_ok=True)

        data, groups = self._load_labeled_data(target=target)

        window_start_dt = pd.to_datetime(data['window_start'])
        test_mask = pd.Series(False, index=data.index)
        for gas in self.gases:
            experiment_name, start_dt, end_dt = self.get_first_experiment_window_range(gas)
            gas_mask = (data['gas'] == gas) & (window_start_dt >= start_dt) & (window_start_dt <= end_dt)
            print(f"Held-out test experiment for {gas}: {experiment_name} "
                  f"({start_dt} to {end_dt}, {gas_mask.sum()} rows)")
            test_mask |= gas_mask

        y = data[target]
        X = data.drop(columns=[target])

        X_test, y_test = X[test_mask], y[test_mask]
        X_dev, y_dev = X[~test_mask], y[~test_mask]
        groups_dev = groups[~test_mask]

        train_idx, val_idx = next(
            GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
            .split(X_dev, y_dev, groups=groups_dev)
        )
        X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
        y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

        print(f"Train: X{X_train.shape}, y{y_train.shape}")
        print(f"Val:   X{X_val.shape},   y{y_val.shape}")
        print(f"Test:  X{X_test.shape},  y{y_test.shape}")

        splits = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}
        for name, (X_split, y_split) in splits.items():
            split_df = X_split.copy()
            split_df[target] = y_split
            out = splits_path / f"{name}.csv"
            split_df.to_csv(out, index=False)
            print(f"Saved {out}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def make_experiment_cv_folds(self, target='class', val_size=0.2, random_state=42):
        """
        Leave-one-experiment-out cross-validation, synchronized across
        gases: fold i holds out experiment[i % n_experiments] of every gas
        simultaneously (cycling for gases with fewer experiments than the
        largest gas), so every fold's test set has at least one held-out
        experiment per gas, and every experiment is held out at least once
        across all folds.

        Number of folds = the largest per-gas experiment count (e.g. with
        5 CO2 / 4 O3 / 7 N2 experiments, that's 7 folds; CO2 and O3 repeat
        some of their experiments across folds since they have fewer than
        7, but every fold still has exactly one held-out experiment from
        each of the three gases).

        Saves each fold's train/val/test CSVs under
        splits_path/fold_<i>/{train,val,test}.csv - same structure
        make_data_set produces for its single held-out-plant split, so
        load_and_process_data_for_classification/train_classifier/etc. can
        be pointed at a specific fold by passing its splits_path.
        """
        splits_path = self.resolve_config_path(self.config_paths['splits_path'])
        data, groups = self._load_labeled_data(target=target)
        window_start_dt = pd.to_datetime(data['window_start'])

        y = data[target]
        X = data.drop(columns=[target])

        n_folds = max(len(self.experiment_config[gas]) for gas in self.gases)
        print(f"Building {n_folds} leave-one-experiment-per-gas folds")

        fold_dirs = []
        for fold in range(n_folds):
            test_mask = pd.Series(False, index=data.index)
            held_out = {}
            for gas in self.gases:
                n_experiments = len(self.experiment_config[gas])
                exp_index = fold % n_experiments
                experiment_name, start_dt, end_dt = self.get_experiment_window_range(gas, exp_index)
                held_out[gas] = experiment_name
                gas_mask = (data['gas'] == gas) & (window_start_dt >= start_dt) & (window_start_dt <= end_dt)
                test_mask |= gas_mask

            X_test, y_test = X[test_mask], y[test_mask]
            X_dev, y_dev = X[~test_mask], y[~test_mask]
            groups_dev = groups[~test_mask]

            train_idx, val_idx = next(
                GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
                .split(X_dev, y_dev, groups=groups_dev)
            )
            X_train, X_val = X_dev.iloc[train_idx], X_dev.iloc[val_idx]
            y_train, y_val = y_dev.iloc[train_idx], y_dev.iloc[val_idx]

            print(f"Fold {fold}: held out {held_out} "
                  f"-> Train: X{X_train.shape}, Val: X{X_val.shape}, Test: X{X_test.shape}")

            fold_dir = splits_path / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            splits = {'train': (X_train, y_train), 'val': (X_val, y_val), 'test': (X_test, y_test)}
            for name, (X_split, y_split) in splits.items():
                split_df = X_split.copy()
                split_df[target] = y_split
                split_df.to_csv(fold_dir / f"{name}.csv", index=False)
            print(f"  Saved to {fold_dir}")
            fold_dirs.append(fold_dir)

        return fold_dirs

    def _build_experiment_fold_indices(self, target='class'):
        """
        Reserve the first experiment listed for each gas (index 0 - the same
        experiments make_data_set holds out as its final test set) as a
        completely untouched final test set, never used by the AutoML
        search. The *remaining* experiments (index 1..n-1 per gas) form the
        dev pool, which is split into leave-one-experiment-out-per-gas folds
        for naiveautoml's custom evaluator to use during model selection.

        Returns (X_dev, y_dev, fold_index_pairs, X_test, y_test):
        - X_dev/y_dev: every row not in the reserved test experiments,
          re-indexed 0..n-1 so fold_index_pairs' positions line up.
        - fold_index_pairs: list of (train_index, test_index) position
          arrays into X_dev/y_dev - number of folds = max(per-gas dev
          experiment count) across gases.
        - X_test/y_test: rows from the reserved (index 0) experiment per
          gas - never touched by any fold, for a final one-shot evaluation.

        Columns are median-imputed globally (not per-fold/split) purely so
        NaN-intolerant sklearn components in naiveautoml's search space don't
        crash - a minor, disclosed simplification (per-fold-train-only
        imputation would need a custom pipeline step). Scaling is
        intentionally NOT done here: naiveautoml's own candidate pipelines
        already search over data-pre-processor components (MinMaxScaler,
        ...), and its evaluator clones + refits the whole pipeline per fold -
        so leaving scaling to it means it's correctly fit on each fold's train
        rows only, with no leakage.
        """
        data, groups = self._load_labeled_data(target=target)
        window_start_dt = pd.to_datetime(data['window_start'])

        y_full = data[target]
        meta_columns = ['gas', 'node', 'window_start', 'signal', 'sub_window_start', 'sub_window_end',
                         'prestimulus', target]
        X_full = data.drop(columns=[c for c in meta_columns if c in data.columns])

        imputer = SimpleImputer(strategy='median')
        X_full = pd.DataFrame(imputer.fit_transform(X_full), columns=imputer.get_feature_names_out(X_full.columns),
                               index=X_full.index)

        # Reserve experiment index 0 per gas as the final, untouched test
        # set - the same experiments make_data_set holds out.
        final_test_mask = pd.Series(False, index=data.index)
        for gas in self.gases:
            experiment_name, start_dt, end_dt = self.get_experiment_window_range(gas, 0)
            gas_mask = (data['gas'] == gas) & (window_start_dt >= start_dt) & (window_start_dt <= end_dt)
            print(f"Reserved final test experiment for {gas}: {experiment_name}")
            final_test_mask |= gas_mask

        dev_mask = ~final_test_mask
        X_test, y_test = X_full[final_test_mask], y_full[final_test_mask]
        X_dev = X_full[dev_mask].reset_index(drop=True)
        y_dev = y_full[dev_mask].reset_index(drop=True)
        dev_data = data[dev_mask].reset_index(drop=True)
        dev_window_start_dt = window_start_dt[dev_mask].reset_index(drop=True)

        # Fold over the remaining (dev) experiments only - index 1..n-1 per gas.
        n_folds = max(len(self.experiment_config[gas]) - 1 for gas in self.gases)
        fold_index_pairs = []
        for fold in range(n_folds):
            test_mask = pd.Series(False, index=dev_data.index)
            held_out = {}
            for gas in self.gases:
                n_dev_experiments = len(self.experiment_config[gas]) - 1
                exp_index = 1 + (fold % n_dev_experiments)
                experiment_name, start_dt, end_dt = self.get_experiment_window_range(gas, exp_index)
                held_out[gas] = experiment_name
                gas_mask = (dev_data['gas'] == gas) & (dev_window_start_dt >= start_dt) & \
                           (dev_window_start_dt <= end_dt)
                test_mask |= gas_mask
            print(f"Fold {fold}: held out {held_out}")
            fold_index_pairs.append((np.where(~test_mask.values)[0], np.where(test_mask.values)[0]))

        return X_dev, y_dev, fold_index_pairs, X_test, y_test

    def classifier_config(self, classifier):
        clf = None
        if classifier == "RF":
            clf = RandomForestClassifier(n_estimators=512, random_state=42, n_jobs=-1)
        elif classifier == "SVM":
            clf = SVC(kernel='linear', probability=True, max_iter=1000000)  # LinearSVC(random_state=42, max_iter=10000, probability=True) #!!!!!!!!!!!!!!!!!!!!!!!
        elif classifier == "KNN":
            clf = KNeighborsClassifier(n_neighbors=5)
        elif classifier == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=(50, 50, 25), activation='relu', solver='adam', early_stopping=True, random_state=42, learning_rate='adaptive', batch_size=32)
        elif classifier == "NB":
            clf = GaussianNB(var_smoothing=1e-9)
        elif classifier == "ETC":
            clf = ExtraTreesClassifier(n_estimators=512, random_state=42, n_jobs=-1)
        elif classifier == "HGB":
            # clf = HistGradientBoostingClassifier(max_iter=100, random_state=42, class_weight='balanced')
            clf = HistGradientBoostingClassifier(
                max_iter=100,
                random_state=42,
                #class_weight='balanced',
                max_depth=3,
                min_samples_leaf=20,
                l2_regularization=1.0,
                learning_rate=0.05,
                max_leaf_nodes=15,
            )
        elif classifier == "AutoML":
            results_path = self.resolve_config_path(self.config_paths['results_path']) / "NaiveAutoML"
            clf = load(results_path / "NaiveAutoML_best_classifier.joblib")  #untrained model
        elif classifier == "TabPFN":
            clf = TabPFNClassifier()
        elif classifier == "TabICL":
            clf = TabICLClassifier()
        return clf
        raise ValueError(f"Unknown classifier: {classifier_name}")

    def load_split(self, name, target='class', fold=None):
        splits_path = self.resolve_config_path(self.config_paths['splits_path'])
        if fold is not None:
            splits_path = splits_path / f"fold_{fold}"
        df = pd.read_csv(splits_path / f"{name}.csv")

        # gas and prestimulus make up the "class" target, so they (and the
        # other identifier columns) must be excluded from the features.
        meta_columns = ['gas', 'node', 'window_start', 'signal', 'sub_window_start', 'sub_window_end',
                         'prestimulus', 'class']
        y = df[target]
        X = df.drop(columns=[c for c in meta_columns if c in df.columns])
        return X, y

    def undersample(self, X, y, exclude_class='prestimulus', random_state=42):
        """
        Downsample every class down to the size of the largest class other
        than exclude_class (i.e. the biggest gas_post class) - not the
        smallest class overall, which would throw away far more rows than
        necessary since prestimulus is usually the biggest class by far.
        Classes already at or below that target size (including
        exclude_class itself) are left untouched. Only meant to be applied
        to the training split - val/test should keep reflecting the real,
        imbalanced class distribution so the reported scores stay
        meaningful.
        """
        counts = y.value_counts()
        other_counts = counts.drop(index=exclude_class, errors='ignore')
        target_size = other_counts.max() if not other_counts.empty else counts.min()

        idx = (
            y.to_frame('y')
            .groupby('y', group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), target_size), random_state=random_state))
            .index
        )
        idx = idx.to_series().sample(frac=1, random_state=random_state).index  # shuffle
        return X.loc[idx], y.loc[idx]

    def cap_class_size(self, X, y, target_class='prestimulus', max_size=300, random_state=42):
        """
        Randomly downsample target_class to at most max_size rows, leaving
        every other class untouched. Meant to run before undersample/SMOTE/
        ADASYN, since prestimulus rows vastly outnumber any single gas's
        post-stimulus class otherwise and would dominate/skew those steps.
        Only meant to be applied to the training split.
        """
        class_y = y[y == target_class]
        if len(class_y) > max_size:
            class_idx = class_y.sample(n=max_size, random_state=random_state).index
        else:
            class_idx = class_y.index
        keep_idx = y.index[y != target_class]
        idx = class_idx.append(keep_idx)
        idx = idx.to_series().sample(frac=1, random_state=random_state).index  # shuffle
        return X.loc[idx], y.loc[idx]

    def smote_oversample(self, X, y, random_state=42):
        """
        Oversample every class up to the size of the largest class with
        SMOTE, so training sees a balanced class distribution without
        throwing away majority-class rows the way undersampling does. Only
        meant to be applied to the training split.

        SMOTE interpolates between nearest neighbours and can't handle NaN,
        so columns are first median-imputed (fit on X only - this is
        already the training split, so no leakage) purely to make the
        synthesis possible. Columns that are entirely NaN in X have no
        median and are dropped by the imputer - the caller must apply the
        same column subset (X_res.columns) to val/test before scoring.
        """
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        retained_columns = imputer.get_feature_names_out(X.columns)
        X_imputed = pd.DataFrame(X_imputed, columns=retained_columns, index=X.index)

        smote = SMOTE(random_state=random_state)
        X_res, y_res = smote.fit_resample(X_imputed, y)
        return X_res, y_res

    def adasyn_oversample(self, X, y, random_state=42):
        """
        Oversample every class up to the size of the largest class with
        ADASYN, so training sees a balanced class distribution. Unlike
        SMOTE (which generates synthetic points uniformly across each
        minority class), ADASYN generates more synthetic points for
        minority samples that are harder to classify (close to
        majority-class neighbours) and fewer for ones that are already
        easy - only meant to be applied to the training split.

        Like SMOTE, ADASYN interpolates between nearest neighbours and
        can't handle NaN, so columns are first median-imputed (fit on X
        only - this is already the training split, so no leakage) purely
        to make the synthesis possible. Columns that are entirely NaN in X
        have no median and are dropped by the imputer - the caller must
        apply the same column subset (X_res.columns) to val/test before
        scoring.
        """
        imputer = SimpleImputer(strategy='median')
        X_imputed = imputer.fit_transform(X)
        retained_columns = imputer.get_feature_names_out(X.columns)
        X_imputed = pd.DataFrame(X_imputed, columns=retained_columns, index=X.index)

        adasyn = ADASYN(random_state=random_state)
        X_res, y_res = adasyn.fit_resample(X_imputed, y)
        return X_res, y_res

    def min_max_scaling(self, X_train, X_val, X_test):
        """
        Min-max scale all three splits, fitting the scaler on X_train only
        so val/test min/max values never leak into the training statistics.
        NaNs (tsfresh leaves some for degenerate signals) are ignored by the
        scaler and preserved in the output.
        """
        scaler = MinMaxScaler()
        X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns, index=X_val.index)
        X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        return X_train_scaled, X_val_scaled, X_test_scaled

    def impute_features(self, X_train, X_val, X_test):
        """
        Median-impute all three splits, fitting on X_train only. Several
        sklearn components (SelectKBest, GenericUnivariateSelect, ...)
        don't accept NaN the way HistGradientBoostingClassifier does, so
        this guarantees a clean matrix regardless of which component a
        downstream pipeline (e.g. naiveautoml, FeatureSelection) picks -
        instead of excluding each NaN-incompatible component one at a time.
        Columns entirely NaN in X_train have no median and are dropped by
        the imputer; val/test are transformed with the same fitted imputer
        so they end up with the same (reduced) column set.
        """
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        retained_columns = imputer.get_feature_names_out(X_train.columns)
        X_train_imputed = pd.DataFrame(X_train_imputed, columns=retained_columns, index=X_train.index)
        X_val_imputed = pd.DataFrame(imputer.transform(X_val), columns=retained_columns, index=X_val.index)
        X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=retained_columns, index=X_test.index)
        return X_train_imputed, X_val_imputed, X_test_imputed

    def robust_min_max_scaling(self, X_train, X_val, X_test, lower_quantile=0.25, upper_quantile=0.75):
        """
        Min-max scale using the interquartile range (Q1/Q3 by default)
        from X_train instead of true min/max, so a handful of outlier rows
        can't compress the rest of the data toward 0/1. Values outside
        [Q1, Q3] end up outside [0, 1] after scaling - that's expected,
        same as val/test can exceed [0, 1] with plain min-max scaling.
        Quantiles are computed on X_train only; NaNs are ignored by
        DataFrame.quantile and preserved in the output.
        """
        q_low = X_train.quantile(lower_quantile)
        q_high = X_train.quantile(upper_quantile)
        scale = (q_high - q_low)
        scale = scale.mask(scale == 0, 1)  # constant columns: avoid div-by-zero, output stays 0

        def transform(X):
            return (X - q_low) / scale

        return transform(X_train), transform(X_val), transform(X_test)

    def load_and_process_data_for_classification(self, apply_smote=False, apply_adasyn=False, scale=True,
                                                   undersample=False, impute=True, target='class', fold=None,
                                                   keep_classes=None, drop_classes=None,
                                                   gas=None):
        """
        Load train/val/test splits and package them into the
        {"train": {"X":..., "y":...}, "val": {...}, "test": {...}}
        structure auto_ml()/save_best_metrics()/FeatureSelection expect,
        applying the same scaling/resampling helpers used by
        train_classifier. Also returns the (gas, node, window_start) group
        key for each training row, for callers that need group-aware CV.

        fold=None (default) reads the flat splits_path/{train,val,test}.csv
        written by make_data_set. Pass a fold index (e.g. fold=3) to
        instead read splits_path/fold_<fold>/{train,val,test}.csv, as
        written by make_experiment_cv_folds.

        keep_classes/drop_classes optionally restrict every split to a
        subset of target classes, applied right after loading - before
        scaling/resampling see any data - e.g.
        keep_classes=['O3_post', 'prestimulus'] turns this into a binary
        gas-vs-baseline problem instead of the full multiclass one.
        keep_classes takes precedence if both are given; only one is
        normally needed.

        gas, if given (a single gas name, e.g. 'O3', or a list of names),
        further restricts every split to rows whose 'gas' column matches.
        This matters because the 'prestimulus' class label is shared across
        every gas (see _load_labeled_data) - keep_classes=['O3_post',
        'prestimulus'] alone would keep prestimulus rows from every
        experiment, including CO2/N2 ones. Pass gas='O3' alongside it to
        keep only O3 experiments' rows, i.e. O3_post vs. that same gas's own
        prestimulus rows.

        apply_smote and apply_adasyn are mutually exclusive oversampling
        options (only meant to use one at a time) - see smote_oversample /
        adasyn_oversample for what each does differently.

        impute=True (default) median-imputes all splits as the last step,
        fit on train only, so every downstream consumer gets a NaN-free
        matrix - some sklearn components (SelectKBest,
        GenericUnivariateSelect, ...) don't accept NaN natively the way
        HistGradientBoostingClassifier does. When apply_smote/apply_adasyn
        is also True, this runs after oversampling (whose internal
        imputation already cleaned train, so this call is then only doing
        real work for val/test).
        """
        X_train, y_train = self.load_split('train', target=target, fold=fold)
        X_val, y_val = self.load_split('val', target=target, fold=fold)
        X_test, y_test = self.load_split('test', target=target, fold=fold)

        splits_path = self.resolve_config_path(self.config_paths['splits_path'])
        if fold is not None:
            splits_path = splits_path / f"fold_{fold}"
        train_meta = pd.read_csv(splits_path / "train.csv")
        val_meta = pd.read_csv(splits_path / "val.csv")
        test_meta = pd.read_csv(splits_path / "test.csv")
        groups = train_meta['gas'] + '|' + train_meta['node'] + '|' + train_meta['window_start'].astype(str)

        gases_to_keep = [gas] if isinstance(gas, str) else gas

        def class_mask(y):
            if keep_classes is not None:
                return y.isin(keep_classes)
            if drop_classes is not None:
                return ~y.isin(drop_classes)
            return pd.Series(True, index=y.index)

        def gas_mask(meta):
            if gases_to_keep is None:
                return pd.Series(True, index=meta.index)
            return meta['gas'].isin(gases_to_keep)

        train_mask = class_mask(y_train) & gas_mask(train_meta)
        X_train, y_train, groups = X_train[train_mask], y_train[train_mask], groups[train_mask]
        val_mask = class_mask(y_val) & gas_mask(val_meta)
        X_val, y_val = X_val[val_mask], y_val[val_mask]
        test_mask = class_mask(y_test) & gas_mask(test_meta)
        X_test, y_test = X_test[test_mask], y_test[test_mask]

        if scale:
            X_train, X_val, X_test = self.min_max_scaling(X_train, X_val, X_test)
            #X_train, X_val, X_test =self.robust_min_max_scaling(X_train, X_val, X_test)

        non_prestim_counts = y_train[y_train != 'prestimulus'].value_counts()
        prestim_cap = int(non_prestim_counts.max()) if not non_prestim_counts.empty else len(y_train)
        X_train, y_train = self.cap_class_size(X_train, y_train, target_class='prestimulus', max_size=prestim_cap)

        if undersample:
            X_train, y_train = self.undersample(X_train, y_train)

        if apply_smote:
            X_train, y_train = self.smote_oversample(X_train, y_train)
            X_val = X_val[X_train.columns]
            X_test = X_test[X_train.columns]
        elif apply_adasyn:
            X_train, y_train = self.adasyn_oversample(X_train, y_train)
            X_val = X_val[X_train.columns]
            X_test = X_test[X_train.columns]

        if impute:
            X_train, X_val, X_test = self.impute_features(X_train, X_val, X_test)

        data = {
            'train': {'X': X_train, 'y': y_train},
            'val': {'X': X_val, 'y': y_val},
            'test': {'X': X_test, 'y': y_test},
        }
        return data, groups

    def save_best_metrics(self, clf, data, feature_subset=None):
        """
        Score `clf` on val/train/test, write a combined classification
        report + confusion matrices to results_path/<classifier_name>/, and
        dump the fitted classifier there too.
        """
        results_path = self.resolve_config_path(self.config_paths['results_path']) / self.classifier_name
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
        results_path = self.resolve_config_path(self.config_paths['results_path']) / self.classifier_name
        results_path.mkdir(parents=True, exist_ok=True)

        fold_index_pairs = None
        X_test = y_test = None
        if use_experiment_folds:
            X_all, y_all, fold_index_pairs, X_test, y_test = self._build_experiment_fold_indices(target=target)
            data_init = {'train': {'X': X_all, 'y': y_all}}
        else:
            data_init, groups = self.load_and_process_data_for_classification(
                apply_smote=smote, apply_adasyn=adasyn, scale=True, undersample=undersample
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
                          undersample=False, smote=True, adasyn=False, fold=0, feature_subset_path="/home/wp/Documents/GitHub/DataProcessing/GasExperiment/03_results/multivariate_ranked_features.csv",
                          feature_column=None, n_features=100):
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
            splits_path = self.resolve_config_path(self.config_paths['splits_path'])
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
                    figures_path = self.resolve_config_path(self.config_paths['figures_path'])
                    figures_path.mkdir(parents=True, exist_ok=True)
                    out = figures_path / f"{classifier_name}_all_folds_{name}_confusion_matrix.png"
                    fig.savefig(out, dpi=150)
                    print(f"Saved {out}")

                if show:
                    plt.show()
                plt.close(fig)

            if save:
                results_path = self.resolve_config_path(self.config_paths['results_path'])
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
        data_init, groups = self.load_and_process_data_for_classification(
            apply_smote=smote, apply_adasyn=adasyn, scale=True, undersample=undersample, target=target, fold=fold
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

        clf = self.classifier_config(classifier_name)
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
                figures_path = self.resolve_config_path(self.config_paths['figures_path'])
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
        data_init, groups = self.load_and_process_data_for_classification(
            apply_smote=False, scale=True, undersample=False, target=target, fold=fold
        )

        results_path = self.resolve_config_path(self.config_paths['results_path'])
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
                clf = self.classifier_config(classifier_name)
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
        clf = self.classifier_config(classifier_name)
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
        results_path = self.resolve_config_path(self.config_paths['results_path'])
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
            figures_path = self.resolve_config_path(self.config_paths['figures_path'])
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
    #GC.make_experiment_cv_folds()
    #GC.make_data_set()
    for classifier in ["TabPFN"]:#["AutoML"]:#"HGB", "RF", "ETC"]:TabICL
        GC.train_classifier(classifier, feature_column="cmim")
    #GC.auto_ml(train=True, save=True)
    #GC.train_classifier_feature_subset()
    #GC.compute_feature_subset_accuracy(use_majority_rank_aggregation=False, max_features=200, save=True)
    #GC.compute_feature_subset_accuracy(ranked_features_path= "03_results/multivariate_ranked_features.csv", use_majority_rank_aggregation=False, max_features=200, save=True, )
    #GC.plot_feature_subset_accuracy(classifier_name="TabPFN",metric="accuracy")
    #data_init, groups = GC.load_and_process_data_for_classification(apply_smote=False, scale=True)
    fs = FeatureSelection()
    #fs.apply_feature_selection(data_init, groups, save=True)
    fs.aggregate_features(majority_voting=True, rank_aggregation=True, use_mrmr=True)
    #fs.apply_mrmr(data_init, None, save=True)
    # fs.apply_multivariate_feature_selection(data_init,k=10000,save=True)
