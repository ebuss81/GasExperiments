import json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.impute import SimpleImputer


class ExperimentFolds:
    """
    Experiment/window bookkeeping and the train/val/test and
    leave-one-experiment-out-per-gas CV fold construction used by
    GasClassification - turns the per-gas 10-minute feature tables
    (written by Preprocess.get_10min_calc_features) into split/fold CSVs.
    """
    def __init__(self, experiments_file=None):
        self.experiments_file = experiments_file or Path(__file__).with_name("gas_experiments.json")
        self.base_dir = Path(__file__).resolve().parent
        self.PNs = ['P1', 'P3']
        self.gases = ["CO2", "N2", "O3"]
        self.read_experiment_config()
        with open(self.base_dir / "config.json", "r") as file:
            self.config = json.load(file)
        self.config_paths = self.config['paths']

    def resolve_config_path(self, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.base_dir / path / 'raw'#'withBackgroundSubstraction'

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

    def _build_experiment_fold_indices(self, target='class', keep_classes=None, drop_classes=None, gas=None):
        """
        Reserve the first experiment listed for each gas (index 0 - the same
        experiments make_data_set holds out as its final test set) as a
        completely untouched final test set, never used by the AutoML
        search. The *remaining* experiments (index 1..n-1 per gas) form the
        dev pool, which is split into leave-one-experiment-out-per-gas folds
        for naiveautoml's custom evaluator to use during model selection.

        keep_classes/drop_classes/gas restrict the classification problem
        the same way they do in utils.load_and_process_data_for_classification
        (e.g. keep_classes=['O3_post', 'prestimulus'], gas='O3' for a binary
        O3-vs-baseline problem) - applied once, right after loading, so
        every dev fold *and* the final held-out test set only ever see rows
        inside that scope; excluded gases simply contribute no rows to any
        fold's held-out mask below.

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

        gases_to_keep = [gas] if isinstance(gas, str) else gas
        if keep_classes is not None:
            scope_mask = data[target].isin(keep_classes)
        elif drop_classes is not None:
            scope_mask = ~data[target].isin(drop_classes)
        else:
            scope_mask = pd.Series(True, index=data.index)
        if gases_to_keep is not None:
            scope_mask &= data['gas'].isin(gases_to_keep)
        data = data[scope_mask].reset_index(drop=True)

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
