import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE, ADASYN


def undersample(X, y, exclude_class='prestimulus', random_state=42):
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
    other_counts = counts.drop(index=exclude_class, errors='ignoref')
    target_size = other_counts.max() if not other_counts.empty else counts.min()

    idx = (
        y.to_frame('y')
        .groupby('y', group_keys=False)
        .apply(lambda g: g.sample(n=min(len(g), target_size), random_state=random_state))
        .index
    )
    idx = idx.to_series().sample(frac=1, random_state=random_state).index  # shuffle
    return X.loc[idx], y.loc[idx]


def cap_class_size(X, y, target_class='prestimulus', max_size=300, random_state=42):
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


def smote_oversample(X, y, random_state=42):
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


def adasyn_oversample(X, y, random_state=42):
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


def min_max_scaling(X_train, X_val, X_test):
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


def impute_features(X_train, X_val, X_test):
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


def robust_min_max_scaling(X_train, X_val, X_test, lower_quantile=0.25, upper_quantile=0.75):
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


def load_and_process_data_for_classification(folds, apply_smote=False, apply_adasyn=False, scale=True,
                                               apply_undersample=False, impute=True, target='class', fold=None,
                                               keep_classes=['O3_post', 'prestimulus'] , drop_classes=None,
                                               gas='O3'):
    """
    Load train/val/test splits (via folds, an ExperimentFolds instance)
    and package them into the {"train": {"X":..., "y":...}, "val": {...},
    "test": {...}} structure auto_ml()/save_best_metrics()/FeatureSelection
    expect, applying the same scaling/resampling helpers used by
    train_classifier. Also returns the (gas, node, window_start) group
    key for each training row, for callers that need group-aware CV.

    fold=None (default) reads the flat splits_path/{train,val,test}.csv
    written by folds.make_data_set. Pass a fold index (e.g. fold=3) to
    instead read splits_path/fold_<fold>/{train,val,test}.csv, as
    written by folds.make_experiment_cv_folds.

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
    every gas (see ExperimentFolds._load_labeled_data) - keep_classes=
    ['O3_post', 'prestimulus'] alone would keep prestimulus rows from
    every experiment, including CO2/N2 ones. Pass gas='O3' alongside it
    to keep only O3 experiments' rows, i.e. O3_post vs. that same gas's
    own prestimulus rows.

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
    X_train, y_train = folds.load_split('train', target=target, fold=fold)
    X_val, y_val = folds.load_split('val', target=target, fold=fold)
    X_test, y_test = folds.load_split('test', target=target, fold=fold)

    splits_path = folds.resolve_config_path(folds.config_paths['splits_path'])
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
        X_train, X_val, X_test = min_max_scaling(X_train, X_val, X_test)
        #X_train, X_val, X_test = robust_min_max_scaling(X_train, X_val, X_test)

    non_prestim_counts = y_train[y_train != 'prestimulus'].value_counts()
    prestim_cap = int(non_prestim_counts.max()) if not non_prestim_counts.empty else len(y_train)
    X_train, y_train = cap_class_size(X_train, y_train, target_class='prestimulus', max_size=prestim_cap)

    if apply_undersample:
        X_train, y_train = undersample(X_train, y_train)

    if apply_smote:
        X_train, y_train = smote_oversample(X_train, y_train)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]
    elif apply_adasyn:
        X_train, y_train = adasyn_oversample(X_train, y_train)
        X_val = X_val[X_train.columns]
        X_test = X_test[X_train.columns]

    if impute:
        X_train, X_val, X_test = impute_features(X_train, X_val, X_test)

    data = {
        'train': {'X': X_train, 'y': y_train},
        'val': {'X': X_val, 'y': y_val},
        'test': {'X': X_test, 'y': y_test},
    }
    return data, groups
