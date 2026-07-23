import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from skfeature.function.similarity_based.reliefF import reliefF
import utils


def relief_score(X, y):
    """
    SelectKBest-compatible wrapper around skfeature's reliefF: it defaults
    to returning a rank-index array (mode="rank"), but SelectKBest expects
    a raw per-feature score array (mode="raw") to fill its .scores_.
    Nearest-neighbor based, so - unlike chi2 - it doesn't assume
    non-negative/categorical features, and picks up feature interactions
    chi2/anova/mutual_info miss.
    """
    return reliefF(np.asarray(X), np.asarray(y), mode="raw")


class FeatureSelection:
    def __init__(self, experiments_file=None):
        self.experiments_file = experiments_file or Path(__file__).with_name("gas_experiments.json")
        self.base_dir = Path(__file__).resolve().parent
        self.PNs = ['P1','P3']
        self.gases = ["CO2", "N2", "O3"]
        self.classifier_name = None
        with open(self.base_dir / "config.json", "r") as file:
            self.config = json.load(file)
        self.config_paths = self.config['paths']

        self.selectors = {
            "mutual_info": mutual_info_classif,
            "anova": f_classif,
            "relief": relief_score,
        }


    def resolve_config_path(self, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.base_dir / path / 'raw'

    def feature_selection_results_path(self):
        """
        results_path/03_01_feature_selection - every ranked-features file
        this class reads/writes lives under here, instead of directly in
        results_path, so feature-selection output is grouped separately
        from classifier results.
        """
        path = self.resolve_config_path(self.config_paths['results_path']) / "03_01_feature_selection"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def apply_univariate_feature_selection(self, data, groups, save, keep_classes=None, drop_classes=None, gas=None):
        """
        keep_classes/drop_classes/gas should be the same scope `data` was
        already restricted to upstream (e.g. by
        utils.load_and_process_data_for_classification) - they don't
        filter anything here, they're only used (via utils.scope_suffix)
        to name the saved file so different scopes don't overwrite each
        other's ranking.
        """
        X_train = data["train"]["X"]
        y_train = data["train"]["y"]


        ranked_columns = {}
        for name, score_func in self.selectors.items():
            selector = SelectKBest(score_func, k=len(X_train.columns))
            selector.fit(X_train, y_train)
            ranked = pd.Series(selector.scores_, index=X_train.columns).sort_values(ascending=False)
            ranked_columns[name] = ranked.index.to_list()
            ranked_columns[f"{name}_score"] = ranked.to_list()

        ranked_df = pd.DataFrame(ranked_columns)
        ranked_df.index = range(1, len(ranked_df) + 1)
        ranked_df.index.name = "rank"
        print(ranked_df)

        if save:
            results_path = self.feature_selection_results_path()
            suffix = utils.scope_suffix(gas, keep_classes, drop_classes)
            out = results_path / f"univariate_ranked_features{suffix}.csv"
            ranked_df.to_csv(out)
            print(f"Saved {out}")

        return ranked_df
    
    def aggregate_features(self, save=True, keep_classes=None, drop_classes=None, gas=None):
        """
        Combine every filter method's ranking - univariate (mutual_info,
        anova, relief, from apply_univariate_feature_selection) and
        multivariate (mrmr_mean, mrmr_max, from
        apply_multivariate_feature_selection) - into one aggregated
        ranking, three different ways to condense each feature's ranks
        across methods into a single number:
        - mean: average rank across methods.
        - median: middle rank across methods - less sensitive than mean to
          one method ranking a feature unusually high/low.
        - product: rank product - a feature consistently near the top
          across every method gets a low product; one bad rank from a
          single method can't be offset by the others the way it can with
          mean/median (product grows fast with any large rank).
        Each is sorted ascending (lower = better) into its own column of
        one output file.

        Multivariate rankings only cover their top-k selected features
        (unlike univariate's full ranking of every feature) - a feature
        missing from a given method's list is NaN for that method and
        skipped (not penalized with a worst-case rank) when computing its
        mean/median/product, i.e. its aggregate rank reflects only the
        methods that actually ranked it.

        keep_classes/drop_classes/gas must match whatever scope
        apply_univariate_feature_selection/apply_multivariate_feature_selection
        were called with, so the right (scope-suffixed) source files are
        found.
        """
        results_path = self.feature_selection_results_path()
        suffix = utils.scope_suffix(gas, keep_classes, drop_classes)

        univariate = pd.read_csv(results_path / f"univariate_ranked_features{suffix}.csv", index_col=0)
        # apply_univariate_feature_selection also saves a "<test>_score"
        # column next to each "<test>" feature-name column - only the
        # feature-name columns are rank->feature mappings.
        univariate = univariate[[c for c in univariate.columns if not c.endswith('_score')]]
        multivariate = pd.read_csv(results_path / f"multivariate_ranked_features{suffix}.csv", index_col=0)

        # invert rank->feature (per method) into feature->rank, across
        # every univariate and multivariate method at once.
        rank_lookup = pd.DataFrame({
            method: pd.Series(df.index, index=df[method])
            for df in (univariate, multivariate)
            for method in df.columns
        })

        aggregated = pd.DataFrame({
            "mean": rank_lookup.mean(axis=1, skipna=True).sort_values().index,
            "median": rank_lookup.median(axis=1, skipna=True).sort_values().index,
            "product": rank_lookup.prod(axis=1, skipna=True).sort_values().index,
        })
        aggregated.index = range(1, len(aggregated) + 1)
        aggregated.index.name = "rank"
        print(aggregated)

        if save:
            out = results_path / f"aggregated_ranked_features{suffix}.csv"
            aggregated.to_csv(out)
            print(f"Saved {out}")

        return aggregated

    def _relevance(self, X_train, y_train, score_func):
        """
        Relevance scores from `score_func` (any SelectKBest-compatible
        scorer - mutual_info_classif, f_classif, relief_score, ...),
        min-max normalized to [0, 1]. F statistics are unbounded and often
        huge, while redundancy (abs correlation, used downstream) is capped
        at 1 - without this normalization, relevance would completely
        dominate any (relevance - redundancy) combination regardless of
        which score_func is used. +-inf (e.g. a near-zero-variance-within-
        class ANOVA blowup) is clipped to the max/min finite score first so
        it doesn't wreck the normalization.
        """
        # SelectKBest normalizes the scorer's output (some, like f_classif,
        # return a (scores, p-values) tuple) into a flat .scores_ array
        # regardless of which score_func is used.
        relevance = pd.Series(
            SelectKBest(score_func, k='all').fit(X_train, y_train).scores_, index=X_train.columns
        )
        finite = relevance.replace([np.inf, -np.inf], np.nan)
        relevance = relevance.clip(lower=finite.min(), upper=finite.max())
        return (relevance - relevance.min()) / (relevance.max() - relevance.min())

    def _mrmr_greedy(self, X_train, relevance, k, redundancy_agg='mean'):
        """
        Core greedy mRMR loop: picks the feature that maximizes (relevance
        - redundancy with the features already selected), where redundancy
        is the absolute Pearson correlation between features.

        redundancy_agg controls how redundancy against the *whole* selected
        set is condensed into one number per candidate:
        - 'mean' (default, classic mRMR/MID): average correlation against
          all selected features. Weakness: once the selected set is large
          and diverse, one or two near-duplicates of a candidate get
          diluted into the average, under-penalizing it.
        - 'max': correlation against the single most-correlated already
          selected feature. Being highly correlated with even one existing
          pick is penalized regardless of how diverse the rest of the
          selected set is - stricter about near-duplicates.

        Returns the plain list of selected feature names, no I/O.
        """
        # fillna(0) as a safety net for NaN correlations (e.g. a feature
        # that is constant only within one class) - treating an undefined
        # correlation as "no redundancy" rather than crashing.
        redundancy = X_train.corr().abs().fillna(0)

        selected = [relevance.idxmax()]
        remaining = [c for c in X_train.columns if c not in selected]

        for _ in range(min(k, len(X_train.columns)) - 1):
            if redundancy_agg == 'max':
                candidate_redundancy = redundancy.loc[remaining, selected].max(axis=1)
            else:
                candidate_redundancy = redundancy.loc[remaining, selected].mean(axis=1)
            mrmr_score = relevance[remaining] - candidate_redundancy
            next_feature = mrmr_score.idxmax()
            selected.append(next_feature)
            remaining.remove(next_feature)

        return selected

    def _mrmr_select(self, X_train, y_train, score_func, k, redundancy_agg='mean'):
        """
        Relevance (from a single score_func) + greedy mRMR selection.
        Returns the plain list of selected feature names, no I/O.
        """
        # Zero-variance columns carry no signal and make Pearson
        # correlation undefined (NaN) against every other feature, which
        # can eventually starve the greedy loop of any valid candidate
        # once k gets large enough to reach them - drop them up front.
        X_train = X_train.loc[:, X_train.std() > 0]
        relevance = self._relevance(X_train, y_train, score_func)
        return self._mrmr_greedy(X_train, relevance, k, redundancy_agg=redundancy_agg)

    def mrmr(self, data, score_func=mutual_info_classif, k=40, redundancy_agg='mean',
             keep_classes=None, drop_classes=None, gas=None):
        """
        Minimum Redundancy Maximum Relevance, using a single relevance
        measure (mutual information by default). Unlike the univariate
        tests in apply_feature_selection, this actively avoids picking
        near-duplicate features. Prints and saves its own ranked feature
        list to mrmr_features.csv. See _mrmr_greedy for redundancy_agg.

        keep_classes/drop_classes/gas should match the scope `data` was
        already restricted to upstream - only used (via utils.scope_suffix)
        to name the saved file.
        """
        selected = self._mrmr_select(data["train"]["X"], data["train"]["y"], score_func, k,
                                      redundancy_agg=redundancy_agg)

        mrmr_df = pd.DataFrame({"feature": selected})
        mrmr_df.index = range(1, len(mrmr_df) + 1)
        mrmr_df.index.name = "rank"
        print(mrmr_df)

        results_path = self.feature_selection_results_path()
        suffix = utils.scope_suffix(gas, keep_classes, drop_classes)
        out = results_path / f"mrmr_features{suffix}.csv"
        mrmr_df.to_csv(out)
        print(f"Saved {out}")

        return mrmr_df

    def apply_mrmr(self, data, k=40, save=True, redundancy_agg='mean',
                    keep_classes=None, drop_classes=None, gas=None):
        """
        Run the mRMR selection once per univariate test (mutual_info,
        anova, relief), so the redundancy-aware rankings can be
        compared/aggregated the same way apply_feature_selection's
        pure-relevance rankings are - kept in their own separate file
        (mrmr_ranked_features.csv) rather than merged into
        ranked_features.csv. See _mrmr_greedy for redundancy_agg.

        keep_classes/drop_classes/gas should match the scope `data` was
        already restricted to upstream - only used (via utils.scope_suffix)
        to name the saved file.
        """
        X_train, y_train = data["train"]["X"], data["train"]["y"]
        tests = {
            "mutual_info": mutual_info_classif,
            "anova": f_classif,
            "relief": relief_score,
        }

        if k == None:
            k = len(X_train.columns)
        ranked_columns = {name: self._mrmr_select(X_train, y_train, score_func, k, redundancy_agg=redundancy_agg)
                           for name, score_func in tests.items()}

        mrmr_df = pd.DataFrame(ranked_columns)
        mrmr_df.index = range(1, len(mrmr_df) + 1)
        mrmr_df.index.name = "rank"
        print(mrmr_df)

        if save:
            results_path = self.feature_selection_results_path()
            suffix = utils.scope_suffix(gas, keep_classes, drop_classes)
            out = results_path / f"mrmr_ranked_features{suffix}.csv"
            mrmr_df.to_csv(out)
            print(f"Saved {out}")

        return mrmr_df

    def apply_multivariate_feature_selection(self, data, k=40, save=True,
                                              keep_classes=None, drop_classes=None, gas=None):
        """
        Run the available multivariate (redundancy-aware) selection
        methods side by side and save their rankings to one file, the same
        way apply_feature_selection does for the univariate tests
        (mutual_info/anova/relief) - but here every column already accounts
        for feature-feature redundancy, not just relevance to the target:
        - mrmr_mean: mutual_info relevance, mean-redundancy mRMR
        - mrmr_max: mutual_info relevance, max-redundancy mRMR (stricter
          about near-duplicates - see _mrmr_greedy)

        JMI/CMIM (skfeature's information-theoretic selectors) used to be
        included here too, but their discrete mutual-information estimators
        aren't valid on tsfresh's continuous features - dropped rather than
        discretized/replaced.

        keep_classes/drop_classes/gas should match the scope `data` was
        already restricted to upstream - only used (via utils.scope_suffix)
        to name the saved file.
        """
        X_train, y_train = data["train"]["X"], data["train"]["y"]

        methods = {
            "mrmr_mean": lambda: self._mrmr_select(X_train, y_train, mutual_info_classif, k, redundancy_agg='mean'),
            "mrmr_max": lambda: self._mrmr_select(X_train, y_train, mutual_info_classif, k, redundancy_agg='max'),
        }

        ranked_columns = {name: method() for name, method in methods.items()}
        ranked_df = pd.DataFrame(ranked_columns)
        ranked_df.index = range(1, len(ranked_df) + 1)
        ranked_df.index.name = "rank"
        print(ranked_df)

        if save:
            results_path = self.feature_selection_results_path()
            suffix = utils.scope_suffix(gas, keep_classes, drop_classes)
            out = results_path / f"multivariate_ranked_features{suffix}.csv"
            ranked_df.to_csv(out)
            print(f"Saved {out}")

        return ranked_df





