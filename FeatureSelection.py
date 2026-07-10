import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    chi2,
    mutual_info_classif,
    mutual_info_regression
)
from skfeature.function.information_theoretical_based import JMI, CMIM



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
            "chi2": chi2,
        }


    def resolve_config_path(self, path_value):
        path = Path(path_value).expanduser()
        if path.is_absolute():
            return path
        return self.base_dir / path

    def apply_feature_selection(self, data, groups, save):
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
            results_path = self.resolve_config_path(self.config_paths['results_path'])
            results_path.mkdir(parents=True, exist_ok=True)
            out = results_path / "univariate_ranked_features.csv"
            ranked_df.to_csv(out)
            print(f"Saved {out}")

        return ranked_df
    
    def aggregate_features(self, majority_voting=True, rank_aggregation=False, use_mrmr=False):
        """
        use_mrmr=False (default) aggregates the plain univariate rankings
        from apply_feature_selection (ranked_features.csv). use_mrmr=True
        aggregates the redundancy-aware rankings from apply_mrmr
        (mrmr_ranked_features.csv) instead. Output files are named
        accordingly so the two don't overwrite each other.
        """
        results_path = self.resolve_config_path(self.config_paths['results_path'])
        prefix = "mrmr_" if use_mrmr else ""
        source = results_path / f"{prefix}ranked_features.csv"
        ranked_features = pd.read_csv(source, index_col=0)
        # apply_feature_selection also saves a "<test>_score" column next
        # to each "<test>" feature-name column - only the feature-name
        # columns are rank->feature mappings, so drop the score columns
        # before aggregating.
        ranked_features = ranked_features[[c for c in ranked_features.columns if not c.endswith('_score')]]

        if majority_voting:
            n = 40
            top_n = ranked_features.loc[1:n]
            occurrences = pd.Series(top_n.values.ravel()).value_counts()

            occurrences = occurrences[occurrences > 1]

            for idx, (feature, count) in enumerate(occurrences.items()):
                print(f"{idx}, {feature}: {count}")

            out = results_path / f"{prefix}majority_voting_features.csv"
            occurrences.rename("count").rename_axis("feature").to_csv(out)
            print(f"Saved {out}")

        if rank_aggregation:
            # invert rank->feature (per test) into feature->rank, then average across tests
            rank_lookup = pd.DataFrame({
                test: pd.Series(ranked_features.index, index=ranked_features[test])
                for test in ranked_features.columns
            })
            avg_rank = rank_lookup.mean(axis=1).sort_values()
            avg_rank.index.name = "feature"
            avg_rank.name = "avg_rank"

            for idx, (feature, rank) in enumerate(avg_rank.items()):
                print(f"{idx}, {feature}: {rank:.2f}")

            out = results_path / f"{prefix}rank_aggregation_features.csv"
            avg_rank.to_csv(out)
            print(f"Saved {out}")

    def combine_majority_rank_aggregation(self, use_mrmr=False, save=True):
        """
        majority_voting_features.csv (feature, count) and
        rank_aggregation_features.csv (feature, avg_rank) are both
        single-approach outputs of aggregate_features with different
        shapes/schemas, so they can't be fed into e.g.
        Gas_Classification.train_classifier_feature_subset directly.
        This combines them into one ranked-features file with a
        "majority_voting" and a "rank_aggregation" column (rank index +
        ordered feature names), so the two can be compared on one
        accuracy-vs-features plot the same way any other ranked-features
        file is. majority_voting is typically much shorter (only features
        that >1 test agreed on) - it's padded with NaN past that point.
        """
        results_path = self.resolve_config_path(self.config_paths['results_path'])
        prefix = "mrmr_" if use_mrmr else ""

        majority = pd.read_csv(results_path / f"{prefix}majority_voting_features.csv")
        rank_agg = pd.read_csv(results_path / f"{prefix}rank_aggregation_features.csv")

        combined = pd.DataFrame({
            "majority_voting": majority["feature"],
            "rank_aggregation": rank_agg["feature"],
        })
        combined.index = range(1, len(combined) + 1)
        combined.index.name = "rank"
        print(combined)

        if save:
            out = results_path / f"{prefix}majority_rank_ranked_features.csv"
            combined.to_csv(out)
            print(f"Saved {out}")

        return combined

    def _relevance(self, X_train, y_train, score_func):
        """
        Relevance scores from `score_func` (any sklearn univariate scorer -
        mutual_info_classif, f_classif, chi2, ...), min-max normalized to
        [0, 1]. F/chi2 statistics are unbounded and often huge, while
        redundancy (abs correlation, used downstream) is capped at 1 -
        without this normalization, relevance would completely dominate any
        (relevance - redundancy) combination regardless of which score_func
        is used. +-inf (e.g. a near-zero-variance-within-class ANOVA
        blowup) is clipped to the max/min finite score first so it doesn't
        wreck the normalization.
        """
        # SelectKBest normalizes the scorer's output (some, like f_classif
        # and chi2, return a (scores, p-values) tuple) into a flat
        # .scores_ array regardless of which score_func is used.
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

    def mrmr(self, data, score_func=mutual_info_classif, k=40, redundancy_agg='mean'):
        """
        Minimum Redundancy Maximum Relevance, using a single relevance
        measure (mutual information by default). Unlike the univariate
        tests in apply_feature_selection, this actively avoids picking
        near-duplicate features. Prints and saves its own ranked feature
        list to mrmr_features.csv. See _mrmr_greedy for redundancy_agg.
        """
        selected = self._mrmr_select(data["train"]["X"], data["train"]["y"], score_func, k,
                                      redundancy_agg=redundancy_agg)

        mrmr_df = pd.DataFrame({"feature": selected})
        mrmr_df.index = range(1, len(mrmr_df) + 1)
        mrmr_df.index.name = "rank"
        print(mrmr_df)

        results_path = self.resolve_config_path(self.config_paths['results_path'])
        results_path.mkdir(parents=True, exist_ok=True)
        out = results_path / "mrmr_features.csv"
        mrmr_df.to_csv(out)
        print(f"Saved {out}")

        return mrmr_df

    def apply_mrmr(self, data, k=40, save=True, redundancy_agg='mean'):
        """
        Run the mRMR selection once per univariate test (mutual_info,
        anova, chi2), so the redundancy-aware rankings can be
        compared/aggregated the same way apply_feature_selection's
        pure-relevance rankings are - kept in their own separate file
        (mrmr_ranked_features.csv) rather than merged into
        ranked_features.csv. See _mrmr_greedy for redundancy_agg.
        """
        X_train, y_train = data["train"]["X"], data["train"]["y"]
        tests = {
            "mutual_info": mutual_info_classif,
            "anova": f_classif,
            "chi2": chi2,
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
            results_path = self.resolve_config_path(self.config_paths['results_path'])
            results_path.mkdir(parents=True, exist_ok=True)
            out = results_path / "mrmr_ranked_features.csv"
            mrmr_df.to_csv(out)
            print(f"Saved {out}")

        return mrmr_df

    def ccombined_mrmr(self, data, k=40, save=True, redundancy_agg='mean'):
        """
        Average the (normalized) relevance scores from mutual_info, anova
        and chi2 into one combined relevance score per feature, then run
        the greedy mRMR loop once on that combined score.

        This is the "do it right" alternative to apply_mrmr +
        aggregate_features(use_mrmr=True): running mRMR three times (once
        per test) and then averaging ranks only removes redundancy within
        each individual test's ranking - each test can still independently
        pick a different, mutually-correlated feature (e.g. a different
        change_quantiles variant each), so the final averaged list can end
        up redundant again. Resolving redundancy once, on the combined
        relevance, avoids that reintroduction entirely. See _mrmr_greedy
        for redundancy_agg.
        """
        X_train, y_train = data["train"]["X"], data["train"]["y"]
        X_train = X_train.loc[:, X_train.std() > 0]

        tests = {
            "mutual_info": mutual_info_classif,
            "anova": f_classif,
            "chi2": chi2,
        }
        relevance = pd.concat(
            [self._relevance(X_train, y_train, score_func) for score_func in tests.values()], axis=1
        ).mean(axis=1)

        if k is None:
            k = len(X_train.columns)
        selected = self._mrmr_greedy(X_train, relevance, k, redundancy_agg=redundancy_agg)

        combined_df = pd.DataFrame({"feature": selected})
        combined_df.index = range(1, len(combined_df) + 1)
        combined_df.index.name = "rank"
        print(combined_df)

        if save:
            results_path = self.resolve_config_path(self.config_paths['results_path'])
            results_path.mkdir(parents=True, exist_ok=True)
            out = results_path / "combined_mrmr_features.csv"
            combined_df.to_csv(out)
            print(f"Saved {out}")

        return combined_df

    def _infotheoretic_select(self, X_train, y_train, algo, k):
        """
        Run a skfeature information-theoretic multivariate selector (JMI,
        CMIM, ...) and return the plain list of selected feature names.
        Unlike mRMR's correlation-based redundancy proxy, these score
        candidate features by their joint/conditional mutual information
        with already-selected features and the target, so they can credit
        genuine feature interactions - at the cost of being much slower
        (tens of seconds even for a few dozen features).
        """
        X_train = X_train.loc[:, X_train.std() > 0]
        y_codes = y_train.astype('category').cat.codes.values
        idx = algo(X_train.values, y_codes, mode='index', n_selected_features=min(k, len(X_train.columns)))
        return X_train.columns[idx].tolist()

    def apply_multivariate_feature_selection(self, data, k=40, save=True):
        """
        Run the available multivariate (redundancy-aware) selection
        methods side by side and save their rankings to one file, the same
        way apply_feature_selection does for the univariate tests
        (mutual_info/anova/chi2) - but here every column already accounts
        for feature-feature redundancy, not just relevance to the target:
        - mrmr_mean: mutual_info relevance, mean-redundancy mRMR
        - mrmr_max: mutual_info relevance, max-redundancy mRMR (stricter
          about near-duplicates - see _mrmr_greedy)
        - combined_mrmr: relevance averaged across mutual_info/anova/chi2
          first, then a single mRMR pass (mean-redundancy)
        - jmi / cmim: information-theoretic selectors (skfeature) that can
          credit feature interactions, not just pairwise redundancy - much
          slower than the mRMR variants
        """
        X_train, y_train = data["train"]["X"], data["train"]["y"]

        methods = {
            "mrmr_mean": lambda: self._mrmr_select(X_train, y_train, mutual_info_classif, k, redundancy_agg='mean'),
            "mrmr_max": lambda: self._mrmr_select(X_train, y_train, mutual_info_classif, k, redundancy_agg='max'),
            "combined_mrmr": lambda: self.combined_mrmr(data, k=k, save=False)["feature"].to_list(),
            "jmi": lambda: self._infotheoretic_select(X_train, y_train, JMI.jmi, k),
            "cmim": lambda: self._infotheoretic_select(X_train, y_train, CMIM.cmim, k),
        }

        ranked_columns = {name: method() for name, method in methods.items()}
        ranked_df = pd.DataFrame(ranked_columns)
        ranked_df.index = range(1, len(ranked_df) + 1)
        ranked_df.index.name = "rank"
        print(ranked_df)

        if save:
            results_path = self.resolve_config_path(self.config_paths['results_path'])
            results_path.mkdir(parents=True, exist_ok=True)
            out = results_path / "multivariate_ranked_features.csv"
            ranked_df.to_csv(out)
            print(f"Saved {out}")

        return ranked_df





