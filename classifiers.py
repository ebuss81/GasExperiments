from joblib import load
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from tabpfn import TabPFNClassifier
from tabicl import TabICLClassifier


def classifier_config(classifier, automl_results_path=None):
    """
    Build an untrained classifier instance by name. automl_results_path is
    only used for classifier == "AutoML", which instead loads the (also
    untrained) best pipeline naiveautoml's search picked, from
    <automl_results_path>/NaiveAutoML_best_classifier.joblib.
    """
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
        clf = load(automl_results_path / "NaiveAutoML_best_classifier.joblib")  #untrained model
    elif classifier == "TabPFN":
        clf = TabPFNClassifier()
    elif classifier == "TabICL":
        clf = TabICLClassifier()
    return clf
    raise ValueError(f"Unknown classifier: {classifier_name}")
