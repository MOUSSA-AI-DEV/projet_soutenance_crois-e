import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline


def optimize_models(preprocessor, X_train, y_train, random_state=42):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    scoring = "f1_macro"

    pipe = {
        "Logistique_Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, random_state=random_state)),
        ]),
        "KNN": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", KNeighborsClassifier(n_neighbors=5)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=random_state)),
        ]),
    }

    param_grid = {
        "Logistique_Regression": {
            "classifier__C": [0.1, 1, 10]
        },
        "KNN": {
            "classifier__n_neighbors": [3, 5, 7, 9, 11],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        },
        "Random Forest": {
            "classifier__n_estimators": [50, 100, 200],
            "classifier__max_depth": [None, 10, 20],
            "classifier__min_samples_split": [2, 5],
        },
    }

    best_models = {}
    optimization_results = []

    for name, model_pipe in pipe.items():
        grid_search = GridSearchCV(
            estimator=model_pipe,
            param_grid=param_grid[name],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
        )
        grid_search.fit(X_train, y_train)
        best_models[name] = grid_search.best_estimator_

        optimization_results.append({
            "Model": name,
            "Best Parameters": grid_search.best_params_,
            "Best Score": grid_search.best_score_,
        })

    optimization_df = pd.DataFrame(optimization_results).sort_values(
        by="Best Score", ascending=False
    )
    best_model_name = optimization_df.iloc[0]["Model"]
    best_model = best_models[best_model_name]

    return optimization_df, best_models, best_model_name, best_model

