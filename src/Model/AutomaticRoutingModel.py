import json
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import timedelta

from scipy.stats import zscore
from sklearn.feature_selection import RFECV, SelectKBest, chi2, mutual_info_classif
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, \
    roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


class AutomaticRoutingModel:
    """
    class for routing a new transaction to a best PSP for processing
    """

    def __init__(self, file_path, fee_structure):
        self.features = {}
        self.best_model_features = None
        self.data_path = file_path
        self.fee_structure = fee_structure
        self.df = None
        self.df_grouped = None
        self.preprocessor = None
        self.models = {}
        self.model_accuracies = {}
        self.best_model = None
        self._model = None

    def load_and_preprocess_data(self):
        """
        Load the existing data from the file as dataframe, detect outliers, add new features if required to the
        data frame
        """
        # Load the dataset
        self.df = pd.read_csv(self.data_path)

        # Drop the unnamed column
        self.df = self.df.loc[:, ~self.df.columns.str.contains('^Unnamed')]

        # Convert 'timestamp' to datetime and sort the DataFrame by 'timestamp'
        self.df['tmsp'] = pd.to_datetime(self.df['tmsp'])
        self.df.sort_values('tmsp', inplace=True)

        # Detect outliers in the initial data
        logging.info("Outliers in initial data:")
        # self.detect_outliers(self.df)

        # Calculate the historical success rate for each PSP
        psp_success_rate = self.df.groupby('PSP')['success'].mean()
        logging.info(f'Average PSP success rate for each PSP is: {psp_success_rate}')

        # Add the success rate as a new feature
        self.df['psp_success_rate'] = self.df['PSP'].map(psp_success_rate)

        # Identify multiple payment attempts within one minute, with the same amount of money and
        # from the same country as an attempt from a single purchase until the attempt is successful.
        time_window = timedelta(minutes=1)
        self.df['attempt_group'] = ((self.df['tmsp'].diff() <= time_window) &
                                    (self.df['amount'].diff() == 0) &
                                    (self.df['country'].shift() == self.df['country']) &
                                    (self.df['success'].shift() == 0)).cumsum()

        # Aggregate the data by 'attempt_group'
        self.df_grouped = self.df.groupby('attempt_group').agg({
            'tmsp': 'first',
            'country': 'first',
            'amount': 'first',
            'success': 'max',
            'PSP': 'first',
            '3D_secured': 'first',
            'card': 'first',
            'psp_success_rate': 'first'
        }).reset_index(drop=True)

        # Detect outliers in the grouped data
        logging.info("Outliers in grouped data:")
        # self.detect_outliers(self.df_grouped)

        # Define the preprocessor
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['country', 'card', 'PSP'])
            ],
            remainder='passthrough'
        )

    def detect_outliers(self, df):
        """
        Detect the outliers available in the data
        :param df: data in the form of dataframe to detect outliers
        """
        numerical_cols = df.select_dtypes(include=['int64', 'float64'])
        # Calculate z-scores for each column
        z_scores = numerical_cols.apply(zscore)

        # Plot separate histogram plots using seaborn for each column
        for column in z_scores.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(z_scores[column], kde=True)
            plt.title(f'Z-Score Distribution for {column}')
            plt.xlabel('Z-Score')
            plt.ylabel('Frequency')
            plt.draw()
            plt.pause(0.01)
            # Using 'plt.show()' instead of plt.pause() will stop the execution of code until the user
            # closes the shown plot

    def visualize_initial_data_analysis(self):
        """
        Creates graphs and visualizes all the important data information
        """
        # Initial Data Analysis Visualization
        # Distribution of transaction amounts
        plt.figure(figsize=(10, 6))
        sns.histplot(self.df['amount'], bins=50, kde=True)
        plt.xscale('log')
        plt.title('Distribution of Transaction Amounts')
        plt.xlabel('Amount (log scale)')
        plt.ylabel('Frequency')
        plt.draw()
        plt.pause(0.01)
        # Using 'plt.show()' instead of plt.pause() will stop the execution of code until the user
        # closes the shown plot

        # Calculate success rates for specified columns and visualize them
        columns_to_log = ['country', 'PSP', '3D_secured', 'card']

        for column in columns_to_log:
            success_rate = self.df.groupby(column)['success'].mean()

            plt.figure(figsize=(10, 6))
            sns.barplot(x=success_rate.index, y=success_rate.values)
            plt.title(f'Success Rate by {column}')
            plt.xlabel(column)
            plt.ylabel('Success Rate')
            plt.draw()
            plt.pause(0.01)

        # Correlation matrix
        numerical_cols = self.df.select_dtypes(include=['int64', 'float64'])
        plt.figure(figsize=(10, 6))
        sns.heatmap(numerical_cols.corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Matrix')
        plt.draw()
        plt.pause(0.01)
        # Using 'plt.show()' instead of plt.pause() will stop the execution of code until the user
        # closes the shown plot

    def select_features(self, model, X, y):
        """
        Selects the features required to train the model
        :param model: model for which the features will be selected
        :param X: dependent variable data
        :param y: target variable data
        """
        # Apply preprocessing
        X_preprocessed = self.preprocessor.fit_transform(X)
        X_preprocessed = pd.DataFrame(X_preprocessed, columns=self.preprocessor.get_feature_names_out())

        # Feature Selection using Chi-Squared
        selector_chi2 = SelectKBest(score_func=chi2, k=5)
        selector_chi2.fit(X_preprocessed, y)
        selected_features_chi2 = X_preprocessed.columns[selector_chi2.get_support()]
        logging.info(f"Selected features by Chi-squared: {selected_features_chi2}")

        # Feature Selection using Mutual Information
        selector_mutual_info = SelectKBest(score_func=mutual_info_classif, k=5)
        selector_mutual_info.fit(X_preprocessed, y)
        selected_features_mutual_info = X_preprocessed.columns[selector_mutual_info.get_support()]
        logging.info(f"Selected features by Mutual Information: {selected_features_mutual_info}")

        # Feature Selection using RFECV
        selector_rfecv = RFECV(estimator=model, step=1, cv=5, scoring='accuracy')
        selector_rfecv = selector_rfecv.fit(X_preprocessed, y)
        selected_features_rfecv = X_preprocessed.columns[selector_rfecv.support_]
        logging.info(f"Selected features by RFECV: {selected_features_rfecv}")

        # Combine selected features from all methods
        selected_features = list(
            set(selected_features_rfecv) | set(selected_features_chi2) | set(selected_features_mutual_info))
        logging.info(f"Combined selected features: {selected_features}")

        # Cross-validation scores for each feature selection method
        cv_scores_rfecv = cross_val_score(model, X_preprocessed[selected_features_rfecv], y, cv=5).mean()
        cv_scores_chi2 = cross_val_score(model, X_preprocessed[selected_features_chi2], y, cv=5).mean()
        cv_scores_mutual_info = cross_val_score(model, X_preprocessed[selected_features_mutual_info], y, cv=5).mean()
        cv_scores_combined = cross_val_score(model, X_preprocessed[selected_features], y, cv=5).mean()

        logging.info(f"Cross-validation scores for RFECV: {cv_scores_rfecv}")
        logging.info(f"Cross-validation scores for Chi-squared: {cv_scores_chi2}")
        logging.info(f"Cross-validation scores for Mutual Information: {cv_scores_mutual_info}")
        logging.info(f"Cross-validation scores for Combined Features: {cv_scores_combined}")

        return selected_features

    def train_and_select_best_model(self, models_and_hyperparameters):
        """
        Selects the features required for training each model, trains the model, evaluates the model with different
        metrics, finally chooses the best model from the list of models
        :param models_and_hyperparameters: different models and their respective hyper parameters
        :return: best model
        """
        # Train the model for predicting success for all PSPs together
        X = self.df_grouped[['country', 'amount', '3D_secured', 'card', 'PSP', 'psp_success_rate']]
        y = self.df_grouped['success']

        # Split the data into training and testing sets with a minimum of 1 sample in the test set
        test_size = min(0.2, max(1 / len(X), 0.01))
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, train_size=0.8, random_state=42)

        best_weighted_score = 0
        best_model = None
        predictions = None

        for model_name, model, param_grid in models_and_hyperparameters:
            # Create a new instance of the model
            best_model_instance = model.__class__()

            # Select features
            selected_features = self.select_features(model, X_train, y_train)
            X_train_preprocessed = self.preprocessor.transform(X_train)
            X_train_preprocessed = pd.DataFrame(X_train_preprocessed,
                                                columns=self.preprocessor.get_feature_names_out())
            X_train_preprocessed = X_train_preprocessed[selected_features]

            X_test_preprocessed = self.preprocessor.transform(X_test)
            X_test_preprocessed = pd.DataFrame(X_test_preprocessed,
                                               columns=self.preprocessor.get_feature_names_out())
            X_test_preprocessed = X_test_preprocessed[selected_features]

            imb_pipeline = ImbPipeline(steps=[
                ('smote', SMOTE(random_state=42)),
                ('classifier', best_model_instance)
            ])

            if param_grid:
                grid_search = GridSearchCV(imb_pipeline, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
                grid_search.fit(X_train_preprocessed, y_train)
                best_model_instance = grid_search.best_estimator_
            else:
                imb_pipeline.fit(X_train_preprocessed, y_train)
                best_model_instance = imb_pipeline

            # Cross-validation
            cv_scores = cross_val_score(best_model_instance, X_train_preprocessed, y_train, cv=5)
            logging.info(f"Cross-validation scores: {cv_scores}")
            logging.info(f"Mean cross-validation score: {cv_scores.mean()}")

            predictions = best_model_instance.predict(X_test_preprocessed)
            precision = precision_score(y_test, predictions)
            recall = recall_score(y_test, predictions)
            f1 = f1_score(y_test, predictions)
            roc_auc = roc_auc_score(y_test, best_model_instance.predict_proba(X_test_preprocessed)[:, 1])
            accuracy = accuracy_score(y_test, predictions)

            # Calculate a combined score: this could be a simple average of accuracy and mean cv score
            combined_accuracy = (accuracy + cv_scores.mean()) / 2
            logging.info(f"Combined Accuracy: {combined_accuracy}")

            # Calculate weighted score
            weighted_score = (precision + recall + f1 + roc_auc + accuracy) / 5

            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_model = best_model_instance
                self.best_model_features = selected_features

            logging.info(f"{model_name} Model Evaluation:")
            logging.info(classification_report(y_test, predictions))
            logging.info(f"Precision: {precision:.2f}")
            logging.info(f"Recall: {recall:.2f}")
            logging.info(f"F1-Score: {f1:.2f}")
            logging.info(f"ROC-AUC: {roc_auc:.2f}")
            logging.info(f"Accuracy: {accuracy:.2f}")
            logging.info(f"Weighted Score: {weighted_score:.2f}")

            # Compute confusion matrix for error analysis
            cm = confusion_matrix(y_test, predictions)
            logging.info(f"Confusion Matrix:\n{cm}")

            # Compute False Positive Rate and False Negative Rate for error analysis
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            FPR = FP / cm.sum(axis=0)
            FNR = FN / cm.sum(axis=1)
            logging.info(f"False Positive Rate for {model_name}: {FPR}")
            logging.info(f"False Negative Rate for {model_name}: {FNR}")

        self.best_model = best_model
        logging.info(f'Best Model is: {self.best_model}')

        return self.best_model

    def predict_best_PSP(self, transactions):
        """
        Uses the best model which is trained on existing data to predict the best PSP for new transaction which will
        increase the success rate and minimize the transaction fees
        :param transactions: single transaction or multiple set of transactions which needs to be routed for best PSP
        :return: a list of tuples which contains transaction and best PSP to route the respective transaction
        """
        best_PSPs = []
        if isinstance(transactions, str):
            transactions = json.loads(transactions)

        # Calculate PSP success rates based on historical data
        psp_success_rate = self.df.groupby('PSP')['success'].mean()

        for transaction in transactions:
            logging.info(f'Finding best PSP for the transaction: {transaction}')
            best_PSP = None
            highest_score = float('-inf')

            for PSP in self.fee_structure.keys():

                # Create a copy of the transaction data
                transaction_copy = transaction.copy()
                transaction_copy['PSP'] = PSP

                # Add the PSP success rate to the transaction
                transaction_copy['psp_success_rate'] = psp_success_rate[PSP]

                # Convert the dictionary to a DataFrame
                transaction_df = pd.DataFrame([transaction_copy])

                # Transform the new data
                transaction_transformed = self.preprocessor.transform(transaction_df)
                transaction_transformed = pd.DataFrame(transaction_transformed,
                                                       columns=self.preprocessor.get_feature_names_out())
                logging.info(f'Transaction transformed features: {list(transaction_transformed.columns)}')

                # Apply feature selection
                selected_features = self.best_model_features  # Get the selected features for this model

                transaction_transformed = transaction_transformed[selected_features]
                logging.info(f'Model: {self.best_model}')
                logging.info(f'features to be used for prediction: {list(transaction_transformed.columns)}')

                success_rate = self.best_model.predict_proba(transaction_transformed)[0][1]

                # Calculate expected cost
                cost_success = self.fee_structure[PSP]['success']
                cost_failure = self.fee_structure[PSP]['failure']
                expected_cost = (success_rate * cost_success) + ((1 - success_rate) * cost_failure)
                # Calculate score considering both success rate and expected cost
                score = success_rate - expected_cost
                logging.info(f'Score is: {score} for PSP: {PSP}')

                # Select the PSP that has the highest score (meaning high success rate and low transaction fees)
                if score > highest_score:
                    best_PSP = PSP
                    highest_score = score
            logging.info(f'Highest score for this transaction is: {highest_score}')
            best_PSPs.append((transaction, best_PSP))
            logging.info(f'Best Psp for each trans: {best_PSPs}')
        return best_PSPs


data_path = 'synthetic_transactions.csv'
fee_structure = {
    'PSP1': {'success': 5, 'failure': 2},
    'PSP2': {'success': 3, 'failure': 1},
    'PSP3': {'success': 9, 'failure': 4},
    'PSP4': {'success': 1, 'failure': 0.5},
    'PSP5': {'success': 10, 'failure': 6}
}

model = AutomaticRoutingModel(data_path, fee_structure)
model.load_and_preprocess_data()
model.visualize_initial_data_analysis()

# Hyperparameters for Random forest classifier
rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

# Hyperparameters for Gradient boosting classifier
gb_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

# Hyperparameters for XGB Classifier
xgb_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__max_depth': [3, 5, 7]
}

# Hyperparameters for Decision Tree classifier
dt_param_grid = {
    'classifier__max_depth': [None, 10, 20, 30],
    'classifier__min_samples_split': [2, 5, 10]
}

models_and_hyperparameters = [
    ('Logistic Regression', LogisticRegression(max_iter=1000), None),
    ('Random Forest', RandomForestClassifier(random_state=42), rf_param_grid),
    ('Gradient Boosting', GradientBoostingClassifier(random_state=42), gb_param_grid),
    ('XGBoost', XGBClassifier(random_state=42), xgb_param_grid),
    ('Decision Tree', DecisionTreeClassifier(random_state=42), dt_param_grid)
]

# training each of the model and selecting the best model based on different evaluation metrics
model.train_and_select_best_model(models_and_hyperparameters)

# Load the transactions from the csv file
transactions_df = pd.read_csv('synthetic_transactions.csv')
transactions_df = transactions_df.loc[:, ~transactions_df.columns.str.contains('^Unnamed')]

# Select only the first 20 transactions
transactions_df = transactions_df.head(20)

transactions_df = transactions_df.drop(['tmsp', 'success', 'PSP'], axis=1)

# Convert the DataFrame to a list of dictionaries
transactions_list = transactions_df.to_dict(orient='records')

best_psps = model.predict_best_PSP(transactions_list)

# Print the results
for transaction, best_psp in best_psps:
    print(f"Transaction: {transaction}, Best PSP: {best_psp}")
