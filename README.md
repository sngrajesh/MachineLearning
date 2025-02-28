# scikit-learn (sklearn)

## sklearn (Core)

- **Preprocessing**
  - `preprocessing.StandardScaler()`: Standardizes features by removing the mean and scaling to unit variance.
  - `preprocessing.MinMaxScaler()`: Scales features to a given range.
  - `preprocessing.RobustScaler()`: Scales features using statistics that are robust to outliers.
  - `preprocessing.Normalizer()`: Normalizes samples individually to unit norm.
  - `preprocessing.LabelEncoder()`: Encodes target labels with value between 0 and n_classes-1.
  - `preprocessing.OneHotEncoder()`: Encodes categorical features as a one-hot numeric array.
  - `preprocessing.PolynomialFeatures()`: Generates polynomial and interaction features.
  - `impute.SimpleImputer()`: Imputation transformer for completing missing values.

- **Feature Selection**
  - `feature_selection.SelectKBest()`: Select features according to the k highest scores.
  - `feature_selection.RFE()`: Recursive feature elimination.
  - `feature_selection.SelectFromModel()`: Select features from a model with coefficients or feature importances.
  - `feature_selection.VarianceThreshold()`: Basic baseline approach to feature selection. It removes all features whose variance doesn’t meet some threshold.

- **Dimensionality Reduction**
  - `decomposition.PCA()`: Principal component analysis.
  - `decomposition.TruncatedSVD()`: Truncated singular value decomposition.
  - `manifold.TSNE()`: t-distributed Stochastic Neighbor Embedding.
  - `manifold.LocallyLinearEmbedding()`: Locally Linear Embedding.
  - `discriminant_analysis.LinearDiscriminantAnalysis()`: Linear Discriminant Analysis.

- **Model Selection and Evaluation**
  - `model_selection.train_test_split()`: Splits arrays or matrices into random train and test subsets.
  - `model_selection.cross_val_score()`: Evaluates a score by cross-validation.
  - `model_selection.GridSearchCV()`: Performs exhaustive search over specified parameter values for an estimator.
  - `metrics.accuracy_score()`: Accuracy classification score.
  - `metrics.precision_score()`: Precision classification score.
  - `metrics.recall_score()`: Recall classification score.
  - `metrics.f1_score()`: F1 classification score.
  - `metrics.roc_auc_score()`: Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
  - `metrics.confusion_matrix()`: Compute confusion matrix to evaluate the accuracy of a classification.
  - `metrics.mean_squared_error()`: Mean squared error regression loss.
  - `metrics.r2_score()`: R² (coefficient of determination) regression score.

- **Linear Models**
  - `linear_model.LinearRegression()`: Ordinary least squares Linear Regression.
  - `linear_model.LogisticRegression()`: Logistic Regression (classification).
  - `linear_model.Ridge()`: Linear least squares with l2 regularization.
  - `linear_model.Lasso()`: Linear Model trained with L1 prior as regularizer (Lasso).
  - `linear_model.ElasticNet()`: Linear regression with combined L1 and L2 priors as regularizer.

- **Tree-based Models**
  - `tree.DecisionTreeClassifier()`: Decision tree classifier.
  - `tree.DecisionTreeRegressor()`: Decision tree regressor.
  - `ensemble.RandomForestClassifier()`: Random forest classifier.
  - `ensemble.RandomForestRegressor()`: Random forest regressor.
  - `ensemble.GradientBoostingClassifier()`: Gradient boosting classifier.
  - `ensemble.GradientBoostingRegressor()`: Gradient boosting regressor.
  - `ensemble.AdaBoostClassifier()`: AdaBoost classifier.
  - `ensemble.AdaBoostRegressor()`: AdaBoost regressor.
  - `ensemble.ExtraTreesClassifier()`: Extra Trees classifier.
  - `ensemble.ExtraTreesRegressor()`: Extra Trees regressor.

- **Support Vector Machines (SVMs)**
  - `svm.SVC()`: Support vector classification.
  - `svm.SVR()`: Support vector regression.
  - `svm.LinearSVC()`: Linear support vector classification.
  - `svm.LinearSVR()`: Linear support vector regression.

- **Nearest Neighbors**
  - `neighbors.KNeighborsClassifier()`: K-nearest neighbors classifier.
  - `neighbors.KNeighborsRegressor()`: K-nearest neighbors regressor.

- **Clustering**
  - `cluster.KMeans()`: K-means clustering.
  - `cluster.AgglomerativeClustering()`: Agglomerative clustering.
  - `cluster.DBSCAN()`: Density-Based Spatial Clustering of Applications with Noise.
  - `cluster.SpectralClustering()`: Spectral clustering.

- **Pipeline and Column Transformer**
  - `pipeline.Pipeline()`: Sequentially apply a list of transforms and a final estimator.
  - `compose.ColumnTransformer()`: Applies transformers to columns of an array or pandas DataFrame.

- **Gaussian Processes**
  - `gaussian_process.GaussianProcessClassifier()`: Gaussian process classification (GPC) based on Laplace approximation.
  - `gaussian_process.GaussianProcessRegressor()`: Gaussian process regressor.

- **Naive Bayes**
  - `naive_bayes.GaussianNB()`: Gaussian Naive Bayes.
  - `naive_bayes.MultinomialNB()`: Multinomial Naive Bayes.
  - `naive_bayes.BernoulliNB()`: Bernoulli Naive Bayes.
