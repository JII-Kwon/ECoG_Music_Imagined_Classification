from utils_basic_music import np, pd, plt, sns
import os

class ml_models:
    
    def scaling(self, data, method='standard'):
        
        if method == 'standard':
            from sklearn.preprocessing import StandardScaler

            scaler = StandardScaler()
        elif method == 'min-max': 
            from sklearn.preprocessing import MinMaxScaler
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            
        data_scaled = scaler.fit_transform(data)
        data_scaled = pd.DataFrame(data_scaled, columns=data.columns)

        return data_scaled
    
    def strat_kfold(self, n_splits=5, random_state=42, shuffle=True):
        from sklearn.model_selection import StratifiedKFold
        if shuffle == False:
            strat_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle)
        else:
            strat_k_fold = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        
        return strat_k_fold
        
    def order_forFW(self, train_data, train_label, random_state = None):
        
        from sklearn.ensemble import RandomForestClassifier

        full_model = RandomForestClassifier(random_state=random_state)
        full_model.fit(train_data, train_label)

        importances = full_model.feature_importances_
        sorted_indices = np.argsort(importances)[::-1]
        sorted_feature_name = train_data.columns[sorted_indices]
        return sorted_indices, sorted_feature_name
    
    def label_transpose(self, train_label):
        from sklearn.preprocessing import LabelEncoder
        
        label_encoder = LabelEncoder()
        transpose_label = label_encoder.fit_transform(train_label)
        orginal_label = label_encoder.inverse_transform(transpose_label)
        return transpose_label, orginal_label
    
    def set_models(self, random_state = 42):
        from sklearn.svm import NuSVC
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import NearestCentroid
        from sklearn.neural_network import MLPClassifier
        from sklearn.neighbors import KNeighborsClassifier as KNN
        
        from sklearn.linear_model import SGDClassifier, LogisticRegression        
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA

        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier, RandomForestClassifier
        
        models = {
        'LDA': LDA(),
       # 'Logistic Regression': LogisticRegression(max_iter=10000, random_state=random_state),
        #'KNN': KNN(),
        'RandomForest': RandomForestClassifier(random_state=random_state),
        #'GaussianNB': GaussianNB(),
        'NuSVC': NuSVC(random_state=random_state, probability=True),
        #'NearestCentroid': NearestCentroid(),
        #"MLPClassifier": MLPClassifier(max_iter=1000, random_state=random_state),
        'SGDClassifier': SGDClassifier(random_state=random_state),
        'Bagging': BaggingClassifier(random_state=random_state)
        }
        
        return models
    
    def fw_cross_val(self, train_data, train_label, sorted_indices, models, strat_k_fold, imbalance = None):
        from sklearn.model_selection import  cross_val_score
        
        results = {name: [] for name in models}

        for num_features in range(1, len(sorted_indices) + 1):
            selected_features = sorted_indices[:num_features]
            X_selected = train_data.iloc[:, selected_features]
            
            # For each model, we perform cross-validation
            for name, model in models.items():
                if imbalance == None:
                    cv_results = cross_val_score(model, X_selected, train_label, cv=strat_k_fold, scoring='balanced_accuracy')
                else:
                    cv_results = cross_val_score(model, X_selected, train_label, cv=strat_k_fold, scoring='balanced_accuracy')
                results[name].append((np.mean(cv_results), np.std(cv_results)))
                
        return results
        
    def evaluate_best_model(self, train_data, train_label, test_data, test_label, sorted_indices, models, strat_k_fold, imbalance=None):
        from sklearn.metrics import balanced_accuracy_score

        cv_results = self.fw_cross_val(train_data, train_label, sorted_indices, models, strat_k_fold, imbalance)

        best_model_name = max(cv_results, key=lambda k: max(mean for mean, std in cv_results[k]))
        best_model = models[best_model_name]

        # Number of features for the best model (you might need a more sophisticated approach to select the best feature set)
        best_num_features = np.argmax([mean for mean, std in cv_results[best_model_name]]) + 1

        # Train the best model on the entire training set
        best_features = sorted_indices[:best_num_features]
        X_train_selected = train_data.iloc[:, best_features]
        best_model.fit(X_train_selected, train_label)

        # Evaluate the best model on the test set
        X_test_selected = test_data.iloc[:, best_features]
        predictions = best_model.predict(X_test_selected)
       # proba = best_model.predict_proba(X_test_selected)
        
        test_accuracy = balanced_accuracy_score(test_label, predictions)

        return best_model, test_accuracy, cv_results, predictions#, proba
    
    def plot_cv_result(self, chance_level, cv_result, sorted_indices, title, fold_name):
        
        x_values = range(1, len(sorted_indices) + 1)
        plt.figure(figsize=(12, 6))
        
        for name, performance in cv_result.items():
            means = [mean for mean, std in performance]
            stds = [std for mean, std in performance]
            ste = [std / (len(sorted_indices) ** 0.5) for std in stds]
            
            plt.errorbar(x_values, means, yerr=stds, label=name, capsize=5)
            
        plt.axhline(chance_level, linestyle=':', color='gray', linewidth=2)
        plt.title('Test accuracy:' + str(title*100) + '%')
        plt.xlabel('Number of Features')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(False)
        plt.savefig(os.path.join(fold_name, 'total_clf.png'), dpi = 300)
        plt.close()
        
        
        for name, performance in cv_result.items():
            x_values = range(1, len(sorted_indices) + 1)
            plt.figure(figsize=(12, 6))
            means = [mean for mean, std in performance]
            stds = [std for mean, std in performance]
            ste = [std / (len(sorted_indices) ** 0.5) for std in stds]
            
            plt.errorbar(x_values, means, yerr=stds, label=name, capsize=5, color='midnightblue')
            
            plt.axhline(chance_level, linestyle=':', color='gray', linewidth=2)
            plt.title('Model Performance with Increasing Number of Features')
            plt.xlabel('Number of Features')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(False)
            plt.savefig(os.path.join(fold_name, name+'.png'), dpi = 300)
            plt.close()
            
    def sampling(self, count, dataset, data_label, randstate=None):
        
        from imblearn.over_sampling import SMOTE
        from imblearn.under_sampling import RandomUnderSampler
        from collections import Counter

        NumTrain_org = Counter(data_label)
        class_names = sorted(set(data_label))
        data_tt = dataset
        class_tt = data_label
    
        for cs in class_names:
            if NumTrain_org[cs] == count:
                data_tt = data_tt
                class_tt = class_tt
            elif NumTrain_org[cs] > count: #undersampling
                under_sampler = RandomUnderSampler(sampling_strategy={cs:count}, random_state=randstate)
                data_tt, class_tt = under_sampler.fit_resample(data_tt, class_tt)
            elif NumTrain_org[cs] < count: #oversampling
                over_sampler = SMOTE(sampling_strategy={cs:count}, random_state=randstate)
                data_tt, class_tt = over_sampler.fit_resample(data_tt, class_tt)
        
        aug_dataset = data_tt
        aug_class = class_tt
        
        return aug_dataset, aug_class
    
    
    
    def plot_confusion(self, cm, fig_name = None, title = None, cmap=plt.cm.Blues, annot = True):
        plt.figure()
        if title != None: plt.title(title)       
        
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
       # plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(['SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6']))
        plt.xticks(tick_marks, ['SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6'], rotation=45)
        plt.yticks(tick_marks, ['SD1', 'SD2', 'SD3', 'SD4', 'SD5', 'SD6'])
    
        sns.heatmap(cm, cmap = cmap, annot = annot)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        if fig_name !=None: plt.savefig(fig_name)
        plt.close()
        
        
        
        
        


"""
 scores_per_fold = {name: [] for name in models.keys()}
 performance_results = {name: [] for name in models.keys()}
 
 for num_features in range(1, len(sorted_indices) + 1):
     for key in scores_per_fold:
         scores_per_fold[key] = []
 
     selected_features = sorted_indices[:num_features]
     X_selected = train_data.iloc[:, selected_features]
 
     for train_index, test_index in strat_k_fold.split(X_selected, train_label):
         # Split the data
         X_train, X_test = X_selected.iloc[train_index], X_selected.iloc[test_index]
         y_train, y_test = train_label[train_index], train_label[test_index]
 
         for name, model in models.items():
             # Train the model on the current subset of features
             model.fit(X_train, y_train)
 
             # Make predictions and evaluate
             predictions = model.predict(X_test)
             accuracy = balanced_accuracy_score(y_test, predictions)  # or balanced_accuracy_score, as needed
 
             # Store the score for this fold
             scores_per_fold[name].append(accuracy)
 
     # After all folds are complete, calculate the mean and standard deviation
     for name in models.keys():
         accuracies = scores_per_fold[name]
         performance_results[name].append((np.mean(accuracies), np.std(accuracies)))
     
"""