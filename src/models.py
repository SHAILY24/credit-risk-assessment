from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

def train_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='f1',
        verbose=1
    )

    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    return best_rf

def train_gradient_boosting(X_train, y_train):
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42
    )
    gb_model.fit(X_train, y_train)
    return gb_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.preprocessing import StandardScaler

def train_neural_network(X_train, y_train):
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Build the neural network
    nn_model = Sequential()
    nn_model.add(Input(shape=(X_train_scaled.shape[1],)))
    nn_model.add(Dense(64, activation='relu'))
    nn_model.add(Dense(32, activation='relu'))
    nn_model.add(Dense(1, activation='sigmoid'))
    
    # Compile the model
    nn_model.compile(
        loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']
    )
    
    # Train the model
    nn_model.fit(
        X_train_scaled, y_train, epochs=100, batch_size=16, verbose=0
    )
    
    return nn_model, scaler