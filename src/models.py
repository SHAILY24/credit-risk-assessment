from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler

def train_random_forest(X_train, y_train):
    rf_model = RandomForestClassifier(n_estimators=200, random_state=42)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_gradient_boosting(X_train, y_train):
    gb_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, random_state=42
    )
    gb_model.fit(X_train, y_train)
    return gb_model

def train_neural_network(X_train, y_train):
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Build the neural network
    nn_model = Sequential()
    nn_model.add(Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'))
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