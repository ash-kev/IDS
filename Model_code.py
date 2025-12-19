import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf


# 1. LOAD DATA

data = pd.read_csv("kdd_sample.csv")

# convert attack types → attack as 1  / normal as 0
data["label"] = data["label"].apply(lambda x: "normal" if x == "normal" else "attack")


# 2. COLUMNS

num_cols = ["duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent", "count"]
cat_cols = ["protocol_type", "service", "flag"]

X = data[num_cols + cat_cols]
y = (data["label"] == "attack").astype(int)


# 3. PREPROCESSING

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

X_processed = preprocess.fit_transform(X)


# 4. TRAIN-TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.25, random_state=42, stratify=y
)


# 5. CLASS WEIGHTS (BALANCING)

weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y),
    y=y
)

class_weights = {0: weights[0], 1: weights[1]}
print("Class Weights:", class_weights)


# 6. NEURAL NETWORK MODEL - VERSION _ 2

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')   # output
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# CALLBACKS
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=5,
        restore_best_weights=True
    )
]
 

# 7. TRAIN MODEL

history = model.fit(
    X_train,
    y_train,
    epochs=40,
    batch_size=16,
    class_weight=class_weights,
    callbacks=callbacks,
    verbose=2
)


# EVALUATE MODEL

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob >= 0.5).astype(int)

print("\nCONFUSION MATRIX:")
print(confusion_matrix(y_test, y_pred))

print("\nCLASSIFICATION REPORT:")
print(classification_report(y_test, y_pred))




# 9. SAMPLE

print("\nSample Predictions:")
for i in range(5):
    print(f"Sample {i}: Predicted = {'attack' if y_pred[i] == 1 else 'normal'} | Confidence = {y_pred_prob[i][0]:.4f}")

"""
def live_predict(sample_dict):
    df = pd.DataFrame([sample_dict])
    X_p = preprocess.transform(df)  
    prob = model.predict(X_p)[0][0]
    pred = "attack" if prob >= 0.5 else "normal"
    print(f"Prediction = {pred.upper()} | Confidence = {prob:.4f}")
    

normal_sample = {
    'duration': 0,
    'protocol_type': 'tcp',
    'service': 'http',
    'flag': 'SF',
    'src_bytes': 181,
    'dst_bytes': 5450,
    'land': 0,
    'wrong_fragment': 0,
    'urgent': 0,
    'hot': 0,
    'num_failed_logins': 0,
    'logged_in': 1,
    'num_compromised': 0,
    'root_shell': 0,
    'su_attempted': 0,
    'num_root': 0,
    'num_file_creations': 0,
    'num_shells': 0,
    'num_access_files': 0,
    'num_outbound_cmds': 0,
    'is_host_login': 0,
    'is_guest_login': 0,
    'count': 1,
    'srv_count': 1
}

live_predict(normal_sample)
"""
