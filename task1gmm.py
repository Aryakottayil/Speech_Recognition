import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    Input,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from scipy.stats import mode

# Load and normalize CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# *Use Functional API for CNN Model*
input_layer = Input(shape=(32, 32, 3))
x = Conv2D(32, (3, 3), padding="same", activation="relu")(input_layer)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
x = BatchNormalization()(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)
feature_layer = Dense(256, activation="relu")(x)  # Feature Extraction Layer
x = Dropout(0.5)(feature_layer)
output_layer = Dense(10, activation="softmax")(x)

# Create Model
cnn_model = Model(inputs=input_layer, outputs=output_layer)

# Training callbacks
early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2)

# Compile model
cnn_model.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Train model
history = cnn_model.fit(
    x_train,
    y_train,
    epochs=15,
    batch_size=128,
    validation_data=(x_test, y_test),
    callbacks=[early_stopping, reduce_lr],
)

# *Feature Extraction Model*
feature_extractor = Model(inputs=cnn_model.input, outputs=feature_layer)

# Print model summary to verify feature extraction
cnn_model.summary()
feature_extractor.summary()

# Extract features for training and testing sets
train_features = feature_extractor.predict(x_train)
test_features = feature_extractor.predict(x_test)

# Print shape of extracted features
print("Extracted feature shape:", train_features.shape)

# Reduce dimensions using PCA
pca = PCA(n_components=50)
train_pca = pca.fit_transform(train_features)
test_pca = pca.transform(test_features)

# Train GMM model
gmm = GaussianMixture(n_components=10, covariance_type="full", random_state=42)
gmm.fit(train_pca)

# Predict clusters
gmm_preds = gmm.predict(test_pca)
print("GMM predicted clusters:", np.unique(gmm_preds))


# Function to map GMM clusters to actual labels
def map_gmm_to_labels(gmm_preds, y_true):
    labels = np.zeros_like(gmm_preds)
    for i in range(10):  # 10 clusters
        mask = gmm_preds == i
        if np.sum(mask) > 0:  # Avoid empty clusters
            labels[mask] = mode(y_true[mask])[0][0]
    return labels


# Map clusters to labels
gmm_labels = map_gmm_to_labels(gmm_preds, y_test)

# Compute accuracy
accuracy = np.mean(gmm_labels == y_test.flatten())
print(f"GMM Classification Accuracy: {accuracy * 100:.2f}%")

# Pick a random test image
index = np.random.randint(0, len(x_test))
test_image = x_test[index]

# Extract features, reduce with PCA, and predict with GMM
test_feature = feature_extractor.predict(test_image.reshape(1, 32, 32, 3))
test_pca = pca.transform(test_feature)
pred_cluster = gmm.predict(test_pca)

# Show image and predicted cluster
plt.imshow(test_image)
plt.title(f"GMM Predicted Cluster: {pred_cluster[0]}")
plt.show()




