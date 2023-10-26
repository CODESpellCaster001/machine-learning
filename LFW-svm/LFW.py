from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Load the LFW dataset
lfw = datasets.fetch_lfw_people(min_faces_per_person=70, resize=0.4)

# Extract the image data and target labels
X = lfw.data
y = lfw.target

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Principal Component Analysis (PCA) for dimensionality reduction
n_components = 150  # You can adjust this number based on your needs
pca = PCA(n_components=n_components, whiten=True, svd_solver='randomized').fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

# Create an SVM classifier
classifier = SVC(kernel='linear', C=1.0, random_state=42)

# Fit the model on the PCA-transformed data
classifier.fit(X_train_pca, y_train)

# Predict
y_pred = classifier.predict(X_test_pca)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
