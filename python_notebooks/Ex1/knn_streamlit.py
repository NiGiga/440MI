from pyclbr import Class

import streamlit as st
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn

################################
# Train data
################################
train_data = [
    # Class 0 — inner region
    ([-0.9,  0.1], 0),
    ([-0.7,  0.6], 0),
    ([-0.6, -0.5], 0),
    ([-0.4,  0.2], 0),
    ([-0.3,  0.8], 0),
    ([-0.2, -0.8], 0),
    ([ 0.0,  0.0], 0),
    ([ 0.1,  0.6], 0),
    ([ 0.2, -0.5], 0),
    ([ 0.4,  0.3], 0),
    ([ 0.5, -0.7], 0),
    ([ 0.6,  0.7], 0),
    ([ 0.7, -0.2], 0),
    ([ 0.8,  0.3], 0),
    ([ 0.0, -1.0], 0),
    ([-1.0, -0.4], 0),

    # Class 1 — outer region
    ([-2.2,  0.0], 1),
    ([-2.0,  0.9], 1),
    ([-1.8, -1.2], 1),
    ([-1.5,  1.6], 1),
    ([-1.3, -1.8], 1),
    ([-0.7,  2.0], 1),
    ([-0.4, -2.1], 1),
    ([ 0.2,  2.2], 1),
    ([ 0.4, -2.0], 1),
    ([ 1.0,  1.8], 1),
    ([ 1.1, -1.7], 1),
    ([ 1.6,  1.3], 1),
    ([ 1.7, -1.0], 1),
    ([ 2.0,  0.6], 1),
    ([ 2.1, -0.3], 1),
    ([ 0.0,  1.7], 1),
    ([-1.6,  0.3], 1),
    ([ 1.4,  0.2], 1),
]

test_points = [
    [0.2, 0.1],
    [1.3, 0.2],
    [-1.1, 1.1],
    [0.3, -1.5],
]

################################
# KNN Implementation
################################
def euclidean_distance(x, z):
    if len(x) != len(z):
        raise ValueError("Points must have the same number of features.")

    squared_sum = 0.0
    for x_j, z_j in zip(x, z):
        squared_sum += (x_j - z_j) ** 2

    return sqrt(squared_sum)


def knn_predict(train_data, x_new, k=3):
    if k <= 0:
        raise ValueError("k must be positive.")
    if k > len(train_data):
        raise ValueError("k cannot be larger than the training set.")

    distances = []

    for features, label in train_data:
        distance = euclidean_distance(features, x_new)
        distances.append((distance, features, label))

    distances.sort(key=lambda item: item[0])
    neighbors = distances[:k]

    counts = {}
    for _, _, label in neighbors:
        counts[label] = counts.get(label, 0) + 1

    prediction = max(counts, key=counts.get)
    return prediction, neighbors


print("Predictions for the test points with k=5:\n")

for point in test_points:
    prediction, neighbors = knn_predict(train_data, point, k=5)
    print(f"{point} -> class {prediction}")


print("\nEffect of k on the point", test_points[1])

for k in [1, 3, 5, 9]:
    prediction, _ = knn_predict(train_data, test_points[1], k=k)
    print(f"k={k}: class {prediction}")


#################################
# Graphical output
#################################


K = 5
X = np.array([point for point, label in train_data])
y = np.array([label for point, label in train_data])

class_names = {
    0: "Class 0",
    1: "Class 1"
}

class_colors = {
    0: "steelblue",
    1: "darkorange"
}

# Predict the class of each test point.
test_predictions = []

for point in test_points:
    prediction, _ = knn_predict(train_data, point, k=K)
    test_predictions.append(prediction)


# ---------------------------------------------------------
# Plot 1: training observations and classified test points
# ---------------------------------------------------------

for label in np.unique(y):
    mask = (y == label)

    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        label=class_names[label],
        color=class_colors[label],
        s=70,
        edgecolor="black",
        alpha=0.85
    )

for point, prediction in zip(test_points, test_predictions):
    plt.scatter(
        point[0],
        point[1],
        color=class_colors[prediction],
        marker="X",
        s=180,
        edgecolor="black"
    )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"kNN classification of new observations (k = {K})")
plt.legend()
plt.grid(alpha=0.2)
plt.show()


# ---------------------------------------------------------
# Plot 2: decision regions
# ---------------------------------------------------------

x1_values = np.linspace(-2.7, 2.7, 120)
x2_values = np.linspace(-2.7, 2.7, 120)

xx, yy = np.meshgrid(x1_values, x2_values)

grid_points = np.column_stack([xx.ravel(), yy.ravel()])

grid_predictions = []

for point in grid_points:
    prediction, _ = knn_predict(train_data, point, k=K)
    grid_predictions.append(prediction)

zz = np.array(grid_predictions).reshape(xx.shape)

plt.contourf(
    xx,
    yy,
    zz,
    levels=[-0.5, 0.5, 1.5],
    cmap="coolwarm",
    alpha=0.25
)

for label in np.unique(y):
    mask = y == label

    plt.scatter(
        X[mask, 0],
        X[mask, 1],
        label=class_names[label],
        color=class_colors[label],
        s=70,
        edgecolor="black"
    )

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title(f"kNN decision regions (k = {K})")
plt.legend()
plt.show()

############################################
# Newral network implementation
############################################
class SimpleClassiier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleClassiier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        s = self.softmax(x)
        return s

model = SimpleClassiier(input_dim=2, hidden_dim=5, output_dim=2)

class Trainer:
    def __init__(self, model, learning_rate=0.01):
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self, X_train, y_train, epochs=100):
        for epoch in range(epochs):
            self.model.train()
            inputs = torch.tensor(X_train, dtype=torch.float32)
            labels = torch.tensor(y_train, dtype=torch.long)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

    def predict(self, X_test):
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X_test, dtype=torch.float32)
            outputs = self.model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        return predicted.numpy()

class Test:
    def __init__(self, model):
        self.model = model

    def evaluate(self, X_test, y_test):
        predicted = self.model.predict(X_test)
        accuracy = np.mean(predicted == y_test)
        return accuracy




        


############################################
# Streamlit app
############################################
st.set_page_config(
    page_title="KNN Classification Demo",
    layout="wide"
)

st.title("KNN Classification Demo")

X = np.array([point for point, label in train_data])
y = np.array([label for point, label in train_data])

class_names = {
    0: "Class 0",
    1: "Class 1"
}

class_colors = {
    0: "steelblue",
    1: "darkorange"
}


def show_knn_decision_boundary(k, test_points):
    # Create a separate figure for this plot.
    fig, ax = plt.subplots(figsize=(8, 6))

    # Build the grid and predict a class for each grid point.
    x1_values = np.linspace(-2.7, 2.7, 120)
    x2_values = np.linspace(-2.7, 2.7, 120)

    xx, yy = np.meshgrid(x1_values, x2_values)
    grid_points = np.column_stack([xx.ravel(), yy.ravel()])

    grid_predictions = [
        knn_predict(train_data, point, k=k)[0]
        for point in grid_points
    ]

    zz = np.array(grid_predictions).reshape(xx.shape)

    # Plot decision regions.
    ax.contourf(
        xx,
        yy,
        zz,
        levels=[-0.5, 0.5, 1.5],
        colors=[class_colors[0], class_colors[1]],
        alpha=0.25
    )

    # Plot training observations.
    for label in np.unique(y):
        mask = y == label

        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            label=class_names[label],
            color=class_colors[label],
            s=70,
            edgecolor="black",
            alpha=0.85
        )

    # Plot classified test points.
    results = []

    for i, point in enumerate(test_points):
        prediction, _ = knn_predict(train_data, point, k=k)

        ax.scatter(
            point[0],
            point[1],
            color=class_colors[prediction],
            marker="X",
            s=180,
            edgecolor="black",
            label="Test points" if i == 0 else None
        )

        results.append({
            "Feature 1": point[0],
            "Feature 2": point[1],
            "Predicted class": class_names[prediction]
        })

    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title(f"kNN decision regions (k = {k})")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    ax.grid(alpha=0.2)

    # Render the figure inside Streamlit.
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Test point predictions")
    st.dataframe(results)


############################################
# Manual test point selection
############################################

# Initialize the list only once per session.
if "manual_test_points" not in st.session_state:
    st.session_state.manual_test_points = []


st.subheader("Inserisci un test point")

with st.form("add_test_point_form"):
    col1, col2 = st.columns(2)

    with col1:
        x1 = st.number_input(
            "Feature 1",
            min_value=-2.7,
            max_value=2.7,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

    with col2:
        x2 = st.number_input(
            "Feature 2",
            min_value=-2.7,
            max_value=2.7,
            value=0.0,
            step=0.1,
            format="%.2f"
        )

    submitted = st.form_submit_button("Aggiungi punto")

if submitted:
    st.session_state.manual_test_points.append([x1, x2])


col1, col2 = st.columns(2)

with col1:
    if st.button("Rimuovi ultimo punto"):
        if st.session_state.manual_test_points:
            st.session_state.manual_test_points.pop()

with col2:
    if st.button("Cancella tutti i punti"):
        st.session_state.manual_test_points = []


k = st.slider(
    "Numero di vicini (k)",
    min_value=1,
    max_value=len(train_data),
    value=5,
    step=2
)

if not st.session_state.manual_test_points:
    st.info("Inserisci le coordinate e premi 'Aggiungi punto'.")

# Always draw the decision regions, even without test points.
show_knn_decision_boundary(
    k,
    st.session_state.manual_test_points
)


