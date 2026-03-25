import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ─────────────────────────────────────────────────────────────────
#  PIMA INDIANS DIABETES DATASET (public domain)
#  Features: Pregnancies, Glucose, BloodPressure, SkinThickness,
#            Insulin, BMI, DiabetesPedigreeFunction, Age
#  Label: 0 = No Diabetes, 1 = Diabetes
# ─────────────────────────────────────────────────────────────────

raw_data = np.array([
    [6,148,72,35,0,33.6,0.627,50,1],[1,85,66,29,0,26.6,0.351,31,0],
    [8,183,64,0,0,23.3,0.672,32,1],[1,89,66,23,94,28.1,0.167,21,0],
    [0,137,40,35,168,43.1,2.288,33,1],[5,116,74,0,0,25.6,0.201,30,0],
    [3,78,50,32,88,31.0,0.248,26,1],[10,115,0,0,0,35.3,0.134,29,0],
    [2,197,70,45,543,30.5,0.158,53,1],[8,125,96,0,0,0.0,0.232,54,1],
    [4,110,92,0,0,37.6,0.191,30,0],[10,168,74,0,0,38.0,0.537,34,1],
    [10,139,80,0,0,27.1,1.441,57,0],[1,189,60,23,846,30.1,0.398,59,1],
    [5,166,72,19,175,25.8,0.587,51,1],[7,100,0,0,0,30.0,0.484,32,1],
    [0,118,84,47,230,45.8,0.551,31,1],[7,107,74,0,0,29.6,0.254,31,1],
    [1,103,30,38,83,43.3,0.183,33,0],[1,115,70,30,96,34.6,0.529,32,1],
    [3,126,88,41,235,39.3,0.704,27,0],[8,99,84,0,0,35.4,0.388,50,0],
    [7,196,90,0,0,39.8,0.451,41,1],[9,119,80,35,0,29.0,0.263,29,1],
    [11,143,94,33,146,36.6,0.254,51,1],[10,125,70,26,115,31.1,0.205,41,1],
    [7,147,76,0,0,39.4,0.257,43,1],[1,97,66,15,140,23.2,0.487,22,0],
    [13,145,82,19,110,22.2,0.245,57,0],[5,117,92,0,0,34.1,0.337,38,0],
    [5,109,75,26,0,36.0,0.546,60,0],[3,158,76,36,245,31.6,0.851,28,1],
    [3,88,58,11,54,24.8,0.267,22,0],[6,92,92,0,0,19.9,0.188,28,0],
    [10,122,78,31,0,27.6,0.512,45,0],[4,103,60,33,192,24.0,0.966,33,0],
    [11,138,76,0,0,33.2,0.420,35,0],[9,102,76,37,0,32.9,0.665,46,1],
    [2,90,68,42,0,38.2,0.503,27,1],[4,111,72,47,207,37.1,1.390,56,1],
    [3,180,64,25,70,34.0,0.271,26,0],[7,133,84,0,0,40.2,0.696,37,0],
    [7,106,92,18,0,22.7,0.235,48,0],[9,171,110,24,240,45.4,0.721,54,1],
    [7,159,64,0,0,27.4,0.294,40,0],[0,180,66,39,0,42.0,1.893,25,1],
    [1,146,56,0,0,29.7,0.564,29,0],[2,71,70,27,0,28.0,0.586,22,0],
    [7,103,66,32,0,39.1,0.344,31,1],[7,105,0,0,0,0.0,0.305,24,0],
    [1,103,80,11,82,19.4,0.491,22,0],[1,101,50,15,36,24.2,0.526,26,0],
    [5,88,66,21,23,24.4,0.342,30,0],[8,176,90,34,300,33.7,0.467,58,1],
    [7,150,66,42,342,34.7,0.718,42,0],[1,73,50,10,0,23.0,0.248,21,0],
    [7,187,68,39,304,37.7,0.254,41,1],[0,100,88,60,110,46.8,0.962,31,0],
    [0,146,82,0,0,40.5,1.781,44,0],[0,105,64,41,142,41.5,0.173,22,0],
    [2,84,0,0,0,0.0,0.304,21,0],[8,133,72,0,0,32.9,0.270,39,1],
    [5,44,62,0,0,25.0,0.587,36,0],[2,141,58,34,128,25.4,0.699,24,0],
    [7,114,66,0,0,32.8,0.258,42,1],[5,99,74,27,0,29.0,0.203,32,0],
    [0,109,88,30,0,32.5,0.855,38,1],[2,109,92,0,0,42.7,0.845,54,0],
    [1,95,66,13,38,19.6,0.334,25,0],[4,146,85,27,100,28.9,0.189,27,0],
    [2,100,66,20,90,32.9,0.867,28,1],[5,139,64,35,140,28.6,0.411,26,0],
    [13,126,90,0,0,43.4,0.583,42,1],[4,129,86,20,270,35.1,0.231,23,0],
    [1,79,75,30,0,32.0,0.396,22,0],[1,0,48,20,0,24.7,0.140,22,0],
    [7,62,78,0,0,32.6,0.391,41,0],[5,95,72,33,0,37.7,0.370,27,0],
    [0,131,0,0,0,43.2,0.270,26,1],[2,112,66,22,0,25.0,0.307,24,0],
    [3,113,44,13,0,22.4,0.140,22,0],[2,74,0,0,0,0.0,0.102,22,0],
    [7,83,78,26,71,29.3,0.767,36,0],[0,101,65,28,0,24.6,0.237,22,0],
    [5,137,108,0,0,48.8,0.227,37,1],[2,110,74,29,125,32.4,0.698,27,0],
    [13,106,70,0,0,34.2,0.251,52,0],[2,100,68,25,71,38.5,0.324,26,0],
    [15,136,70,32,110,37.1,0.153,43,1],[1,107,68,19,0,26.5,0.165,24,0],
    [1,80,55,0,0,19.1,0.258,21,0],[4,123,80,15,176,32.0,0.443,34,0],
    [7,81,78,40,48,46.7,0.261,42,0],[4,134,72,0,0,23.8,0.277,60,1],
    [2,142,82,18,64,24.7,0.761,21,0],[6,144,72,27,228,33.9,0.255,40,0],
    [2,92,62,28,0,31.6,0.130,24,0],[1,71,48,18,76,20.4,0.323,22,0],
    [6,93,50,30,64,28.7,0.356,23,0],[1,122,90,51,220,49.7,0.325,31,1],
    [1,163,72,0,0,39.0,1.222,33,1],[1,151,60,0,0,26.1,0.179,22,0],
    [0,125,96,0,0,22.5,0.262,21,0],[1,81,72,18,40,26.6,0.283,24,0],
    [2,85,65,0,0,39.6,0.930,27,0],[1,126,56,29,152,28.7,0.801,21,0],
    [1,96,122,0,0,22.4,0.207,27,0],[4,144,58,28,140,29.5,0.287,37,0],
    [3,83,58,31,18,34.3,0.336,25,0],[0,95,85,25,36,37.4,0.247,24,1],
    [3,171,72,33,135,33.3,0.199,24,1],[8,155,62,26,495,34.0,0.543,46,1],
    [1,89,76,34,37,31.2,0.192,23,0],[4,76,62,0,0,34.0,0.391,25,0],
    [7,160,54,32,175,30.5,0.588,39,1],[4,146,92,0,0,31.2,0.539,61,1],
    [5,124,74,0,0,34.0,0.220,38,1],[5,78,48,0,0,33.7,0.654,25,0],
    [4,97,60,23,0,28.2,0.443,22,0],[4,99,76,15,51,23.2,0.223,21,0],
    [0,162,76,56,100,53.2,0.759,25,1],[6,111,64,39,0,34.2,0.260,24,0],
    [2,107,74,30,100,33.6,0.404,23,0],[5,132,80,0,0,26.8,0.186,69,0],
    [0,113,76,0,0,33.3,0.278,23,1],[1,88,30,42,99,55.0,0.496,26,1],
    [3,120,70,30,135,42.9,0.452,30,0],[1,118,58,36,94,33.3,0.261,23,0],
    [1,117,88,24,145,34.5,0.403,40,1],[0,105,84,0,0,27.9,0.741,62,1],
    [4,173,70,14,168,29.7,0.361,33,1],[9,122,56,0,0,33.3,1.114,33,1],
    [3,107,62,13,48,22.9,0.678,23,1],[4,132,86,31,0,28.0,0.401,22,0],
    [3,158,70,30,328,35.5,0.344,35,1],[0,123,88,37,0,35.2,0.197,29,0],
    [4,85,58,22,49,27.7,0.306,28,0],[0,84,82,31,125,38.2,0.233,23,0],
    [0,145,0,0,0,44.2,0.630,31,1],[0,135,68,42,250,42.3,0.365,24,1],
    [1,139,62,41,480,40.7,0.536,21,0],[0,173,78,32,265,46.5,1.159,58,0],
    [4,99,72,17,0,25.6,0.294,28,0],[8,194,80,0,0,26.1,0.551,67,0],
    [2,83,66,23,50,32.2,0.497,22,0],[0,101,64,17,0,21.0,0.252,21,0],
    [2,56,56,28,45,24.2,0.332,22,0],[0,187,0,0,0,45.5,0.520,37,1],
    [0,89,66,23,94,28.1,0.167,21,0],[8,155,62,26,495,34.0,0.543,46,1],
    [1,85,66,29,0,26.6,0.351,31,0],[6,148,72,35,0,33.6,0.627,50,1],
    [3,78,50,32,88,31.0,0.248,26,1],[0,137,40,35,168,43.1,2.288,33,1],
    [2,197,70,45,543,30.5,0.158,53,1],[5,116,74,0,0,25.6,0.201,30,0],
    [1,89,76,34,37,31.2,0.192,23,0],[4,76,62,0,0,34.0,0.391,25,0],
    [0,162,76,56,100,53.2,0.759,25,1],[6,111,64,39,0,34.2,0.260,24,0],
    [2,107,74,30,100,33.6,0.404,23,0],[0,84,82,31,125,38.2,0.233,23,0],
    [4,132,86,31,0,28.0,0.401,22,0],[3,158,70,30,328,35.5,0.344,35,1],
    [0,123,88,37,0,35.2,0.197,29,0],[1,117,88,24,145,34.5,0.403,40,1],
    [9,122,56,0,0,33.3,1.114,33,1],[3,107,62,13,48,22.9,0.678,23,1],
    [0,105,84,0,0,27.9,0.741,62,1],[4,173,70,14,168,29.7,0.361,33,1],
    [5,139,64,35,140,28.6,0.411,26,0],[2,141,58,34,128,25.4,0.699,24,0],
    [7,114,66,0,0,32.8,0.258,42,1],[5,99,74,27,0,29.0,0.203,32,0],
    [0,109,88,30,0,32.5,0.855,38,1],[2,109,92,0,0,42.7,0.845,54,0],
    [4,146,85,27,100,28.9,0.189,27,0],[2,100,66,20,90,32.9,0.867,28,1],
])

# ─────────────────────────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────────────────────────

FEATURE_NAMES = [
    "Pregnancies", "Glucose", "Blood Pressure", "Skin Thickness",
    "Insulin", "BMI", "Diabetes Pedigree", "Age"
]

X_all = raw_data[:, :8].astype(np.float64)
y_all = raw_data[:, 8].astype(np.uint8)

# Replace 0s in medical fields with column mean (they're missing values)
zero_invalid_cols = [1, 2, 3, 4, 5]   # Glucose, BP, Skin, Insulin, BMI
for col in zero_invalid_cols:
    col_mean = X_all[X_all[:, col] != 0, col].mean()
    X_all[X_all[:, col] == 0, col] = col_mean

# Normalize features to [0, 1]  (min-max scaling)
X_min = X_all.min(axis=0)
X_max = X_all.max(axis=0)
X_norm = (X_all - X_min) / (X_max - X_min + 1e-8)

# Train / test split  (80 / 20)
indices  = np.random.permutation(len(X_norm))
split    = int(0.8 * len(indices))
train_ix = indices[:split]
test_ix  = indices[split:]

X_train, y_train = X_norm[train_ix], y_all[train_ix]
X_test,  y_test  = X_norm[test_ix],  y_all[test_ix]

print(f"Dataset  : {len(X_all)} samples  |  "
      f"Diabetic: {y_all.sum()}  |  Non-diabetic: {(y_all==0).sum()}")
print(f"Train    : {len(X_train)} samples")
print(f"Test     : {len(X_test)} samples\n")


# ─────────────────────────────────────────────────────────────────
#  LAYERS
# ─────────────────────────────────────────────────────────────────

class layer_dense:
    def __init__(self, n_inputs, neurons):
        self.weights = np.random.randn(n_inputs, neurons) * np.sqrt(2.0 / n_inputs)  # He init
        self.biases  = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases  = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs  = np.dot(dvalues, self.weights.T)


# ─────────────────────────────────────────────────────────────────
#  ACTIVATIONS
# ─────────────────────────────────────────────────────────────────

class activation_Relu:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


class activation_softmax:
    def forward(self, inputs):
        exp_values  = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self, dvalues):
        self.dinputs = np.empty_like(dvalues)
        for i, (out, dval) in enumerate(zip(self.output, dvalues)):
            out = out.reshape(-1, 1)
            jacobian = np.diagflat(out) - np.dot(out, out.T)
            self.dinputs[i] = np.dot(jacobian, dval)


# ─────────────────────────────────────────────────────────────────
#  FUSED SOFTMAX + CROSS-ENTROPY BACKWARD
# ─────────────────────────────────────────────────────────────────

class softmax_loss_backward:
    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        self.dinputs = dvalues.copy()
        self.dinputs[range(samples), y_true] -= 1
        self.dinputs /= samples


# ─────────────────────────────────────────────────────────────────
#  LOSS
# ─────────────────────────────────────────────────────────────────

class losscategoricalcrossentropy:
    def calculate(self, output, y):
        return np.mean(self.forward(output, y))

    def forward(self, y_pred, y_true):
        samples        = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        if len(y_true.shape) == 1:
            correct = y_pred_clipped[range(samples), y_true]
        else:
            correct = np.sum(y_pred_clipped * y_true, axis=1)
        return -np.log(correct)


# ─────────────────────────────────────────────────────────────────
#  ACCURACY
# ─────────────────────────────────────────────────────────────────

class accuracyfunction:
    def calculate(self, y_pred, y_true):
        predictions = np.argmax(y_pred, axis=1)
        return np.mean(predictions == y_true)


# ─────────────────────────────────────────────────────────────────
#  SGD OPTIMIZER  (with momentum + decay)
# ─────────────────────────────────────────────────────────────────

class optimizer_SGD:
    def __init__(self, learning_rate=0.5, decay=1e-4, momentum=0.9):
        self.learning_rate         = learning_rate
        self.current_learning_rate = learning_rate
        self.decay                 = decay
        self.iterations            = 0
        self.momentum              = momentum

    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (
                1.0 / (1.0 + self.decay * self.iterations))

    def update_params(self, layer):
        if self.momentum:
            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums   = np.zeros_like(layer.biases)
            w_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
            layer.weight_momentums = w_updates
            b_updates = self.momentum * layer.bias_momentums   - self.current_learning_rate * layer.dbiases
            layer.bias_momentums   = b_updates
        else:
            w_updates = -self.current_learning_rate * layer.dweights
            b_updates = -self.current_learning_rate * layer.dbiases
        layer.weights += w_updates
        layer.biases  += b_updates

    def post_update_params(self):
        self.iterations += 1


# ─────────────────────────────────────────────────────────────────
#  BUILD MODEL
#  Architecture: 8 → 64 → 32 → 2
#  Input  : 8 health features
#  Output : 2 classes  (0 = No Diabetes, 1 = Diabetes)
# ─────────────────────────────────────────────────────────────────

dense1      = layer_dense(8, 64)   # 8 health features in
activation1 = activation_Relu()
dense2      = layer_dense(64, 32)
activation2 = activation_Relu()
dense3      = layer_dense(32, 2)   # 2 classes out
activation3 = activation_softmax()

loss_fn     = losscategoricalcrossentropy()
accuracy_fn = accuracyfunction()
sl_bwd      = softmax_loss_backward()
optimizer   = optimizer_SGD(learning_rate=0.1, decay=5e-5, momentum=0.85)


def forward_pass(X):
    dense1.forward(X);      activation1.forward(dense1.output)
    dense2.forward(activation1.output); activation2.forward(dense2.output)
    dense3.forward(activation2.output); activation3.forward(dense3.output)
    return activation3.output


def backward_pass(y):
    sl_bwd.backward(activation3.output, y)
    dense3.backward(sl_bwd.dinputs)
    activation2.backward(dense3.dinputs)
    dense2.backward(activation2.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)


# ─────────────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────────────

EPOCHS           = 5001
train_loss_hist  = []
train_acc_hist   = []
test_acc_hist    = []

print("Training...\n")
print(f"{'Epoch':>6}  {'Train Loss':>10}  {'Train Acc':>10}  {'Test Acc':>10}  {'LR':>10}")
print("-" * 55)

for epoch in range(EPOCHS):
    output = forward_pass(X_train)
    loss   = loss_fn.calculate(output, y_train)
    t_acc  = accuracy_fn.calculate(output, y_train)

    # Test accuracy (no gradient needed)
    test_out  = forward_pass(X_test)
    test_acc  = accuracy_fn.calculate(test_out, y_test)

    train_loss_hist.append(loss)
    train_acc_hist.append(t_acc)
    test_acc_hist.append(test_acc)

    # Re-run forward on train set before backward (test forward above may have changed layer state)
    forward_pass(X_train)
    backward_pass(y_train)

    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.update_params(dense3)
    optimizer.post_update_params()

    if epoch % 500 == 0:
        print(f"{epoch:>6}  {loss:>10.4f}  {t_acc:>10.4f}  {test_acc:>10.4f}  "
              f"{optimizer.current_learning_rate:>10.6f}")

print("-" * 55)
print(f"\nFinal Test Accuracy : {test_acc_hist[-1]*100:.2f}%")


# ─────────────────────────────────────────────────────────────────
#  PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────────

def predict_diabetes(patient_data: dict) -> None:
    """
    patient_data keys:
      pregnancies, glucose, blood_pressure, skin_thickness,
      insulin, bmi, diabetes_pedigree, age
    """
    values = np.array([[
        patient_data["pregnancies"],
        patient_data["glucose"],
        patient_data["blood_pressure"],
        patient_data["skin_thickness"],
        patient_data["insulin"],
        patient_data["bmi"],
        patient_data["diabetes_pedigree"],
        patient_data["age"],
    ]], dtype=np.float64)

    # Apply same normalization as training data
    values_norm = (values - X_min) / (X_max - X_min + 1e-8)
    probs       = forward_pass(values_norm)
    pred_class  = np.argmax(probs, axis=1)[0]
    confidence  = probs[0][pred_class] * 100

    print("\n── Patient Prediction ──────────────────────")
    for k, v in patient_data.items():
        print(f"  {k:<22}: {v}")
    print(f"\n  Prediction  : {'⚠ DIABETIC' if pred_class == 1 else '✓ NON-DIABETIC'}")
    print(f"  Confidence  : {confidence:.1f}%")
    print(f"  P(No Diab.) : {probs[0][0]*100:.1f}%  |  P(Diabetic): {probs[0][1]*100:.1f}%")
    print("────────────────────────────────────────────\n")


# ─── Example predictions ─────────────────────
predict_diabetes({
    "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
    "skin_thickness": 35, "insulin": 0, "bmi": 33.6,
    "diabetes_pedigree": 0.627, "age": 50
})

predict_diabetes({
    "pregnancies": 1, "glucose": 85, "blood_pressure": 66,
    "skin_thickness": 29, "insulin": 0, "bmi": 26.6,
    "diabetes_pedigree": 0.351, "age": 31
})


# ─────────────────────────────────────────────────────────────────
#  PLOTS
# ─────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("Diabetes Neural Network — Training Results", fontsize=13, fontweight='bold')

# Loss curve
axes[0].plot(train_loss_hist, color='crimson', linewidth=1.5)
axes[0].set_title("Training Loss")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].grid(True, alpha=0.3)

# Accuracy curves
axes[1].plot(train_acc_hist, color='steelblue',  linewidth=1.5, label='Train')
axes[1].plot(test_acc_hist,  color='darkorange', linewidth=1.5, label='Test', linestyle='--')
axes[1].set_title("Accuracy")
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
axes[1].set_ylim(0, 1); axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Final confusion matrix
final_preds = np.argmax(forward_pass(X_test), axis=1)
cm = np.zeros((2, 2), dtype=int)
for t, p in zip(y_test, final_preds):
    cm[t][p] += 1

im = axes[2].imshow(cm, cmap='Blues')
axes[2].set_xticks([0,1]); axes[2].set_yticks([0,1])
axes[2].set_xticklabels(['Pred: No', 'Pred: Yes'])
axes[2].set_yticklabels(['True: No', 'True: Yes'])
axes[2].set_title("Confusion Matrix (Test Set)")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i][j]), ha='center', va='center',
                     fontsize=14, color='white' if cm[i][j] > cm.max()/2 else 'black')

plt.tight_layout()
plt.savefig('diabetes_training.png', dpi=150, bbox_inches='tight')
plt.show()
print("Plot saved.")
