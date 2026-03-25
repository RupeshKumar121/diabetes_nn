# 🧠 Diabetes Neural Network

A fully **handcrafted neural network** built from scratch using only **NumPy** — no TensorFlow, no PyTorch, no sklearn — trained to detect diabetes from patient health data.

---

## 📸 Results

### Training Loss, Accuracy & Confusion Matrix
![Training Results](assets/diabetes_training.png)

---

## 📖 About

This project implements a complete feedforward neural network without any ML frameworks. Every component — layers, activations, loss functions, backpropagation, and the optimizer — is written manually in pure Python and NumPy. The model is trained on the **PIMA Indians Diabetes Dataset** to classify whether a patient is diabetic or not based on 8 clinical features.

---

## 🗃️ Dataset

**PIMA Indians Diabetes Dataset** — a well-known public domain medical dataset originally from the National Institute of Diabetes and Digestive and Kidney Diseases.

| Feature | Description |
|---|---|
| Pregnancies | Number of times pregnant |
| Glucose | Plasma glucose concentration |
| Blood Pressure | Diastolic blood pressure (mm Hg) |
| Skin Thickness | Triceps skinfold thickness (mm) |
| Insulin | 2-hour serum insulin (μU/ml) |
| BMI | Body mass index |
| Diabetes Pedigree | Diabetes pedigree function (genetic influence) |
| Age | Age in years |
| **Label** | 0 = No Diabetes · 1 = Diabetes |

**Preprocessing applied:**
- Zero values in medical columns (Glucose, Blood Pressure, Skin Thickness, Insulin, BMI) are treated as missing and replaced with the **column mean**
- All features normalized to **[0, 1]** using min-max scaling
- **80/20 train-test split** with random shuffling (`seed=42`)

---

## 🏗️ Network Architecture

```
Input Layer      →   8 features (health metrics)
Hidden Layer 1   →   64 neurons  +  ReLU
Hidden Layer 2   →   32 neurons  +  ReLU
Output Layer     →   2 neurons   +  Softmax  (No Diabetes / Diabetes)
```

Weights are initialized using **He initialization** (`× √(2/n_inputs)`) — the recommended initialization for ReLU networks to prevent vanishing/exploding gradients.

---

## ⚡ Activation Functions

### ReLU — Hidden Layers
Used in both hidden layers. ReLU outputs the input directly if positive, otherwise zero. It introduces non-linearity while being fast to compute and avoiding the vanishing gradient problem.

```
f(x) = max(0, x)
Backward: gradient passes through if x > 0, else blocked (set to 0)
```

### Softmax — Output Layer
Used in the output layer for multi-class probability distribution. Converts raw scores into probabilities that sum to 1, giving P(No Diabetes) and P(Diabetes) for each prediction.

```
f(xᵢ) = exp(xᵢ) / Σ exp(xⱼ)
```
The implementation uses `exp(x - max(x))` for numerical stability.

---

## 📉 Loss Function

**Categorical Cross-Entropy** — standard loss for classification tasks. Measures how far the predicted probability distribution is from the true labels. Values are clipped to `[1e-7, 1-1e-7]` to avoid `log(0)` errors.

```
L = -log(predicted probability of the correct class)
```

The backward pass uses a **fused Softmax + Cross-Entropy gradient** — a mathematical shortcut that simplifies the combined derivative to just:

```
dinputs = predicted_probabilities
dinputs[correct_class] -= 1
dinputs /= number_of_samples
```

This is more numerically stable and faster than computing them separately.

---

## 🚀 Optimizer

**SGD with Momentum and Learning Rate Decay**

| Parameter | Value |
|---|---|
| Initial Learning Rate | 0.1 |
| Decay | 5e-5 |
| Momentum | 0.85 |

**Momentum** prevents oscillations by accumulating a velocity in the direction of consistent gradients — like a ball rolling downhill that builds up speed. Instead of a raw gradient step, it blends the current gradient with the previous update direction.

**Learning Rate Decay** gradually reduces the learning rate over epochs:
```
current_lr = initial_lr × (1 / (1 + decay × iterations))
```
This allows larger steps early in training for fast convergence, and smaller precise steps later for fine-tuning.

---

## 📊 Training

| Setting | Value |
|---|---|
| Epochs | 5000 |
| Training Samples | ~160 |
| Test Samples | ~40 |
| Loss Function | Categorical Cross-Entropy |
| Optimizer | SGD + Momentum + Decay |

Every 500 epochs the script prints: epoch number, training loss, training accuracy, test accuracy, and current learning rate.

---

## 📈 Plots Generated

Three plots are produced and saved as `diabetes_training.png`:

- **Training Loss Curve** — shows how loss decreases over 5000 epochs
- **Train vs Test Accuracy** — both plotted together to monitor for overfitting
- **Confusion Matrix** — on the test set, showing True Positives, True Negatives, False Positives, and False Negatives

---

## 🔍 Prediction

The `predict_diabetes()` function takes a dictionary of patient values, applies the same min-max normalization used during training, runs a forward pass, and outputs:

- Whether the patient is **DIABETIC ⚠** or **NON-DIABETIC ✓**
- Confidence percentage
- Full probability breakdown: P(No Diabetes) and P(Diabetes)

**Example:**
```python
predict_diabetes({
    "pregnancies": 6, "glucose": 148, "blood_pressure": 72,
    "skin_thickness": 35, "insulin": 0, "bmi": 33.6,
    "diabetes_pedigree": 0.627, "age": 50
})
# Output → ⚠ DIABETIC  |  Confidence: 91.3%
```

---

## ⚙️ Setup & Run

```bash
pip install numpy matplotlib
python diabetes_nn.py
```

No other dependencies needed. The script trains the model, prints epoch logs, runs example predictions, generates all three plots, and saves them as `diabetes_training.png`.

---

## 🗺️ Roadmap

- [x] Neural network from scratch (no ML frameworks)
- [x] He weight initialization
- [x] ReLU + Softmax activations
- [x] Categorical cross-entropy loss
- [x] SGD with momentum and learning rate decay
- [x] Training + test accuracy tracking
- [x] Loss curve, accuracy curve & confusion matrix plots
- [x] Patient prediction function
- [ ] Dropout regularization
- [ ] Batch training support
- [ ] Save / load trained weights

---

## 📄 License

Built for educational purposes. All rights reserved by Rupesh Kumar.
