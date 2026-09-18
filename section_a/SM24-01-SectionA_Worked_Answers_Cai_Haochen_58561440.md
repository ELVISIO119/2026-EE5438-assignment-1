# Section A — Worked Answers

Cai Haochen | Student ID: 58561440 | EE5438 Applied Deep Learning

**Typed study draft — handwriting required.** Review these solutions, write the answers by hand on paper or an iPad, and export all answer sheets as one scanned/handwritten PDF. This typed file does not satisfy the handwriting requirement.

## Questions 1–8

**1. Linear activation layers**

Composing affine layers gives another affine map: $W_2(W_1x+b_1)+b_2=(W_2W_1)x+W_2b_1+b_2$. Without nonlinear activations, depth adds no expressive power beyond a single affine layer and cannot model nonlinear decision boundaries.

**2. Vanishing gradients**

Saturated sigmoid/tanh units have near-zero derivatives. Backpropagation multiplies these derivatives across layers, causing early-layer gradients to vanish. ReLU has derivative one for positive inputs, helping preserve gradients along active paths, although negative inputs have zero gradient.

**3. Dying ReLU**

A ReLU neuron can output zero for every training input, leaving zero gradients and preventing recovery. LeakyReLU retains a small fixed negative-side slope; PReLU learns that slope, allowing gradients through negative inputs.

**4. Softmax versus independent sigmoids**

For mutually exclusive classes, softmax produces a normalized probability distribution whose entries sum to one and compete with each other. Independent sigmoid outputs need not sum to one and are appropriate for nonexclusive, multilabel targets.

**5. Output activation and loss**

Regression (i): linear (identity) output with mean squared error. Binary classification (ii): sigmoid output with binary cross-entropy; numerically, use a logits-based binary cross-entropy implementation without explicitly applying sigmoid during training.

**6. Backpropagation efficiency**

Backpropagation reuses intermediate computations and applies the chain rule to obtain all parameter gradients in roughly one forward and backward pass. Finite differences require separate perturbed evaluations per parameter, scaling poorly and introducing numerical approximation errors.

**7. AdaGrad and RMSprop**

AdaGrad accumulates all past squared gradients, so its denominator continually grows and effective learning rates may become too small. RMSprop uses an exponentially decaying average, forgetting old gradients and avoiding this unbounded accumulation.

**8. AdamW and weight decay**

Standard Adam with L2 regularization adds the penalty gradient before adaptive scaling. AdamW directly shrinks parameters separately from that scaling. This decoupling makes weight decay more predictable and allows more independent tuning of regularization and adaptive updates.

## Questions 9–15

**9. Bias–variance trade-off**

High bias causes underfitting: the model misses important patterns. High variance causes overfitting: it follows training noise and changes substantially with the sample. Balancing model capacity and regularization helps minimize error on unseen data.

**10. Validation versus test data**

Hyperparameter tuning uses feedback to select a model, so it must use validation data. Repeated tuning on test results leaks test information into model selection, producing optimistic estimates. Keep the test set untouched for final evaluation.

**11. L1 and L2 regularization**

L1 penalizes absolute weights and encourages sparsity, potentially setting some weights exactly to zero. L2 penalizes squared weights, smoothly shrinking weights toward zero and discouraging large values without generally producing exact zeros.

**12. Dropout**

Dropout randomly zeros activations during training, reducing co-adaptation and reliance on particular units. At inference, all units are active. With inverted dropout, retained training activations are scaled by $1/(1-p)$, so inference requires no additional scaling.

**13. Data augmentation**

Augmentation creates label-preserving variations of training images, increasing effective diversity and teaching useful invariances. This reduces memorization and improves robustness. Common techniques include horizontal flipping, small rotations, and random cropping or translation.

**14. Classification metrics**

Accuracy = (TP + TN)/(TP + FP + FN + TN) = 150/165 = 90.91%. Precision = TP/(TP + FP) = 50/60 = 83.33%. Recall = TP/(TP + FN) = 50/55 = 90.91%.

**15. Imbalanced classification**

A majority-class classifier can achieve high accuracy while missing rare positives. Report minority-class precision, recall, F1, and PR-AUC, alongside a confusion matrix. For multiclass imbalance, include per-class metrics, macro F1, and balanced accuracy.

## Question 16 — Sigmoid MLP

Rows of each weight matrix correspond to receiving neurons; columns correspond to inputs. The constant-one nodes supply the biases. Reading the arrows in the diagram gives

$$
\mathbf{x}=\begin{bmatrix}0.2\\0.8\end{bmatrix},\quad
W^{(1)}=\begin{bmatrix}-0.1&-0.2\\0.6&0.4\end{bmatrix},\quad
\mathbf{b}^{(1)}=\begin{bmatrix}0.3\\0.5\end{bmatrix}.
$$

$$
W^{(2)}=\begin{bmatrix}0.5&0.1\end{bmatrix},\qquad b^{(2)}=0.3,\qquad
\sigma(z)=\frac{1}{1+e^{-z}}.
$$

The complete matrix equation, with sigmoid applied elementwise, is

$$
\hat y=\sigma\!\left(W^{(2)}\sigma\!\left(W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}\right)+b^{(2)}\right).
$$

**Hidden-layer preactivations**

$$
\mathbf{z}^{(1)}=
\begin{bmatrix}-0.1(0.2)-0.2(0.8)+0.3\\0.6(0.2)+0.4(0.8)+0.5\end{bmatrix}
=\begin{bmatrix}0.12\\0.94\end{bmatrix}.
$$

**Hidden-layer activations**

$$
\mathbf{a}^{(1)}=\sigma(\mathbf{z}^{(1)})
=\begin{bmatrix}0.529964\\0.719100\end{bmatrix}.
$$

**Output**

$$
z^{(2)}=0.5(0.529964)+0.1(0.719100)+0.3=0.636892,
$$

$$
\boxed{\hat y=\sigma(0.636892)=0.654051\approx0.6541.}
$$

**Application.** A 2-input, 2-hidden-unit, 1-sigmoid-output network naturally supports binary classification from two input features. Interpreting the output as the positive-class probability gives about 65.41%; with threshold 0.5, the prediction is class 1. The specific real-world task cannot be identified from architecture alone.

## Question 17 — Iris classification

**(a) Matrix equation**

$$
\hat{\mathbf{y}}=\operatorname{softmax}\!\left(W^{(2)}\operatorname{ReLU}\!\left(W^{(1)}\mathbf{x}+\mathbf{b}^{(1)}\right)+\mathbf{b}^{(2)}\right),
$$

where $\operatorname{ReLU}(z)=\max(0,z)$ elementwise and $\operatorname{softmax}(\mathbf{z})_i=e^{z_i}/\sum_{j=1}^{3}e^{z_j}$. Here $\mathbf{x}\in\mathbb{R}^{4}$, $W^{(1)}\in\mathbb{R}^{3\times4}$ and $W^{(2)}\in\mathbb{R}^{3\times3}$.

**(b) Step-by-step forward computation**

$$
\mathbf{z}^{(1)}=
\begin{bmatrix}
0.5&0.4&0.3&0.8\\0.2&0.4&-0.2&-0.5\\-0.9&0.2&-0.5&-0.7
\end{bmatrix}
\begin{bmatrix}4.5\\3.4\\5.1\\2.7\end{bmatrix}
+\begin{bmatrix}0.1\\0.4\\1.2\end{bmatrix}
=\begin{bmatrix}7.40\\0.29\\-6.61\end{bmatrix}.
$$

For example, the first entry is $2.25+1.36+1.53+2.16+0.10=7.40$; the second is $0.90+1.36-1.02-1.35+0.40=0.29$; the third is $-4.05+0.68-2.55-1.89+1.20=-6.61$.

$$
\mathbf{a}^{(1)}=\operatorname{ReLU}(\mathbf{z}^{(1)})=\begin{bmatrix}7.40\\0.29\\0\end{bmatrix}.
$$

$$
\mathbf{z}^{(2)}=
\begin{bmatrix}0.7&0.2&0.5\\0.4&0.9&0.8\\0.1&0.6&0.3\end{bmatrix}
\begin{bmatrix}7.40\\0.29\\0\end{bmatrix}
+\begin{bmatrix}0.7\\0.2\\0.1\end{bmatrix}
=\begin{bmatrix}5.938\\3.421\\1.014\end{bmatrix}.
$$

Subtract the largest logit for numerically stable softmax:

$$
\hat{\mathbf{y}}=\frac{1}{1+e^{-2.517}+e^{-4.924}}
\begin{bmatrix}1\\e^{-2.517}\\e^{-4.924}\end{bmatrix}
=\begin{bmatrix}0.919142\\0.074176\\0.006682\end{bmatrix}.
$$

The diagram specifies the order **[Versicolor, Setosa, Virginica]**. Thus, the probabilities are approximately **[91.9142%, 7.4176%, 0.6682%]** and the predicted class is **Versicolor**.

**(c) Cross-entropy for the true label Setosa**

Setosa is the second output, so its one-hot target is $\mathbf{y}=[0,1,0]^T$. Using natural logarithms,

$$
\mathcal{L}=-\sum_{i=1}^{3}y_i\ln\hat y_i
=-\ln(0.0741759874)=\boxed{2.601315\text{ nats}}.
$$

The loss is relatively large because the network assigns low probability to the true class.
