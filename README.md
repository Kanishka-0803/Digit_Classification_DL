# Handwritten Digit Classification (MNIST)

Hands-on neural network intro that classifies MNIST handwritten digits
(0–9), experimenting one hyperparameter at a time (optimizer/batch
size, hidden layers, neurons, activation functions, epochs) to see
their effect on accuracy, loss, and training time.

## Dataset

MNIST: 70,000 grayscale 28x28 images of digits 0–9, pixel values 0–255.

## Requirements

```bash
pip install tensorflow==2.18.0 scikit-learn==1.3.2 matplotlib==3.8.3 \
    seaborn==0.13.2 numpy==1.26.4 pandas==2.2.2
```

Built for Google Colab but runs in any standard Jupyter environment.

## Structure

1. **Data overview & preprocessing** — scaling, reshaping, one-hot
   encoding labels
2. **Models 0–13** — incremental experiments: batch GD vs. SGD, batch
   size, hidden layers/neurons, activation functions, epoch count
3. **Model comparison table** across all 14 models
4. **Testing on new inputs** and final conclusion

## Key Findings

- Switching from full-batch gradient descent to mini-batch SGD
  roughly doubled accuracy
- Activation function choice had the biggest single impact (sigmoid →
  tanh gave the largest jump, ~4%)
- Adding hidden layers gave diminishing returns
- Training too long overfits: 100% train accuracy but flat validation
  accuracy on the deepest model

## Final Model

**Model 10** (ReLU(128) → tanh(64), two hidden layers) — best
validation accuracy (0.9750), lowest validation loss, and least
overfitting of all models tested.
