# 💰 Support This Research - Please Donate!

**🙏 If this library helps your research or project, please consider donating to support continued development:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

[![CI](https://github.com/benedictchen/universal-learning/workflows/CI/badge.svg)](https://github.com/benedictchen/universal-learning/actions)
[![PyPI version](https://badge.fury.io/py/universal-learning.svg)](https://badge.fury.io/py/universal-learning)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Custom%20Non--Commercial-red.svg)](LICENSE)

---

# Universal Learning

🧠 Solomonoff induction and AIXI for universal artificial intelligence

**Solomonoff, R. J. (1964)** - "A formal theory of inductive inference"  
**Hutter, M. (2005)** - "Universal Artificial Intelligence: Sequential Decisions Based on Algorithmic Probability"

## 📦 Installation

```bash
pip install universal-learning
```

## 🚀 Quick Start

### Solomonoff Induction Example
```python
from universal_learning import SolomonoffInductor
import numpy as np

# Create Solomonoff inductor
inductor = SolomonoffInductor(
    max_program_length=100,
    universal_machine='utm',
    approximation_method='jtw'  # Jürgen's Time-Weighted approximation
)

# Binary sequence prediction
sequence = [0, 1, 0, 1, 0, 1]  # Simple alternating pattern
prediction = inductor.predict_next(sequence)
print(f"Next bit prediction: {prediction}")

# Get probability distribution
probs = inductor.get_probabilities(sequence)
print(f"P(next=0): {probs[0]:.4f}, P(next=1): {probs[1]:.4f}")

# Sequence completion
partial_seq = [1, 1, 0, 1]
completions = inductor.complete_sequence(partial_seq, max_length=10)
print("Most likely completions:", completions[:3])
```

### AIXI Agent Example
```python
from universal_learning import AIXI
import numpy as np

# Create AIXI agent for simple environment
agent = AIXI(
    action_space_size=4,
    observation_space_size=8,
    horizon=10,
    approximation='ctx',  # Context Tree Weighting
    exploration_factor=0.1
)

# Simple interaction loop
total_reward = 0
for step in range(100):
    # Agent selects action based on current beliefs
    action = agent.select_action()
    
    # Environment responds (example: simple reward function)
    observation = env.step(action)  # Your environment
    reward = env.get_reward()
    
    # Agent updates its world model
    agent.update(action, observation, reward)
    total_reward += reward

print(f"Total reward: {total_reward}")
print(f"Learned model complexity: {agent.get_model_complexity()}")
```

### Kolmogorov Complexity Estimation
```python
from universal_learning import KolmogorovComplexity

# Estimate algorithmic complexity
kc = KolmogorovComplexity(
    reference_machine='utm',
    approximation_method='lzw'
)

# Analyze different sequences
sequences = [
    [0, 0, 0, 0, 0, 0, 0, 0],  # Regular pattern
    [0, 1, 0, 1, 0, 1, 0, 1],  # Alternating pattern  
    [1, 0, 1, 1, 0, 0, 1, 0],  # Complex pattern
    np.random.randint(0, 2, 100)  # Random sequence
]

for i, seq in enumerate(sequences):
    complexity = kc.estimate_complexity(seq)
    normalized = kc.normalize_complexity(seq)
    print(f"Sequence {i+1}: K(x) ≈ {complexity:.2f}, normalized: {normalized:.4f}")
```

## 🧬 Advanced Features

### Universal Turing Machine Simulation
```python
from universal_learning.solomonoff_modules import UniversalTuringMachine

# Create UTM with custom instruction set
utm = UniversalTuringMachine(
    tape_size=1000,
    instruction_set='binary',
    halt_detection=True
)

# Run programs and measure complexity
programs = [
    "01010101",  # Simple alternating output
    "001100110011",  # Periodic pattern
    utm.generate_random_program(50)  # Random program
]

for prog in programs:
    try:
        output = utm.run_program(prog, max_steps=1000)
        steps = utm.get_execution_steps()
        print(f"Program: {prog[:20]}...")
        print(f"Output: {output[:50]}...")
        print(f"Steps: {steps}")
    except utm.HaltException:
        print("Program halted")
```

### Context Tree Weighting (CTW)
```python
from universal_learning import ContextTreeWeighting

# Efficient sequence prediction using CTW
ctw = ContextTreeWeighting(
    alphabet_size=2,
    max_depth=8,
    beta=0.5  # Mixing parameter
)

# Online learning and prediction
sequence = [1, 0, 1, 1, 0, 0, 1, 0, 1, 1]
predictions = []

for i, symbol in enumerate(sequence):
    if i > 0:  # Need context for prediction
        pred = ctw.predict_next()
        predictions.append(pred)
    
    ctw.update(symbol)

print("Predictions:", predictions)
print("Accuracy:", np.mean(np.array(predictions) == np.array(sequence[1:])))
```

## 🔬 Research Foundation

This implementation provides research-accurate implementations of:

- **Solomonoff Induction**: Ray Solomonoff's theory of universal inductive inference
- **AIXI Framework**: Marcus Hutter's universal artificial intelligence agent
- **Kolmogorov Complexity**: Algorithmic information theory and computational complexity
- **Universal Turing Machines**: Theoretical foundations of computation

### Key Theoretical Components
- **Algorithmic Probability**: P(x) = Σ 2^(-|p|) over all programs p that output x
- **Universal Prior**: Theoretically optimal prior for inductive inference  
- **Sequence Prediction**: Optimal prediction using algorithmic probability
- **Reinforcement Learning**: AIXI agent for general environment interaction

## 📊 Implementation Highlights

- **Research Accuracy**: Faithful implementation of theoretical foundations
- **Approximation Methods**: Practical approximations for intractable exact computation
- **Educational Value**: Clear code structure for understanding fundamental concepts
- **Extensible Framework**: Easy to modify for research applications
- **Performance Optimized**: Efficient implementations of complex algorithms

## 🧮 Approximation Methods

Since exact Solomonoff induction is uncomputable, we provide several approximations:

### For Solomonoff Induction
- **Length-based Weighting**: Weight programs by inverse length
- **Time-bounded Computation**: Limit UTM execution time
- **Context Tree Weighting**: Efficient practical approximation
- **Jürgen Schmidhuber's Speed Prior**: Consider program runtime

### For AIXI
- **Monte Carlo AIXI**: Sampling-based approximation
- **Context Tree Weighting AIXI**: CTW-based world model
- **Feature AIXI**: Hand-crafted feature approximation
- **Neural AIXI**: Neural network approximations

## 🎓 About the Implementation

Implemented by **Benedict Chen** - bringing foundational AI research to modern Python.

📧 Contact: benedict@benedictchen.com

---

## 💰 Support This Work - Donation Appreciated!

**This implementation represents hundreds of hours of research and development. If you find it valuable, please consider donating:**

**[💳 DONATE VIA PAYPAL - CLICK HERE](https://www.paypal.com/cgi-bin/webscr?cmd=_s-xclick&hosted_button_id=WXQKYYKPHWXHS)**

**Your support helps maintain and expand these research implementations! 🙏**