# Delivery Drone Navigation - Reinforcement Learning Project

## Project Overview
This repository contains a comprehensive Reinforcement Learning (RL) project focused on **Delivery Drone Navigation**. The project explores how autonomous drones can learn optimal navigation strategies in dynamic urban environments using various RL algorithms.

## 🚁 Features
- **Realistic Pygame Simulation**: High-fidelity visualization with physics-based movement
- **Multiple RL Algorithms**: Value Iteration, Policy Iteration, Q-Learning, SARSA
- **Dynamic Environment**: Weather conditions, obstacles, battery management
- **Custom Gymnasium Environment**: Fully compatible with standard RL frameworks
- **Comprehensive Analysis**: Training metrics, visualizations, and performance tracking

## 🎯 Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/Manish-2458/Delivery-Drone-Navigation.git
cd Delivery-Drone-Navigation
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Run Simulation

```bash
python environments/realistic_drone_env.py
```

This will launch the pygame visualization with a random agent.

## 📁 Repository Structure

```
Delivery-Drone-Navigation/
├── README.md
├── requirements.txt
├── environments/
│   ├── realistic_drone_env.py    # Main simulation environment
│   ├── utils.py                   # Helper functions
│   └── __init__.py
├── algorithms/                     # RL algorithms (coming soon)
├── notebooks/                      # Jupyter notebooks
├── src/                           # Training scripts
└── test_realistic_env.py          # Test scripts
```

## 🎮 Environment Details

### State Space
- **Position**: (x, y) coordinates
- **Battery**: 0-100%
- **Package Status**: No package, Has package, Delivered
- **Weather**: Clear, Windy, Rainy, Stormy
- **Velocity**: (vx, vy)
- **Wind Vector**: (wind_x, wind_y)

### Action Space (12 actions)
- 0-3: Cardinal directions (N, S, E, W)
- 4-7: Diagonal directions (NE, NW, SE, SW)
- 8: Hover
- 9: Pick up package
- 10: Deliver package
- 11: Return to base

### Rewards
- +100: Successful delivery
- +50: Package pickup
- +20: Recharge at depot
- -1: Time step penalty
- -50: Collision
- -100: Battery depletion

## 👥 Contributors
- **Manish-2458** - Lead Developer
- **Avishek8136** - Implementation & Development

## 📄 License
MIT License

## 🔗 Links
- Repository: https://github.com/Manish-2458/Delivery-Drone-Navigation
- Issues: https://github.com/Manish-2458/Delivery-Drone-Navigation/issues

---
**Status**: 🚧 Active Development
