# Delivery Drone Navigation - Reinforcement Learning Project

## Project Overview
This repository contains a comprehensive Reinforcement Learning (RL) project focused on **Delivery Drone Navigation**. The project explores how autonomous drones can learn optimal navigation strategies in dynamic urban environments using various RL algorithms.

## 🚁 Features
- **Realistic 3D Pygame Simulation**: High-fidelity 3D visualization with physics-based movement and altitude control
- **Real-World Uncertainties**: GPS noise, sensor drift, wind gusts, turbulence, and battery uncertainty
- **Advanced Physics**: 3D wind fields, altitude-dependent drag, ground effect, and realistic battery consumption
- **Multiple RL Algorithms**: Value Iteration, Policy Iteration, Q-Learning, SARSA
- **Dynamic 3D Environment**: Weather conditions with 3D effects, obstacles at various heights, battery management
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
# Run the 3D environment with random agent
python environments/environments_realistic_drone_env.py

# Or run comprehensive tests
python test_3d_environment.py

# Or run the original test suite
python test_realistic_env.py
```

This will launch the pygame visualization with 3D perspective and altitude indicators.

## 📁 Repository Structure

```
Delivery-Drone-Navigation/
├── README.md
├── requirements.txt
├── environments/
│   ├── environments_realistic_drone_env.py  # Main 3D simulation environment
│   ├── environments_utils.py                # Helper functions for 3D navigation
│   └── __init__.py
├── algorithms/                              # RL algorithms (coming soon)
├── notebooks/                               # Jupyter notebooks
├── src/                                    # Training scripts
├── test_realistic_env.py                   # Original test scripts
└── test_3d_environment.py                  # 3D environment test suite
```

## 🎮 Environment Details

### State Space (3D with Uncertainties)
- **Position**: (x, y, z) coordinates with altitude (5-100m)
- **Battery**: 0-100% with uncertainty
- **Package Status**: No package, Has package, Delivered
- **Weather**: Clear, Windy, Rainy, Stormy (affects 3D wind field)
- **Velocity**: (vx, vy, vz) - 3D velocity vector
- **Wind Vector**: (wind_x, wind_y, wind_z) - 3D wind field
- **Orientation**: (pitch, roll, yaw) - drone attitude
- **Sensor Noise**: GPS drift and positioning errors
- **Turbulence**: Atmospheric turbulence level (0-1)

### Action Space (14 actions)
- 0-3: Cardinal directions (N, S, E, W)
- 4-7: Diagonal directions (NE, NW, SE, SW)
- 8: Ascend (increase altitude)
- 9: Descend (decrease altitude)
- 10: Hover (maintain position)
- 11: Pick up package
- 12: Deliver package
- 13: Return to base

### Realistic Physics & Uncertainties
- **3D Wind Effects**: Altitude-dependent wind with updrafts/downdrafts
- **Turbulence**: Random perturbations affecting stability
- **GPS Noise**: Gaussian noise on position readings (σ=0.5m)
- **Sensor Failures**: Rare events causing temporary sensor lag
- **Wind Gusts**: Sudden random wind changes
- **Ground Effect**: Enhanced efficiency near ground
- **Altitude-Dependent Drag**: Less drag at higher altitudes
- **Battery Uncertainty**: Variable consumption based on conditions

### Rewards
- +100: Successful delivery
- +50: Package pickup
- +20: Recharge at depot
- +2: Reaching cruise altitude (30m)
- -1: Time step penalty
- -5: Flying too low or invalid actions
- -20: Entering restricted zones
- -50: Collision with obstacles
- -100: Battery depletion

## 🌟 3D Simulation Features

### Real-World Environmental Uncertainties
The simulation now includes realistic uncertainties to better prepare RL agents for real-world deployment:

1. **GPS and Sensor Noise**: Position readings include Gaussian noise (σ=0.5m)
2. **Wind Gusts**: Random sudden wind changes (5% probability per step)
3. **Atmospheric Turbulence**: Affects stability and orientation
4. **Sensor Failures**: Rare events causing temporary data lag (1% probability)
5. **Battery Uncertainty**: Variable consumption based on multiple factors

### 3D Physics Model
- **Altitude Control**: Separate ascend/descend actions (5-100m range)
- **3D Wind Field**: Wind effects in all three dimensions with altitude dependence
- **Ground Effect**: Enhanced efficiency when flying close to ground (<10m)
- **Drag Model**: Altitude-dependent air resistance
- **Orientation Dynamics**: Realistic pitch, roll, yaw responses
- **Turbulence Effects**: Random perturbations based on weather severity

### 3D Obstacles
- **Static Obstacles**: Buildings with varying heights (25-50m)
- **Dynamic Obstacles**: Other drones and birds flying at different altitudes
- **Safety Margins**: 1.5m collision avoidance radius
- **3D Collision Detection**: Full spatial awareness

### Weather System Enhancement
Each weather condition now has 3D effects:
- **Clear**: No wind, no turbulence
- **Windy**: 3D wind vector with updrafts/downdrafts, 0.2 turbulence
- **Rainy**: Downdrafts, increased drag, 0.4 turbulence
- **Stormy**: Strong 3D winds, high turbulence (0.8), severe battery drain

### Visualization Enhancements
- **3D Perspective**: Drone size scales with altitude
- **Altitude Indicators**: Real-time altitude display and ground shadows
- **Path Coloring**: Color-coded by altitude (blue=low, yellow=high)
- **Orientation Display**: Visual pitch and roll indicators
- **Expanded UI**: Shows 3D position, orientation, turbulence, and wind magnitude

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
