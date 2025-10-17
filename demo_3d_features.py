"""
Quick Demo of 3D Features
Run this to see the 3D environment in action with visual demonstrations
"""

import numpy as np
from environments import RealisticDroneEnv, Weather


def demo_3d_flight():
    """
    Demonstrate 3D flight capabilities including:
    - Altitude control
    - 3D wind effects
    - Turbulence
    - GPS noise
    - Altitude-dependent battery consumption
    """
    
    print("=" * 80)
    print("3D REALISTIC DRONE ENVIRONMENT DEMO")
    print("=" * 80)
    print("\nThis demo showcases the enhanced 3D simulation with real-world uncertainties")
    print("for reinforcement learning training.\n")
    
    # Create environment
    env = RealisticDroneEnv()
    
    print("Key Features:")
    print("  ✓ 3D position tracking (x, y, z) with altitude range 5-100m")
    print("  ✓ 14 actions including ascend/descend for altitude control")
    print("  ✓ 3D wind fields with altitude-dependent effects")
    print("  ✓ Realistic uncertainties: GPS noise, turbulence, wind gusts")
    print("  ✓ Altitude-dependent battery consumption")
    print("  ✓ 3D obstacle detection with safety margins")
    print()
    
    # Demo 1: Different weather conditions
    print("-" * 80)
    print("DEMO 1: Weather Effects on 3D Flight")
    print("-" * 80)
    
    for weather in [Weather.CLEAR, Weather.WINDY, Weather.RAINY, Weather.STORMY]:
        obs, info = env.reset()
        env.weather = weather
        env._update_wind()
        
        print(f"\n{weather.name} Weather:")
        print(f"  Wind (3D): ({env.wind_vector[0]:6.3f}, {env.wind_vector[1]:6.3f}, {env.wind_vector[2]:6.3f})")
        print(f"  Turbulence: {env.turbulence_level:.2f}")
        
        # Simulate hovering to see drift
        initial_pos = obs['position'].copy()
        for _ in range(10):
            obs, _, _, _, _ = env.step(10)  # Hover
        
        drift = np.linalg.norm(obs['position'] - initial_pos)
        print(f"  Position drift: {drift:.3f} units")
        print(f"  Orientation affected: pitch={obs['orientation'][0]:.1f}°, roll={obs['orientation'][1]:.1f}°")
    
    # Demo 2: Altitude effects
    print("\n" + "-" * 80)
    print("DEMO 2: Altitude-Dependent Effects")
    print("-" * 80)
    
    altitudes = [10, 30, 50, 80]
    print("\nBattery consumption at different altitudes (10 steps of horizontal flight):")
    
    for alt in altitudes:
        obs, _ = env.reset()
        env.drone_state.z = alt
        battery_before = 100.0
        
        for _ in range(10):
            obs, _, _, _, _ = env.step(2)  # Move East
        
        consumption = battery_before - obs['battery'][0]
        print(f"  {alt:2d}m altitude: {consumption:5.2f}% battery consumed")
    
    # Demo 3: 3D Navigation
    print("\n" + "-" * 80)
    print("DEMO 3: 3D Navigation Path")
    print("-" * 80)
    
    obs, _ = env.reset()
    print(f"\nStarting position: ({obs['position'][0]:.1f}, {obs['position'][1]:.1f}, {obs['position'][2]:.1f}m)")
    
    # Execute a 3D flight path
    actions = [
        (8, "Ascend to cruise altitude"),
        (8, "Continue ascending"),
        (2, "Move East"),
        (2, "Continue East"),
        (4, "Move Northeast"),
        (8, "Climb higher"),
        (9, "Descend slightly"),
        (9, "Descend more"),
        (10, "Hover")
    ]
    
    print("\nFlight path:")
    for i, (action, description) in enumerate(actions):
        obs, reward, _, _, _ = env.step(action)
        print(f"  Step {i+1}: {description:25s} -> "
              f"Pos=({obs['position'][0]:5.1f}, {obs['position'][1]:5.1f}, {obs['position'][2]:5.1f}m), "
              f"Battery={obs['battery'][0]:5.1f}%")
    
    # Demo 4: Uncertainties
    print("\n" + "-" * 80)
    print("DEMO 4: Real-World Uncertainties")
    print("-" * 80)
    
    obs, _ = env.reset()
    env.weather = Weather.STORMY
    env._update_wind()
    
    print("\nMeasured uncertainties over 20 steps in stormy weather:")
    gps_errors = []
    turbulence_values = []
    
    for i in range(20):
        obs, _, _, _, _ = env.step(10)  # Hover
        gps_error = np.linalg.norm(obs['sensor_noise'])
        gps_errors.append(gps_error)
        turbulence_values.append(obs['turbulence'][0])
    
    print(f"  GPS noise: mean={np.mean(gps_errors):.3f}m, std={np.std(gps_errors):.3f}m")
    print(f"  Turbulence: mean={np.mean(turbulence_values):.2f}, max={np.max(turbulence_values):.2f}")
    print(f"  Orientation variance: pitch σ={np.std([obs['orientation'][0] for _ in range(10)]):.1f}°")
    
    # Demo 5: 3D Obstacles
    print("\n" + "-" * 80)
    print("DEMO 5: 3D Obstacle Environment")
    print("-" * 80)
    
    obs, _ = env.reset()
    
    print(f"\nStatic obstacles (buildings):")
    for i, obstacle in enumerate(env.static_obstacles):
        print(f"  Building {i+1}: ({obstacle.x:4.1f}, {obstacle.y:4.1f}) - Height: {obstacle.depth:4.1f}m")
    
    print(f"\nDynamic obstacles (drones/birds):")
    for i, obstacle in enumerate(env.dynamic_obstacles):
        print(f"  Obstacle {i+1}: ({obstacle.x:4.1f}, {obstacle.y:4.1f}, {obstacle.z:4.1f}m) - "
              f"Velocity: ({obstacle.velocity_x:5.2f}, {obstacle.velocity_y:5.2f}, {obstacle.velocity_z:5.2f})")
    
    env.close()
    
    print("\n" + "=" * 80)
    print("DEMO COMPLETE!")
    print("=" * 80)
    print("\nThe environment is ready for RL training with:")
    print("  • Realistic 3D physics and dynamics")
    print("  • Real-world uncertainties and sensor noise")
    print("  • Complex 3D navigation challenges")
    print("  • Weather effects and environmental variability")
    print("\nRun 'python test_3d_environment.py' for comprehensive testing.")
    print("=" * 80)


if __name__ == "__main__":
    demo_3d_flight()
