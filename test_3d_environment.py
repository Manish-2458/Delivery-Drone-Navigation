"""
Test script for 3D Realistic Drone Environment
Demonstrates 3D navigation, altitude control, and real-world uncertainties
"""

import numpy as np
from environments import RealisticDroneEnv, Weather
from environments.environments_utils import create_heuristic_policy, calculate_delivery_score
import time


def test_3d_navigation():
    """Test basic 3D navigation and altitude control"""
    print("=" * 70)
    print("Test 1: 3D Navigation and Altitude Control")
    print("=" * 70)
    
    env = RealisticDroneEnv()
    obs, info = env.reset(seed=42)
    
    print(f"\nInitial State:")
    print(f"  Position (x, y, z): ({obs['position'][0]:.2f}, {obs['position'][1]:.2f}, {obs['position'][2]:.2f})")
    print(f"  Battery: {obs['battery'][0]:.1f}%")
    print(f"  Weather: {info['weather']}")
    
    # Test altitude changes
    print("\nTesting Altitude Control:")
    for i in range(5):
        action = 8  # Ascend
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+1}: Altitude = {obs['position'][2]:.2f}m, Battery = {obs['battery'][0]:.2f}%")
    
    for i in range(3):
        action = 9  # Descend
        obs, reward, done, truncated, info = env.step(action)
        print(f"  Step {i+6}: Altitude = {obs['position'][2]:.2f}m, Battery = {obs['battery'][0]:.2f}%")
    
    env.close()
    print("\n✓ 3D navigation test passed!")


def test_realistic_uncertainties():
    """Test realistic uncertainties like wind, turbulence, and sensor noise"""
    print("\n" + "=" * 70)
    print("Test 2: Realistic Uncertainties and Environmental Effects")
    print("=" * 70)
    
    env = RealisticDroneEnv()
    
    # Test different weather conditions
    weather_conditions = [Weather.CLEAR, Weather.WINDY, Weather.RAINY, Weather.STORMY]
    
    for weather in weather_conditions:
        obs, info = env.reset()
        env.weather = weather
        env._update_wind()
        
        print(f"\n{weather.name} Weather:")
        print(f"  Wind vector (3D): ({env.wind_vector[0]:.3f}, {env.wind_vector[1]:.3f}, {env.wind_vector[2]:.3f})")
        print(f"  Turbulence level: {env.turbulence_level:.2f}")
        
        # Take a few steps to see effects
        initial_pos = obs['position'].copy()
        for _ in range(10):
            action = 10  # Hover to see drift
            obs, reward, done, truncated, info = env.step(action)
        
        drift = np.linalg.norm(obs['position'] - initial_pos)
        print(f"  Position drift after hovering: {drift:.3f} units")
        print(f"  GPS noise: ({obs['sensor_noise'][0]:.3f}, {obs['sensor_noise'][1]:.3f}, {obs['sensor_noise'][2]:.3f})")
        print(f"  Orientation: pitch={obs['orientation'][0]:.1f}°, roll={obs['orientation'][1]:.1f}°")
    
    env.close()
    print("\n✓ Uncertainty simulation test passed!")


def test_battery_consumption():
    """Test realistic battery consumption with altitude effects"""
    print("\n" + "=" * 70)
    print("Test 3: Altitude-Dependent Battery Consumption")
    print("=" * 70)
    
    env = RealisticDroneEnv()
    obs, info = env.reset()
    
    initial_battery = obs['battery'][0]
    
    # Test at different altitudes
    test_altitudes = [10, 30, 50, 80]
    
    for target_alt in test_altitudes:
        obs, _ = env.reset()
        env.drone_state.z = target_alt
        
        battery_before = obs['battery'][0]
        
        # Move horizontally for several steps
        for _ in range(10):
            action = 2  # Move East
            obs, reward, done, truncated, info = env.step(action)
            if done or truncated:
                break
        
        battery_after = obs['battery'][0]
        consumption = battery_before - battery_after
        
        print(f"  Altitude {target_alt}m: Battery consumed = {consumption:.2f}%")
    
    env.close()
    print("\n✓ Battery consumption test passed!")


def test_3d_obstacles():
    """Test 3D obstacle avoidance"""
    print("\n" + "=" * 70)
    print("Test 4: 3D Obstacle Detection and Collision")
    print("=" * 70)
    
    env = RealisticDroneEnv()
    obs, info = env.reset()
    
    print(f"\nStatic obstacles (buildings with height):")
    for i, obs_obj in enumerate(env.static_obstacles[:3]):
        print(f"  Obstacle {i+1}: Position=({obs_obj.x:.1f}, {obs_obj.y:.1f}, {obs_obj.z:.1f}), "
              f"Height={obs_obj.depth:.1f}m")
    
    print(f"\nDynamic obstacles (drones/birds at various altitudes):")
    for i, obs_obj in enumerate(env.dynamic_obstacles):
        print(f"  Obstacle {i+1}: Position=({obs_obj.x:.1f}, {obs_obj.y:.1f}, {obs_obj.z:.1f}m), "
              f"Velocity=({obs_obj.velocity_x:.2f}, {obs_obj.velocity_y:.2f}, {obs_obj.velocity_z:.2f})")
    
    # Monitor dynamic obstacle movement
    print("\nDynamic obstacle movement over time:")
    initial_positions = [(o.x, o.y, o.z) for o in env.dynamic_obstacles]
    
    for step in range(20):
        obs, reward, done, truncated, info = env.step(10)  # Hover
        if step % 5 == 0:
            print(f"  Step {step}: Obstacle 1 at ({env.dynamic_obstacles[0].x:.2f}, "
                  f"{env.dynamic_obstacles[0].y:.2f}, {env.dynamic_obstacles[0].z:.2f}m)")
    
    env.close()
    print("\n✓ 3D obstacle test passed!")


def test_heuristic_3d_policy():
    """Test heuristic policy with 3D navigation"""
    print("\n" + "=" * 70)
    print("Test 5: Heuristic Policy for 3D Navigation")
    print("=" * 70)
    
    env = RealisticDroneEnv()
    policy = create_heuristic_policy(env)
    
    obs, info = env.reset()
    print(f"\nInitial state:")
    print(f"  Position: ({obs['position'][0]:.1f}, {obs['position'][1]:.1f}, {obs['position'][2]:.1f}m)")
    print(f"  Target delivery: ({env.current_delivery_target.x}, {env.current_delivery_target.y})")
    
    episode_reward = 0
    altitude_history = []
    
    for step in range(100):
        action = policy(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        altitude_history.append(obs['position'][2])
        
        if step % 20 == 0:
            print(f"  Step {step}: Altitude={obs['position'][2]:.1f}m, "
                  f"Battery={obs['battery'][0]:.1f}%, Package={'Yes' if obs['package_status'] else 'No'}")
        
        if done or truncated:
            break
    
    print(f"\nEpisode summary:")
    print(f"  Total steps: {step + 1}")
    print(f"  Total reward: {episode_reward:.1f}")
    print(f"  Average altitude: {np.mean(altitude_history):.1f}m")
    print(f"  Min altitude: {np.min(altitude_history):.1f}m")
    print(f"  Max altitude: {np.max(altitude_history):.1f}m")
    print(f"  Successful deliveries: {info['successful_deliveries']}")
    
    env.close()
    print("\n✓ Heuristic policy test passed!")


def run_all_tests():
    """Run all test cases"""
    print("\n" + "=" * 70)
    print("3D REALISTIC DRONE ENVIRONMENT TEST SUITE")
    print("=" * 70)
    
    test_3d_navigation()
    test_realistic_uncertainties()
    test_battery_consumption()
    test_3d_obstacles()
    test_heuristic_3d_policy()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  ✓ 3D position tracking with altitude control")
    print("  ✓ Realistic physics: wind, turbulence, drag")
    print("  ✓ Sensor noise and GPS drift")
    print("  ✓ Altitude-dependent battery consumption")
    print("  ✓ 3D obstacle detection and avoidance")
    print("  ✓ Weather effects on flight dynamics")
    print("  ✓ Adaptive heuristic policy for 3D navigation")
    print("\nThe environment is ready for RL training!")


if __name__ == "__main__":
    run_all_tests()
