"""
Test script for the realistic drone environment
Demonstrates various scenarios and capabilities
"""

import numpy as np
from environments.environments_realistic_drone_env import RealisticDroneEnv, Weather
from environments.environments_utils import create_heuristic_policy, calculate_delivery_score
import time


def test_random_agent(num_episodes: int = 5):
    """Test environment with random actions"""
    print("=" * 60)
    print("Testing Random Agent")
    print("=" * 60)
    
    env = RealisticDroneEnv(render_mode="human")
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Target Delivery: ({env.current_delivery_target.x}, {env.current_delivery_target.y})")
        
        done = False
        truncated = False
        
        while not (done or truncated):
            action = env.action_space.sample()
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.05)  # Slow down for visualization
            
            if steps % 50 == 0:
                print(f"  Step {steps}: Battery={info['battery_level']:.1f}%, "
                      f"Reward={episode_reward:.1f}")
        
        print(f"Episode finished: Steps={steps}, Total Reward={episode_reward:.1f}")
        print(f"  Successful Deliveries: {info['successful_deliveries']}")
        print(f"  Collisions: {info['collisions']}")
        print(f"  Distance Traveled: {info['total_distance']:.2f}")
    
    env.close()


def test_heuristic_agent(num_episodes: int = 5):
    """Test environment with heuristic policy"""
    print("\n" + "=" * 60)
    print("Testing Heuristic Agent")
    print("=" * 60)
    
    env = RealisticDroneEnv(render_mode="human")
    policy = create_heuristic_policy(env)
    
    for episode in range(num_episodes):
        observation, info = env.reset()
        episode_reward = 0
        steps = 0
        
        print(f"\nEpisode {episode + 1}")
        print(f"Target Delivery: ({env.current_delivery_target.x}, {env.current_delivery_target.y})")
        
        done = False
        truncated = False
        
        while not (done or truncated):
            action = policy(observation)
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.05)
            
            if steps % 50 == 0:
                print(f"  Step {steps}: Battery={info['battery_level']:.1f}%, "
                      f"Reward={episode_reward:.1f}, At Depot={info['at_depot']}")
        
        score = calculate_delivery_score(
            steps, 100 - info['battery_level'], 
            info['collisions'], info['total_distance']
        )
        
        print(f"Episode finished: Steps={steps}, Total Reward={episode_reward:.1f}")
        print(f"  Successful Deliveries: {info['successful_deliveries']}")
        print(f"  Collisions: {info['collisions']}")
        print(f"  Distance Traveled: {info['total_distance']:.2f}")
        print(f"  Performance Score: {score:.1f}")
    
    env.close()


def test_weather_scenarios():
    """Test different weather conditions"""
    print("\n" + "=" * 60)
    print("Testing Weather Scenarios")
    print("=" * 60)
    
    env = RealisticDroneEnv(render_mode="human")
    policy = create_heuristic_policy(env)
    
    weather_conditions = [Weather.CLEAR, Weather.WINDY, Weather.RAINY, Weather.STORMY]
    
    for weather in weather_conditions:
        print(f"\n--- Testing {weather.name} Weather ---")
        observation, info = env.reset()
        
        # Force specific weather
        env.weather = weather
        env._update_wind()
        
        steps = 0
        episode_reward = 0
        
        for _ in range(100):
            action = policy(observation)
            observation, reward, done, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            env.render()
            time.sleep(0.03)
            
            if done or truncated:
                break
        
        print(f"  Steps: {steps}, Reward: {episode_reward:.1f}")
        print(f"  Battery Used: {100 - info['battery_level']:.1f}%")
    
    env.close()


def test_collision_avoidance():
    """Test collision detection and avoidance"""
    print("\n" + "=" * 60)
    print("Testing Collision Detection")
    print("=" * 60)
    
    env = RealisticDroneEnv(render_mode="human")
    observation, info = env.reset()
    
    print(f"Number of static obstacles: {len(env.static_obstacles)}")
    print(f"Number of dynamic obstacles: {len(env.dynamic_obstacles)}")
    print(f"Number of restricted zones: {len(env.restricted_zones)}")
    
    # Try to navigate through obstacles
    for _ in range(300):
        # Use heuristic with some randomness
        if np.random.random() < 0.8:
            policy = create_heuristic_policy(env)
            action = policy(observation)
        else:
            action = env.action_space.sample()
        
        observation, reward, done, truncated, info = env.step(action)
        
        env.render()
        time.sleep(0.05)
        
        if info['collisions'] > 0:
            print(f"  Collision detected at step {info['time_step']}!")
        
        if done or truncated:
            print(f"\nEpisode ended after {info['time_step']} steps")
            print(f"  Total Collisions: {info['collisions']}")
            break
    
    env.close()


def test_battery_management():
    """Test battery consumption and management"""
    print("\n" + "=" * 60)
    print("Testing Battery Management")
    print("=" * 60)
    
    env = RealisticDroneEnv(render_mode="human")
    observation, info = env.reset()
    
    battery_levels = []
    actions_taken = []
    
    policy = create_heuristic_policy(env)
    
    for step in range(200):
        action = policy(observation)
        observation, reward, done, truncated, info = env.step(action)
        
        battery_levels.append(info['battery_level'])
        actions_taken.append(action)
        
        env.render()
        time.sleep(0.03)
        
        if step % 20 == 0:
            print(f"  Step {step}: Battery={info['battery_level']:.1f}%, "
                  f"Weather={info['weather']}")
        
        if done or truncated:
            break
    
    print(f