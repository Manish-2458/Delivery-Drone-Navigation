"""
Utility functions for the drone environment
"""

import numpy as np
from typing import Tuple, List
import math


def euclidean_distance(pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two positions"""
    return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)


def manhattan_distance(pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_direction_vector(from_pos: Tuple[float, float], to_pos: Tuple[float, float]) -> Tuple[float, float]:
    """Get normalized direction vector from one position to another"""
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    distance = math.sqrt(dx**2 + dy**2)
    
    if distance == 0:
        return (0.0, 0.0)
    
    return (dx / distance, dy / distance)


def get_optimal_action(current_pos: Tuple[int, int], target_pos: Tuple[int, int]) -> int:
    """
    Get optimal action to move from current position to target position
    
    Actions:
    0: North, 1: South, 2: East, 3: West
    4: NE, 5: NW, 6: SE, 7: SW
    8: Hover
    """
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    
    if abs(dx) < 0.5 and abs(dy) < 0.5:
        return 8  # Hover
    
    # Determine primary direction
    if abs(dx) > abs(dy):
        # Horizontal movement dominant
        if dx > 0:
            if dy > 0:
                return 6  # Southeast
            elif dy < 0:
                return 4  # Northeast
            else:
                return 2  # East
        else:
            if dy > 0:
                return 7  # Southwest
            elif dy < 0:
                return 5  # Northwest
            else:
                return 3  # West
    else:
        # Vertical movement dominant
        if dy > 0:
            if dx > 0:
                return 6  # Southeast
            elif dx < 0:
                return 7  # Southwest
            else:
                return 1  # South
        else:
            if dx > 0:
                return 4  # Northeast
            elif dx < 0:
                return 5  # Northwest
            else:
                return 0  # North


def calculate_path_length(path: List[Tuple[int, int]]) -> float:
    """Calculate total path length"""
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        total_length += euclidean_distance(path[i], path[i+1])
    
    return total_length


def is_path_clear(start: Tuple[int, int], end: Tuple[int, int], 
                  obstacles: List[Tuple[int, int, int, int]]) -> bool:
    """
    Check if path from start to end is clear of obstacles
    
    Args:
        start: Starting position (x, y)
        end: Ending position (x, y)
        obstacles: List of obstacles as (x, y, width, height)
    
    Returns:
        True if path is clear, False otherwise
    """
    # Use Bresenham's line algorithm to check intermediate points
    x0, y0 = start
    x1, y1 = end
    
    points = bresenham_line(x0, y0, x1, y1)
    
    for point in points:
        for obs in obstacles:
            obs_x, obs_y, obs_w, obs_h = obs
            if (obs_x <= point[0] < obs_x + obs_w and 
                obs_y <= point[1] < obs_y + obs_h):
                return False
    
    return True


def bresenham_line(x0: int, y0: int, x1: int, y1: int) -> List[Tuple[int, int]]:
    """
    Bresenham's line algorithm to get all points on a line
    
    Returns:
        List of (x, y) coordinates on the line
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    x, y = x0, y0
    
    while True:
        points.append((x, y))
        
        if x == x1 and y == y1:
            break
        
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
    
    return points


def calculate_battery_efficiency(distance: float, battery_used: float) -> float:
    """Calculate battery efficiency (distance per unit battery)"""
    if battery_used == 0:
        return 0.0
    return distance / battery_used


def estimate_required_battery(distance: float, weather_factor: float = 1.0, 
                             has_package: bool = False) -> float:
    """
    Estimate battery required for a given distance
    
    Args:
        distance: Distance to travel
        weather_factor: Weather impact multiplier (1.0 = clear, 1.5 = windy, 2.0 = rainy)
        has_package: Whether carrying a package
    
    Returns:
        Estimated battery percentage required
    """
    base_consumption = 0.5  # Base consumption per unit distance
    
    total_consumption = base_consumption * distance * weather_factor
    
    if has_package:
        total_consumption *= 1.2  # 20% more with package
    
    return total_consumption


def get_weather_impact_factor(weather_code: int) -> float:
    """
    Get weather impact factor
    
    Args:
        weather_code: 0=Clear, 1=Windy, 2=Rainy, 3=Stormy
    
    Returns:
        Impact factor multiplier
    """
    weather_factors = {
        0: 1.0,   # Clear
        1: 1.5,   # Windy
        2: 2.0,   # Rainy
        3: 3.0,   # Stormy
    }
    return weather_factors.get(weather_code, 1.0)


def calculate_delivery_score(delivery_time: int, battery_used: float, 
                            collisions: int, distance: float) -> float:
    """
    Calculate overall delivery performance score
    
    Args:
        delivery_time: Time steps taken
        battery_used: Battery percentage consumed
        collisions: Number of collisions
        distance: Total distance traveled
    
    Returns:
        Performance score (higher is better)
    """
    # Base score
    score = 1000.0
    
    # Time penalty (prefer faster deliveries)
    score -= delivery_time * 2
    
    # Battery efficiency bonus
    if distance > 0:
        efficiency = distance / battery_used if battery_used > 0 else 0
        score += efficiency * 10
    
    # Collision penalty
    score -= collisions * 100
    
    # Distance penalty (prefer shorter paths)
    score -= distance * 5
    
    return max(0, score)


def normalize_observation(obs: dict, grid_size: int) -> dict:
    """
    Normalize observation values to [0, 1] range
    
    Args:
        obs: Raw observation dictionary
        grid_size: Size of the grid
    
    Returns:
        Normalized observation dictionary
    """
    normalized = {}
    
    # Normalize position
    if 'position' in obs:
        normalized['position'] = obs['position'] / (grid_size - 1)
    
    # Battery already in 0-100 range, normalize to 0-1
    if 'battery' in obs:
        normalized['battery'] = obs['battery'] / 100.0
    
    # Package status already 0-2
    if 'package_status' in obs:
        normalized['package_status'] = obs['package_status'] / 2.0
    
    # Weather already 0-3
    if 'weather' in obs:
        normalized['weather'] = obs['weather'] / 3.0
    
    # Time normalization (assume max 500 steps)
    if 'time' in obs:
        normalized['time'] = np.clip(obs['time'] / 500.0, 0, 1)
    
    # Velocity already in reasonable range
    if 'velocity' in obs:
        normalized['velocity'] = np.clip((obs['velocity'] + 2) / 4, 0, 1)
    
    # Wind already in reasonable range
    if 'wind' in obs:
        normalized['wind'] = np.clip((obs['wind'] + 1) / 2, 0, 1)
    
    return normalized


def create_heuristic_policy(env) -> callable:
    """
    Create a simple heuristic policy for the environment
    
    Returns:
        Policy function that takes observation and returns action
    """
    def policy(obs):
        """
        Heuristic policy:
        1. If no package and at depot, hover
        2. If no package and not at depot, go to depot
        3. If has package, go to delivery point
        4. If low battery, return to depot
        """
        position = obs['position']
        battery = obs['battery'][0]
        has_package = obs['package_status']
        
        depot = env.depot_location
        current_pos = (int(position[0]), int(position[1]))
        
        # Low battery - return to depot
        if battery < 30 and not has_package:
            return get_optimal_action(current_pos, depot)
        
        # Has package - go to delivery
        if has_package and env.current_delivery_target:
            target = (env.current_delivery_target.x, env.current_delivery_target.y)
            
            # If at delivery point, deliver
            if euclidean_distance(current_pos, target) < 1:
                return 10  # Deliver action
            
            return get_optimal_action(current_pos, target)
        
        # No package - go to depot
        if not has_package:
            # If at depot, pick up package
            if euclidean_distance(current_pos, depot) < 1:
                return 9  # Pick up action
            
            return get_optimal_action(current_pos, depot)
        
        # Default: hover
        return 8
    
    return policy


def generate_random_delivery_points(num_points: int, grid_size: int, 
                                    obstacles: List, depot: Tuple[int, int],
                                    seed: int = None) -> List:
    """
    Generate random delivery points avoiding obstacles
    
    Args:
        num_points: Number of delivery points to generate
        grid_size: Size of the grid
        obstacles: List of obstacles
        depot: Depot location to avoid
        seed: Random seed
    
    Returns:
        List of delivery point dictionaries
    """
    if seed is not None:
        np.random.seed(seed)
    
    delivery_points = []
    attempts = 0
    max_attempts = num_points * 100
    
    while len(delivery_points) < num_points and attempts < max_attempts:
        x = np.random.randint(2, grid_size - 2)
        y = np.random.randint(2, grid_size - 2)
        
        # Check if position is valid
        valid = True
        
        # Check depot
        if euclidean_distance((x, y), depot) < 2:
            valid = False
        
        # Check obstacles
        for obs in obstacles:
            if hasattr(obs, 'x'):
                if (obs.x <= x < obs.x + obs.width and 
                    obs.y <= y < obs.y + obs.height):
                    valid = False
                    break
        
        # Check other delivery points
        for point in delivery_points:
            if euclidean_distance((x, y), (point['x'], point['y'])) < 3:
                valid = False
                break
        
        if valid:
            priority = np.random.randint(1, 6)
            reward = 50 + priority * 10
            time_window = (0, np.random.randint(200, 500))
            
            delivery_points.append({
                'x': x,
                'y': y,
                'priority': priority,
                'reward': reward,
                'time_window': time_window
            })
        
        attempts += 1
    
    return delivery_points