"""
Realistic Delivery Drone Navigation Environment
Implements a high-fidelity simulation with pygame visualization
"""

import gymnasium as gym
import numpy as np
import pygame
from gymnasium import spaces
from typing import Tuple, Dict, Optional, List
from enum import Enum
from dataclasses import dataclass
import math

# Constants
WINDOW_WIDTH = 1200
WINDOW_HEIGHT = 800
GRID_SIZE = 20
CELL_SIZE = 40
FPS = 30

# 3D Environment Constants
MIN_ALTITUDE = 5.0  # Minimum safe altitude in meters
MAX_ALTITUDE = 100.0  # Maximum altitude in meters
CRUISE_ALTITUDE = 30.0  # Optimal cruise altitude
GROUND_EFFECT_HEIGHT = 10.0  # Height where ground effect matters

# Physics Constants
GRAVITY = 9.81  # m/s^2
AIR_DENSITY = 1.225  # kg/m^3 at sea level
DRAG_COEFFICIENT = 0.5
MASS = 2.0  # kg (drone + package)

# Colors (RGB)
COLORS = {
    'sky_clear': (135, 206, 235),
    'sky_cloudy': (169, 169, 169),
    'sky_rainy': (105, 105, 105),
    'ground': (34, 139, 34),
    'road': (128, 128, 128),
    'building': (139, 90, 43),
    'restricted': (220, 20, 60),
    'depot': (0, 0, 255),
    'delivery': (255, 215, 0),
    'drone': (255, 0, 0),
    'battery_high': (0, 255, 0),
    'battery_medium': (255, 255, 0),
    'battery_low': (255, 165, 0),
    'battery_critical': (255, 0, 0),
    'package': (160, 82, 45),
    'wind': (200, 200, 255),
    'obstacle': (64, 64, 64),
    'path': (255, 255, 255),
    'text': (255, 255, 255),
    'text_bg': (0, 0, 0),
}


class Weather(Enum):
    """Weather conditions affecting drone navigation"""
    CLEAR = 0
    WINDY = 1
    RAINY = 2
    STORMY = 3


class BatteryLevel(Enum):
    """Battery status"""
    HIGH = 3      # 75-100%
    MEDIUM = 2    # 50-75%
    LOW = 1       # 25-50%
    CRITICAL = 0  # 0-25%


class PackageStatus(Enum):
    """Package delivery status"""
    NO_PACKAGE = 0
    HAS_PACKAGE = 1
    DELIVERED = 2


@dataclass
class DroneState:
    """Comprehensive drone state"""
    x: float
    y: float
    z: float  # Altitude in meters (0-100)
    battery: float  # 0-100
    has_package: bool
    velocity_x: float
    velocity_y: float
    velocity_z: float  # Vertical velocity
    pitch: float  # Pitch angle in degrees
    roll: float  # Roll angle in degrees
    yaw: float  # Yaw angle in degrees


@dataclass
class DeliveryPoint:
    """Delivery location"""
    x: int
    y: int
    priority: int  # 1-5, higher is more urgent
    reward: float
    time_window: Tuple[int, int]  # (start, end) steps


@dataclass
class Obstacle:
    """Dynamic or static obstacle with 3D properties"""
    x: float
    y: float
    z: float  # Altitude
    width: float
    height: float
    depth: float  # Vertical extent
    is_dynamic: bool
    velocity_x: float = 0
    velocity_y: float = 0
    velocity_z: float = 0


class RealisticDroneEnv(gym.Env):
    """
    Realistic 3D Drone Delivery Environment with Pygame Visualization
    
    State Space:
        - Position (x, y, z): 3D continuous coordinates with altitude
        - Battery level: 0-100%
        - Package status: {0, 1, 2}
        - Weather: {0, 1, 2, 3}
        - Time: continuous
        - Velocity: (vx, vy, vz)
        - Wind: (wind_x, wind_y, wind_z) - 3D wind field
        - Orientation: (pitch, roll, yaw)
        - Sensor noise: GPS drift and positioning errors
        - Turbulence: atmospheric turbulence level
        
    Action Space:
        - 0: Move North
        - 1: Move South
        - 2: Move East
        - 3: Move West
        - 4: Move Northeast
        - 5: Move Northwest
        - 6: Move Southeast
        - 7: Move Southwest
        - 8: Ascend (increase altitude)
        - 9: Descend (decrease altitude)
        - 10: Hover (maintain position)
        - 11: Pick up package
        - 12: Deliver package
        - 13: Return to base
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = GRID_SIZE):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 14 discrete actions (added altitude control)
        self.action_space = spaces.Discrete(14)
        
        # Observation space: continuous 3D state with uncertainties
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=grid_size-1, shape=(3,), dtype=np.float32),  # x, y, z
            'battery': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'package_status': spaces.Discrete(3),
            'weather': spaces.Discrete(4),
            'time': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'velocity': spaces.Box(low=-5, high=5, shape=(3,), dtype=np.float32),  # vx, vy, vz
            'wind': spaces.Box(low=-2, high=2, shape=(3,), dtype=np.float32),  # 3D wind
            'orientation': spaces.Box(low=-180, high=180, shape=(3,), dtype=np.float32),  # pitch, roll, yaw
            'sensor_noise': spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32),  # GPS error
            'turbulence': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
        })
        
        # Initialize pygame
        self.window = None
        self.clock = None
        self.font = None
        self.small_font = None
        
        # Environment parameters
        self.max_steps = 500
        self.battery_consumption_base = 0.5
        self.battery_consumption_windy = 1.0
        self.battery_consumption_rainy = 1.5
        
        # Depot and delivery locations
        self.depot_location = (1, 1)
        self.delivery_points: List[DeliveryPoint] = []
        self.current_delivery_target: Optional[DeliveryPoint] = None
        
        # Obstacles
        self.static_obstacles: List[Obstacle] = []
        self.dynamic_obstacles: List[Obstacle] = []
        self.restricted_zones: List[Tuple[int, int, int, int]] = []
        
        # Drone state
        self.drone_state: Optional[DroneState] = None
        
        # Environment state
        self.weather = Weather.CLEAR
        self.wind_vector = (0.0, 0.0, 0.0)  # 3D wind field
        self.turbulence_level = 0.0
        self.time_step = 0
        
        # Uncertainty and noise parameters
        self.gps_noise_std = 0.5  # GPS position error in meters
        self.sensor_failure_prob = 0.01  # Probability of temporary sensor failure
        self.wind_gust_prob = 0.05  # Probability of sudden wind gust
        self.battery_uncertainty = 0.02  # Battery reading uncertainty
        
        # Statistics
        self.total_deliveries = 0
        self.successful_deliveries = 0
        self.collisions = 0
        self.total_distance = 0.0
        
        # Path tracking for visualization
        self.path_history: List[Tuple[int, int]] = []
        
        self._setup_environment()
    
    def _setup_environment(self):
        """Initialize environment layout"""
        # Create delivery points
        self.delivery_points = [
            DeliveryPoint(x=18, y=5, priority=5, reward=100, time_window=(0, 200)),
            DeliveryPoint(x=15, y=15, priority=3, reward=80, time_window=(50, 300)),
            DeliveryPoint(x=5, y=18, priority=4, reward=90, time_window=(100, 400)),
        ]
        
        # Create static obstacles (buildings with height)
        self.static_obstacles = [
            Obstacle(x=5, y=5, z=0, width=3, height=3, depth=40, is_dynamic=False),  # Tall building
            Obstacle(x=10, y=8, z=0, width=2, height=4, depth=50, is_dynamic=False),  # Skyscraper
            Obstacle(x=15, y=3, z=0, width=2, height=2, depth=25, is_dynamic=False),  # Medium building
            Obstacle(x=8, y=14, z=0, width=3, height=2, depth=35, is_dynamic=False),  # Office building
            Obstacle(x=12, y=12, z=0, width=2, height=3, depth=30, is_dynamic=False),  # Apartment
            Obstacle(x=3, y=7, z=0, width=2, height=2, depth=45, is_dynamic=False),  # Tower
        ]
        
        # Create dynamic obstacles (other drones, birds flying at various altitudes)
        self.dynamic_obstacles = [
            Obstacle(x=7, y=7, z=20, width=1, height=1, depth=2, is_dynamic=True, 
                    velocity_x=0.1, velocity_y=0.05, velocity_z=0.02),  # Drone
            Obstacle(x=13, y=10, z=35, width=1, height=1, depth=2, is_dynamic=True, 
                    velocity_x=-0.08, velocity_y=0.12, velocity_z=-0.03),  # High-flying drone
            Obstacle(x=5, y=12, z=15, width=0.5, height=0.5, depth=1, is_dynamic=True,
                    velocity_x=0.15, velocity_y=-0.1, velocity_z=0.05),  # Bird
        ]
        
        # Restricted zones (no-fly zones)
        self.restricted_zones = [
            (10, 1, 3, 3),  # (x, y, width, height)
            (3, 10, 2, 2),
        ]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset drone state with 3D position
        self.drone_state = DroneState(
            x=float(self.depot_location[0]),
            y=float(self.depot_location[1]),
            z=MIN_ALTITUDE,  # Start at minimum safe altitude
            battery=100.0,
            has_package=False,
            velocity_x=0.0,
            velocity_y=0.0,
            velocity_z=0.0,
            pitch=0.0,
            roll=0.0,
            yaw=0.0
        )
        
        # Reset environment
        self.time_step = 0
        self.weather = Weather.CLEAR
        self.wind_vector = (0.0, 0.0, 0.0)
        self.turbulence_level = 0.0
        self.path_history = [(self.drone_state.x, self.drone_state.y, self.drone_state.z)]
        
        # Select random delivery target
        if self.delivery_points:
            self.current_delivery_target = self.np_random.choice(self.delivery_points)
        
        # Randomize weather
        if self.np_random.random() < 0.3:
            self.weather = Weather(self.np_random.integers(0, 4))
            self._update_wind()
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """Execute one step in the environment"""
        self.time_step += 1
        reward = -1  # Time penalty
        done = False
        truncated = False
        
        # Store previous position
        prev_x, prev_y = self.drone_state.x, self.drone_state.y
        
        # Execute action
        action_reward, action_done = self._execute_action(action)
        reward += action_reward
        done = done or action_done
        
        # Update dynamic obstacles
        self._update_dynamic_obstacles()
        
        # Apply physics (wind effect, momentum)
        self._apply_physics()
        
        # Check collisions
        if self._check_collision():
            reward -= 50
            self.collisions += 1
            done = True
        
        # Check restricted zones
        if self._in_restricted_zone():
            reward -= 20
        
        # Update battery
        battery_consumed = self._calculate_battery_consumption(action)
        self.drone_state.battery -= battery_consumed
        
        # Check battery depletion
        if self.drone_state.battery <= 0:
            reward -= 100
            done = True
        elif self.drone_state.battery < 25 and not self.drone_state.has_package:
            # Incentive to return to base when battery is low
            reward -= 5
        
        # Calculate distance traveled
        distance = math.sqrt((self.drone_state.x - prev_x)**2 + 
                           (self.drone_state.y - prev_y)**2)
        self.total_distance += distance
        
        # Update path history with 3D coordinates
        self.path_history.append((self.drone_state.x, self.drone_state.y, self.drone_state.z))
        if len(self.path_history) > 100:
            self.path_history.pop(0)
        
        # Dynamic weather changes
        if self.time_step % 100 == 0:
            if self.np_random.random() < 0.3:
                self.weather = Weather(self.np_random.integers(0, 4))
                self._update_wind()
        
        # Check time limit
        if self.time_step >= self.max_steps:
            truncated = True
        
        # Bonus for efficient delivery
        if self.drone_state.has_package:
            reward += 0.5  # Slight bonus for carrying package
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, done, truncated, info
    
    def _execute_action(self, action: int) -> Tuple[float, bool]:
        """Execute drone action in 3D space and return reward and done status"""
        reward = 0
        done = False
        
        # Horizontal movement actions (0-7)
        if action <= 7:
            dx, dy = 0, 0
            
            if action == 0:  # North
                dy = -1
            elif action == 1:  # South
                dy = 1
            elif action == 2:  # East
                dx = 1
            elif action == 3:  # West
                dx = -1
            elif action == 4:  # Northeast
                dx, dy = 1, -1
            elif action == 5:  # Northwest
                dx, dy = -1, -1
            elif action == 6:  # Southeast
                dx, dy = 1, 1
            elif action == 7:  # Southwest
                dx, dy = -1, 1
            
            # Apply movement with bounds checking
            new_x = np.clip(self.drone_state.x + dx, 0, self.grid_size - 1)
            new_y = np.clip(self.drone_state.y + dy, 0, self.grid_size - 1)
            
            self.drone_state.x = new_x
            self.drone_state.y = new_y
            self.drone_state.velocity_x = dx * 0.5
            self.drone_state.velocity_y = dy * 0.5
            
            # Update orientation based on movement
            if dx != 0 or dy != 0:
                self.drone_state.yaw = np.degrees(np.arctan2(dy, dx))
                # Add slight pitch/roll for realistic banking
                self.drone_state.pitch = -dy * 5
                self.drone_state.roll = dx * 5
        
        # Ascend (action 8)
        elif action == 8:
            dz = 2.0  # Ascend rate
            new_z = np.clip(self.drone_state.z + dz, MIN_ALTITUDE, MAX_ALTITUDE)
            self.drone_state.z = new_z
            self.drone_state.velocity_z = 1.0
            
            # Reward for reaching cruise altitude
            if abs(self.drone_state.z - CRUISE_ALTITUDE) < 5:
                reward += 2
        
        # Descend (action 9)
        elif action == 9:
            dz = -2.0  # Descend rate
            new_z = np.clip(self.drone_state.z + dz, MIN_ALTITUDE, MAX_ALTITUDE)
            self.drone_state.z = new_z
            self.drone_state.velocity_z = -1.0
            
            # Penalty for flying too low (danger)
            if self.drone_state.z < MIN_ALTITUDE + 2:
                reward -= 5
        
        # Hover (action 10)
        elif action == 10:
            # Maintain position
            self.drone_state.velocity_x *= 0.5
            self.drone_state.velocity_y *= 0.5
            self.drone_state.velocity_z *= 0.5
        
        # Pick up package (action 11)
        elif action == 11:
            if self._at_depot() and not self.drone_state.has_package and self.drone_state.z < MIN_ALTITUDE + 3:
                self.drone_state.has_package = True
                reward = 50
            else:
                reward = -5  # Invalid action penalty
        
        # Deliver package (action 12)
        elif action == 12:
            if self.drone_state.has_package and self._at_delivery_point() and self.drone_state.z < MIN_ALTITUDE + 3:
                self.drone_state.has_package = False
                reward = self.current_delivery_target.reward
                self.successful_deliveries += 1
                self.total_deliveries += 1
                done = True
            else:
                reward = -5  # Invalid action penalty
        
        # Return to base (action 13)
        elif action == 13:
            if self._at_depot() and self.drone_state.z < MIN_ALTITUDE + 3:
                # Recharge at depot
                self.drone_state.battery = min(100, self.drone_state.battery + 20)
                reward = 20
            else:
                # Just a regular move towards depot
                reward = -1
        
        return reward, done
    
    def _apply_physics(self):
        """Apply realistic 3D physics including wind, drag, gravity, and turbulence"""
        # Apply 3D wind effect with altitude-dependent strength
        altitude_factor = (self.drone_state.z / MAX_ALTITUDE) ** 0.5  # Wind increases with altitude
        wind_effect_x = self.wind_vector[0] * 0.1 * altitude_factor
        wind_effect_y = self.wind_vector[1] * 0.1 * altitude_factor
        wind_effect_z = self.wind_vector[2] * 0.05 * altitude_factor
        
        # Add random wind gusts
        if self.np_random.random() < self.wind_gust_prob:
            gust_x = self.np_random.uniform(-0.5, 0.5)
            gust_y = self.np_random.uniform(-0.5, 0.5)
            gust_z = self.np_random.uniform(-0.2, 0.2)
            wind_effect_x += gust_x
            wind_effect_y += gust_y
            wind_effect_z += gust_z
        
        # Apply turbulence (random perturbations)
        if self.turbulence_level > 0:
            turb_x = self.np_random.normal(0, self.turbulence_level * 0.1)
            turb_y = self.np_random.normal(0, self.turbulence_level * 0.1)
            turb_z = self.np_random.normal(0, self.turbulence_level * 0.05)
            wind_effect_x += turb_x
            wind_effect_y += turb_y
            wind_effect_z += turb_z
            
            # Turbulence affects orientation
            self.drone_state.pitch += self.np_random.normal(0, self.turbulence_level * 2)
            self.drone_state.roll += self.np_random.normal(0, self.turbulence_level * 2)
        
        self.drone_state.x += wind_effect_x
        self.drone_state.y += wind_effect_y
        self.drone_state.z += wind_effect_z
        
        # Ground effect: increased efficiency near ground
        if self.drone_state.z < GROUND_EFFECT_HEIGHT:
            ground_effect_factor = 1.0 - (self.drone_state.z / GROUND_EFFECT_HEIGHT) * 0.2
            self.drone_state.velocity_z *= ground_effect_factor
        
        # Keep within bounds
        self.drone_state.x = np.clip(self.drone_state.x, 0, self.grid_size - 1)
        self.drone_state.y = np.clip(self.drone_state.y, 0, self.grid_size - 1)
        self.drone_state.z = np.clip(self.drone_state.z, MIN_ALTITUDE, MAX_ALTITUDE)
        
        # Apply drag (reduce velocity over time) - realistic air resistance
        drag_factor = 0.8
        if self.drone_state.z > 50:  # Less drag at higher altitudes (thinner air)
            drag_factor = 0.85
        
        self.drone_state.velocity_x *= drag_factor
        self.drone_state.velocity_y *= drag_factor
        self.drone_state.velocity_z *= drag_factor
        
        # Stabilize orientation over time
        self.drone_state.pitch *= 0.9
        self.drone_state.roll *= 0.9
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles in 3D space"""
        for obstacle in self.dynamic_obstacles:
            obstacle.x += obstacle.velocity_x
            obstacle.y += obstacle.velocity_y
            obstacle.z += obstacle.velocity_z
            
            # Add some random movement for realism (birds, unpredictable drones)
            obstacle.x += self.np_random.normal(0, 0.02)
            obstacle.y += self.np_random.normal(0, 0.02)
            obstacle.z += self.np_random.normal(0, 0.01)
            
            # Bounce off boundaries
            if obstacle.x <= 0 or obstacle.x >= self.grid_size - 1:
                obstacle.velocity_x *= -1
            if obstacle.y <= 0 or obstacle.y >= self.grid_size - 1:
                obstacle.velocity_y *= -1
            if obstacle.z <= MIN_ALTITUDE or obstacle.z >= MAX_ALTITUDE:
                obstacle.velocity_z *= -1
    
    def _update_wind(self):
        """Update 3D wind vector and turbulence based on weather"""
        if self.weather == Weather.CLEAR:
            self.wind_vector = (0.0, 0.0, 0.0)
            self.turbulence_level = 0.0
        elif self.weather == Weather.WINDY:
            angle = self.np_random.random() * 2 * math.pi
            magnitude = self.np_random.random() * 0.5 + 0.3
            vertical_wind = self.np_random.random() * 0.2 - 0.1  # Updrafts/downdrafts
            self.wind_vector = (math.cos(angle) * magnitude, 
                              math.sin(angle) * magnitude,
                              vertical_wind)
            self.turbulence_level = 0.2
        elif self.weather == Weather.RAINY:
            self.wind_vector = (self.np_random.random() * 0.4 - 0.2, 
                              self.np_random.random() * 0.3,
                              -0.1)  # Downdraft from rain
            self.turbulence_level = 0.4
        elif self.weather == Weather.STORMY:
            angle = self.np_random.random() * 2 * math.pi
            magnitude = self.np_random.random() * 1.0 + 0.5
            vertical_wind = self.np_random.random() * 0.6 - 0.3  # Strong vertical winds
            self.wind_vector = (math.cos(angle) * magnitude, 
                              math.sin(angle) * magnitude,
                              vertical_wind)
            self.turbulence_level = 0.8
    
    def _calculate_battery_consumption(self, action: int) -> float:
        """Calculate realistic battery consumption based on action, altitude, and weather"""
        base_consumption = self.battery_consumption_base
        
        # Horizontal movement actions
        if action <= 7:
            base_consumption *= 1.5
        # Vertical movement (more energy intensive)
        elif action == 8:  # Ascend
            base_consumption *= 2.5  # Climbing takes more energy
        elif action == 9:  # Descend
            base_consumption *= 0.8  # Descending is more efficient
        elif action == 10:  # Hover
            base_consumption *= 1.0  # Hovering requires continuous thrust
        
        # Altitude effects - more energy at higher altitudes (thinner air = less efficient)
        altitude_factor = 1.0 + (self.drone_state.z / MAX_ALTITUDE) * 0.3
        base_consumption *= altitude_factor
        
        # Weather effects - wind resistance and turbulence
        if self.weather == Weather.WINDY:
            base_consumption *= 1.5
        elif self.weather == Weather.RAINY:
            base_consumption *= 2.0  # Rain adds weight and drag
        elif self.weather == Weather.STORMY:
            base_consumption *= 3.0  # Fighting strong winds
        
        # Turbulence increases battery consumption (constant corrections)
        base_consumption *= (1.0 + self.turbulence_level * 0.5)
        
        # Package weight effect - more pronounced at higher altitudes
        if self.drone_state.has_package:
            package_factor = 1.2 + (self.drone_state.z / MAX_ALTITUDE) * 0.1
            base_consumption *= package_factor
        
        # Add battery reading uncertainty
        uncertainty = self.np_random.normal(1.0, self.battery_uncertainty)
        base_consumption *= abs(uncertainty)
        
        return base_consumption
    
    def _check_collision(self) -> bool:
        """Check if drone collides with any obstacle in 3D space"""
        drone_x, drone_y, drone_z = self.drone_state.x, self.drone_state.y, self.drone_state.z
        
        # Check static obstacles (buildings)
        for obstacle in self.static_obstacles:
            if (obstacle.x <= drone_x < obstacle.x + obstacle.width and
                obstacle.y <= drone_y < obstacle.y + obstacle.height and
                obstacle.z <= drone_z < obstacle.z + obstacle.depth):
                return True
        
        # Check dynamic obstacles (other drones, birds) with safety margin
        safety_margin = 1.5  # meters
        for obstacle in self.dynamic_obstacles:
            distance_3d = math.sqrt(
                (drone_x - obstacle.x) ** 2 + 
                (drone_y - obstacle.y) ** 2 +
                (drone_z - obstacle.z) ** 2
            )
            if distance_3d < safety_margin:
                return True
        
        return False
    
    def _in_restricted_zone(self) -> bool:
        """Check if drone is in a restricted zone"""
        drone_x, drone_y = int(self.drone_state.x), int(self.drone_state.y)
        
        for zone in self.restricted_zones:
            x, y, w, h = zone
            if x <= drone_x < x + w and y <= drone_y < y + h:
                return True
        
        return False
    
    def _at_depot(self) -> bool:
        """Check if drone is at depot"""
        return (abs(self.drone_state.x - self.depot_location[0]) < 1 and
                abs(self.drone_state.y - self.depot_location[1]) < 1)
    
    def _at_delivery_point(self) -> bool:
        """Check if drone is at current delivery point"""
        if self.current_delivery_target is None:
            return False
        
        return (abs(self.drone_state.x - self.current_delivery_target.x) < 1 and
                abs(self.drone_state.y - self.current_delivery_target.y) < 1)
    
    def _get_observation(self) -> Dict:
        """Get current observation with realistic sensor noise and uncertainties"""
        # Add GPS noise to position
        gps_noise = self.np_random.normal(0, self.gps_noise_std, 3)
        noisy_x = self.drone_state.x + gps_noise[0]
        noisy_y = self.drone_state.y + gps_noise[1]
        noisy_z = self.drone_state.z + gps_noise[2]
        
        # Simulate sensor failures (rare events)
        if self.np_random.random() < self.sensor_failure_prob:
            # Return previous position (sensor lag)
            if len(self.path_history) > 1:
                prev_pos = self.path_history[-2]
                noisy_x, noisy_y, noisy_z = prev_pos[0], prev_pos[1], prev_pos[2]
        
        return {
            'position': np.array([noisy_x, noisy_y, noisy_z], dtype=np.float32),
            'battery': np.array([self.drone_state.battery], dtype=np.float32),
            'package_status': int(self.drone_state.has_package),
            'weather': self.weather.value,
            'time': np.array([self.time_step], dtype=np.float32),
            'velocity': np.array([
                self.drone_state.velocity_x, 
                self.drone_state.velocity_y,
                self.drone_state.velocity_z
            ], dtype=np.float32),
            'wind': np.array(self.wind_vector, dtype=np.float32),
            'orientation': np.array([
                self.drone_state.pitch,
                self.drone_state.roll,
                self.drone_state.yaw
            ], dtype=np.float32),
            'sensor_noise': np.array(gps_noise, dtype=np.float32),
            'turbulence': np.array([self.turbulence_level], dtype=np.float32),
        }
    
    def _get_info(self) -> Dict:
        """Get additional information"""
        return {
            'battery_level': self.drone_state.battery,
            'has_package': self.drone_state.has_package,
            'weather': self.weather.name,
            'time_step': self.time_step,
            'total_deliveries': self.total_deliveries,
            'successful_deliveries': self.successful_deliveries,
            'collisions': self.collisions,
            'total_distance': self.total_distance,
            'at_depot': self._at_depot(),
            'at_delivery': self._at_delivery_point(),
        }
    
    def render(self):
        """Render the environment"""
        if self.render_mode is None:
            return
        
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("Delivery Drone Navigation - Realistic Simulation")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.small_font = pygame.font.Font(None, 24)
        
        # Clear screen with weather-based sky color
        if self.weather == Weather.CLEAR:
            self.window.fill(COLORS['sky_clear'])
        elif self.weather == Weather.WINDY:
            self.window.fill(COLORS['sky_cloudy'])
        else:
            self.window.fill(COLORS['sky_rainy'])
        
        # Calculate offset to center grid
        offset_x = (WINDOW_WIDTH - self.grid_size * CELL_SIZE) // 2
        offset_y = 50
        
        # Draw grid
        self._draw_grid(offset_x, offset_y)
        
        # Draw restricted zones
        self._draw_restricted_zones(offset_x, offset_y)
        
        # Draw obstacles
        self._draw_obstacles(offset_x, offset_y)
        
        # Draw depot
        self._draw_depot(offset_x, offset_y)
        
        # Draw delivery points
        self._draw_delivery_points(offset_x, offset_y)
        
        # Draw path history
        self._draw_path(offset_x, offset_y)
        
        # Draw wind indicator
        if self.weather in [Weather.WINDY, Weather.STORMY]:
            self._draw_wind(offset_x, offset_y)
        
        # Draw drone
        self._draw_drone(offset_x, offset_y)
        
        # Draw UI elements
        self._draw_ui()
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1, 0, 2)
            )
    
    def _draw_grid(self, offset_x: int, offset_y: int):
        """Draw background grid"""
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                rect = pygame.Rect(
                    offset_x + x * CELL_SIZE,
                    offset_y + y * CELL_SIZE,
                    CELL_SIZE,
                    CELL_SIZE
                )
                
                # Alternate ground colors for visual clarity
                if (x + y) % 2 == 0:
                    color = COLORS['ground']
                else:
                    color = tuple(max(0, c - 10) for c in COLORS['ground'])
                
                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (100, 100, 100), rect, 1)
    
    def _draw_restricted_zones(self, offset_x: int, offset_y: int):
        """Draw restricted no-fly zones"""
        for zone in self.restricted_zones:
            x, y, w, h = zone
            rect = pygame.Rect(
                offset_x + x * CELL_SIZE,
                offset_y + y * CELL_SIZE,
                w * CELL_SIZE,
                h * CELL_SIZE
            )
            pygame.draw.rect(self.window, COLORS['restricted'], rect)
            pygame.draw.rect(self.window, (255, 255, 255), rect, 2)
            
            # Draw warning stripes
            for i in range(0, w + h):
                start = (offset_x + x * CELL_SIZE + i * CELL_SIZE // 2,
                        offset_y + y * CELL_SIZE)
                end = (offset_x + x * CELL_SIZE,
                      offset_y + y * CELL_SIZE + i * CELL_SIZE // 2)
                pygame.draw.line(self.window, (255, 255, 0), start, end, 2)
    
    def _draw_obstacles(self, offset_x: int, offset_y: int):
        """Draw static and dynamic obstacles"""
        # Static obstacles (buildings)
        for obstacle in self.static_obstacles:
            rect = pygame.Rect(
                offset_x + obstacle.x * CELL_SIZE,
                offset_y + obstacle.y * CELL_SIZE,
                obstacle.width * CELL_SIZE,
                obstacle.height * CELL_SIZE
            )
            pygame.draw.rect(self.window, COLORS['building'], rect)
            pygame.draw.rect(self.window, (0, 0, 0), rect, 2)
            
            # Draw windows on buildings
            for i in range(obstacle.width):
                for j in range(obstacle.height):
                    window_rect = pygame.Rect(
                        offset_x + (obstacle.x + i) * CELL_SIZE + 10,
                        offset_y + (obstacle.y + j) * CELL_SIZE + 10,
                        10, 10
                    )
                    pygame.draw.rect(self.window, (173, 216, 230), window_rect)
        
        # Dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            center_x = offset_x + int(obstacle.x * CELL_SIZE) + CELL_SIZE // 2
            center_y = offset_y + int(obstacle.y * CELL_SIZE) + CELL_SIZE // 2
            pygame.draw.circle(self.window, COLORS['obstacle'], (center_x, center_y), CELL_SIZE // 3)
            pygame.draw.circle(self.window, (255, 165, 0), (center_x, center_y), CELL_SIZE // 3, 2)
    
    def _draw_depot(self, offset_x: int, offset_y: int):
        """Draw depot/base station"""
        x, y = self.depot_location
        rect = pygame.Rect(
            offset_x + x * CELL_SIZE,
            offset_y + y * CELL_SIZE,
            CELL_SIZE,
            CELL_SIZE
        )
        pygame.draw.rect(self.window, COLORS['depot'], rect)
        pygame.draw.rect(self.window, (255, 255, 255), rect, 3)
        
        # Draw "H" for helipad
        text = self.small_font.render("H", True, (255, 255, 255))
        text_rect = text.get_rect(center=rect.center)
        self.window.blit(text, text_rect)
    
    def _draw_delivery_points(self, offset_x: int, offset_y: int):
        """Draw delivery locations"""
        for i, point in enumerate(self.delivery_points):
            center_x = offset_x + point.x * CELL_SIZE + CELL_SIZE // 2
            center_y = offset_y + point.y * CELL_SIZE + CELL_SIZE // 2
            
            # Highlight current target
            if point == self.current_delivery_target:
                pygame.draw.circle(self.window, (255, 255, 0), (center_x, center_y), CELL_SIZE // 2 + 5)
            
            pygame.draw.circle(self.window, COLORS['delivery'], (center_x, center_y), CELL_SIZE // 2)
            pygame.draw.circle(self.window, (0, 0, 0), (center_x, center_y), CELL_SIZE // 2, 2)
            
            # Draw priority number
            text = self.small_font.render(str(point.priority), True, (0, 0, 0))
            text_rect = text.get_rect(center=(center_x, center_y))
            self.window.blit(text, text_rect)
    
    def _draw_path(self, offset_x: int, offset_y: int):
        """Draw drone's 3D path history with altitude coloring"""
        if len(self.path_history) < 2:
            return
        
        for i in range(len(self.path_history) - 1):
            x1, y1, z1 = self.path_history[i]
            x2, y2, z2 = self.path_history[i + 1]
            
            start_pos = (offset_x + int(x1 * CELL_SIZE) + CELL_SIZE // 2,
                        offset_y + int(y1 * CELL_SIZE) + CELL_SIZE // 2)
            end_pos = (offset_x + int(x2 * CELL_SIZE) + CELL_SIZE // 2,
                      offset_y + int(y2 * CELL_SIZE) + CELL_SIZE // 2)
            
            # Color based on altitude (blue for low, yellow for high)
            altitude_ratio = (z1 - MIN_ALTITUDE) / (MAX_ALTITUDE - MIN_ALTITUDE)
            red = int(255 * altitude_ratio)
            blue = int(255 * (1 - altitude_ratio))
            
            # Fade older path segments
            alpha = int(200 * (i / len(self.path_history)))
            color = (red + alpha // 3, alpha, blue + alpha // 3)
            
            # Line width based on altitude
            line_width = max(1, int(2 + altitude_ratio * 2))
            pygame.draw.line(self.window, color, start_pos, end_pos, line_width)
    
    def _draw_wind(self, offset_x: int, offset_y: int):
        """Draw wind indicators"""
        for x in range(0, self.grid_size, 3):
            for y in range(0, self.grid_size, 3):
                center_x = offset_x + x * CELL_SIZE + CELL_SIZE // 2
                center_y = offset_y + y * CELL_SIZE + CELL_SIZE // 2
                
                # Draw wind arrow
                wind_length = 20
                end_x = center_x + int(self.wind_vector[0] * wind_length)
                end_y = center_y + int(self.wind_vector[1] * wind_length)
                
                pygame.draw.line(self.window, COLORS['wind'], 
                               (center_x, center_y), (end_x, end_y), 1)
                
                # Draw arrowhead
                angle = math.atan2(self.wind_vector[1], self.wind_vector[0])
                arrow_size = 5
                pygame.draw.line(self.window, COLORS['wind'],
                               (end_x, end_y),
                               (end_x - arrow_size * math.cos(angle + math.pi/6),
                                end_y - arrow_size * math.sin(angle + math.pi/6)), 1)
                pygame.draw.line(self.window, COLORS['wind'],
                               (end_x, end_y),
                               (end_x - arrow_size * math.cos(angle - math.pi/6),
                                end_y - arrow_size * math.sin(angle - math.pi/6)), 1)
    
    def _draw_drone(self, offset_x: int, offset_y: int):
        """Draw the drone with 3D perspective and altitude indication"""
        center_x = offset_x + int(self.drone_state.x * CELL_SIZE) + CELL_SIZE // 2
        center_y = offset_y + int(self.drone_state.y * CELL_SIZE) + CELL_SIZE // 2
        
        # Scale based on altitude (perspective effect)
        altitude_scale = 1.0 + (self.drone_state.z - MIN_ALTITUDE) / (MAX_ALTITUDE - MIN_ALTITUDE) * 0.5
        drone_radius = int(CELL_SIZE // 3 * altitude_scale)
        
        # Draw altitude indicator (shadow on ground)
        shadow_size = int(CELL_SIZE // 4)
        shadow_alpha = max(50, int(150 * (1 - self.drone_state.z / MAX_ALTITUDE)))
        shadow_surface = pygame.Surface((shadow_size * 2, shadow_size * 2), pygame.SRCALPHA)
        pygame.draw.circle(shadow_surface, (0, 0, 0, shadow_alpha), 
                         (shadow_size, shadow_size), shadow_size)
        self.window.blit(shadow_surface, (center_x - shadow_size, center_y - shadow_size))
        
        # Draw altitude line from ground to drone
        if self.drone_state.z > MIN_ALTITUDE + 5:
            pygame.draw.line(self.window, (100, 100, 100, 128), 
                           (center_x, center_y), 
                           (center_x, center_y - int(self.drone_state.z * 2)), 1)
        
        # Draw drone body with tilt based on orientation
        tilt_offset_x = int(self.drone_state.roll * 0.5)
        tilt_offset_y = int(self.drone_state.pitch * 0.5)
        drone_center = (center_x + tilt_offset_x, center_y + tilt_offset_y)
        
        pygame.draw.circle(self.window, COLORS['drone'], drone_center, drone_radius)
        pygame.draw.circle(self.window, (139, 0, 0), drone_center, drone_radius, 2)
        
        # Draw propellers with altitude scaling
        propeller_distance = int(CELL_SIZE // 2 * altitude_scale)
        propeller_positions = [
            (drone_center[0] - propeller_distance, drone_center[1] - propeller_distance),
            (drone_center[0] + propeller_distance, drone_center[1] - propeller_distance),
            (drone_center[0] - propeller_distance, drone_center[1] + propeller_distance),
            (drone_center[0] + propeller_distance, drone_center[1] + propeller_distance),
        ]
        
        propeller_size = int(5 * altitude_scale)
        for pos in propeller_positions:
            pygame.draw.circle(self.window, (50, 50, 50), pos, propeller_size)
            # Rotating propeller effect
            rotation = (self.time_step * 20) % 360
            for angle in [rotation, rotation + 90]:
                rad = math.radians(angle)
                blade_length = int(8 * altitude_scale)
                blade_end = (pos[0] + int(blade_length * math.cos(rad)),
                           pos[1] + int(blade_length * math.sin(rad)))
                pygame.draw.line(self.window, (100, 100, 100), pos, blade_end, 2)
        
        # Draw package if carrying
        if self.drone_state.has_package:
            package_size = int(8 * altitude_scale)
            package_rect = pygame.Rect(drone_center[0] - package_size, 
                                      drone_center[1] + drone_radius, 
                                      package_size * 2, int(package_size * 1.5))
            pygame.draw.rect(self.window, COLORS['package'], package_rect)
            pygame.draw.rect(self.window, (0, 0, 0), package_rect, 1)
        
        # Draw velocity vector
        if abs(self.drone_state.velocity_x) > 0.1 or abs(self.drone_state.velocity_y) > 0.1:
            vel_end_x = drone_center[0] + int(self.drone_state.velocity_x * 30)
            vel_end_y = drone_center[1] + int(self.drone_state.velocity_y * 30)
            pygame.draw.line(self.window, (0, 255, 0), 
                           drone_center, (vel_end_x, vel_end_y), 2)
        
        # Draw altitude text above drone
        altitude_text = self.small_font.render(f"{self.drone_state.z:.1f}m", True, (255, 255, 255))
        altitude_bg = pygame.Surface((altitude_text.get_width() + 4, altitude_text.get_height() + 2), pygame.SRCALPHA)
        altitude_bg.fill((0, 0, 0, 180))
        self.window.blit(altitude_bg, (drone_center[0] - altitude_text.get_width() // 2 - 2, 
                                       drone_center[1] - drone_radius - 20))
        self.window.blit(altitude_text, (drone_center[0] - altitude_text.get_width() // 2, 
                                         drone_center[1] - drone_radius - 19))
    
    def _draw_ui(self):
        """Draw user interface elements with 3D information"""
        # Background panel - expanded for 3D info
        panel_rect = pygame.Rect(10, 10, 320, 280)
        pygame.draw.rect(self.window, (0, 0, 0, 180), panel_rect)
        pygame.draw.rect(self.window, (255, 255, 255), panel_rect, 2)
        
        y_offset = 20
        
        # Battery indicator
        battery_text = f"Battery: {self.drone_state.battery:.1f}%"
        if self.drone_state.battery > 75:
            battery_color = COLORS['battery_high']
        elif self.drone_state.battery > 50:
            battery_color = COLORS['battery_medium']
        elif self.drone_state.battery > 25:
            battery_color = COLORS['battery_low']
        else:
            battery_color = COLORS['battery_critical']
        
        text_surface = self.small_font.render(battery_text, True, battery_color)
        self.window.blit(text_surface, (20, y_offset))
        
        # Battery bar
        battery_bar_rect = pygame.Rect(150, y_offset, 150, 20)
        pygame.draw.rect(self.window, (50, 50, 50), battery_bar_rect)
        fill_width = int(150 * self.drone_state.battery / 100)
        fill_rect = pygame.Rect(150, y_offset, fill_width, 20)
        pygame.draw.rect(self.window, battery_color, fill_rect)
        pygame.draw.rect(self.window, (255, 255, 255), battery_bar_rect, 1)
        
        y_offset += 30
        
        # 3D Position
        pos_text = f"Pos: ({self.drone_state.x:.1f}, {self.drone_state.y:.1f}, {self.drone_state.z:.1f})"
        text_surface = self.small_font.render(pos_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 25
        
        # Altitude with visual indicator
        alt_text = f"Altitude: {self.drone_state.z:.1f}m"
        alt_color = (0, 255, 0) if MIN_ALTITUDE < self.drone_state.z < MAX_ALTITUDE - 10 else (255, 165, 0)
        text_surface = self.small_font.render(alt_text, True, alt_color)
        self.window.blit(text_surface, (20, y_offset))
        
        # Altitude bar
        alt_bar_rect = pygame.Rect(150, y_offset, 150, 20)
        pygame.draw.rect(self.window, (50, 50, 50), alt_bar_rect)
        alt_fill = int(150 * (self.drone_state.z - MIN_ALTITUDE) / (MAX_ALTITUDE - MIN_ALTITUDE))
        alt_fill_rect = pygame.Rect(150, y_offset, alt_fill, 20)
        pygame.draw.rect(self.window, alt_color, alt_fill_rect)
        pygame.draw.rect(self.window, (255, 255, 255), alt_bar_rect, 1)
        
        y_offset += 30
        
        # Orientation
        orient_text = f"P:{self.drone_state.pitch:.0f}° R:{self.drone_state.roll:.0f}° Y:{self.drone_state.yaw:.0f}°"
        text_surface = self.small_font.render(orient_text, True, (200, 200, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 25
        
        # Package status
        package_text = f"Package: {'Carrying' if self.drone_state.has_package else 'None'}"
        package_color = (0, 255, 0) if self.drone_state.has_package else (255, 255, 255)
        text_surface = self.small_font.render(package_text, True, package_color)
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 25
        
        # Weather and turbulence
        weather_text = f"Weather: {self.weather.name}"
        text_surface = self.small_font.render(weather_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 20
        turb_text = f"Turbulence: {self.turbulence_level:.1f}"
        turb_color = (255, int(255 * (1 - self.turbulence_level)), 0)
        text_surface = self.small_font.render(turb_text, True, turb_color)
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 25
        
        # Wind (3D)
        wind_mag = math.sqrt(sum(w**2 for w in self.wind_vector))
        wind_text = f"Wind: {wind_mag:.2f} m/s"
        text_surface = self.small_font.render(wind_text, True, (200, 200, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 25
        
        # Time step
        time_text = f"Time: {self.time_step}/{self.max_steps}"
        text_surface = self.small_font.render(time_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 25
        
        # Statistics
        stats_text = f"Deliveries: {self.successful_deliveries}/{self.total_deliveries}"
        text_surface = self.small_font.render(stats_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        # Right side panel - Legend
        legend_rect = pygame.Rect(WINDOW_WIDTH - 210, 10, 200, 250)
        pygame.draw.rect(self.window, (0, 0, 0, 180), legend_rect)
        pygame.draw.rect(self.window, (255, 255, 255), legend_rect, 2)
        
        legend_y = 20
        legend_items = [
            ("Depot", COLORS['depot']),
            ("Delivery", COLORS['delivery']),
            ("Building", COLORS['building']),
            ("Restricted", COLORS['restricted']),
            ("Drone", COLORS['drone']),
        ]
        
        for label, color in legend_items:
            # Draw color box
            box_rect = pygame.Rect(WINDOW_WIDTH - 200, legend_y, 20, 20)
            pygame.draw.rect(self.window, color, box_rect)
            pygame.draw.rect(self.window, (255, 255, 255), box_rect, 1)
            
            # Draw label
            text_surface = self.small_font.render(label, True, (255, 255, 255))
            self.window.blit(text_surface, (WINDOW_WIDTH - 170, legend_y))
            
            legend_y += 30
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None


if __name__ == "__main__":
    # Test the environment
    env = RealisticDroneEnv(render_mode="human")
    
    observation, info = env.reset()
    
    for _ in range(1000):
        # Random action for testing
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        
        env.render()
        
        if done or truncated:
            observation, info = env.reset()
    
    env.close()