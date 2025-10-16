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
    x: int
    y: int
    battery: float  # 0-100
    has_package: bool
    velocity_x: float
    velocity_y: float
    altitude: float  # Simulated altitude


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
    """Dynamic or static obstacle"""
    x: int
    y: int
    width: int
    height: int
    is_dynamic: bool
    velocity_x: float = 0
    velocity_y: float = 0


class RealisticDroneEnv(gym.Env):
    """
    Realistic Drone Delivery Environment with Pygame Visualization
    
    State Space:
        - Position (x, y): continuous coordinates
        - Battery level: 0-100%
        - Package status: {0, 1, 2}
        - Weather: {0, 1, 2, 3}
        - Time: continuous
        - Velocity: (vx, vy)
        - Wind: (wind_x, wind_y)
        
    Action Space:
        - 0: Move North
        - 1: Move South
        - 2: Move East
        - 3: Move West
        - 4: Move Northeast
        - 5: Move Northwest
        - 6: Move Southeast
        - 7: Move Southwest
        - 8: Hover
        - 9: Pick up package
        - 10: Deliver package
        - 11: Return to base
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': FPS}
    
    def __init__(self, render_mode: Optional[str] = None, grid_size: int = GRID_SIZE):
        super().__init__()
        
        self.grid_size = grid_size
        self.render_mode = render_mode
        
        # Action space: 12 discrete actions
        self.action_space = spaces.Discrete(12)
        
        # Observation space: continuous state
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.float32),
            'battery': spaces.Box(low=0, high=100, shape=(1,), dtype=np.float32),
            'package_status': spaces.Discrete(3),
            'weather': spaces.Discrete(4),
            'time': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'velocity': spaces.Box(low=-2, high=2, shape=(2,), dtype=np.float32),
            'wind': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32),
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
        self.wind_vector = (0.0, 0.0)
        self.time_step = 0
        
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
        
        # Create static obstacles (buildings)
        self.static_obstacles = [
            Obstacle(x=5, y=5, width=3, height=3, is_dynamic=False),
            Obstacle(x=10, y=8, width=2, height=4, is_dynamic=False),
            Obstacle(x=15, y=3, width=2, height=2, is_dynamic=False),
            Obstacle(x=8, y=14, width=3, height=2, is_dynamic=False),
            Obstacle(x=12, y=12, width=2, height=3, is_dynamic=False),
        ]
        
        # Create dynamic obstacles (other drones, vehicles)
        self.dynamic_obstacles = [
            Obstacle(x=7, y=7, width=1, height=1, is_dynamic=True, 
                    velocity_x=0.1, velocity_y=0.05),
            Obstacle(x=13, y=10, width=1, height=1, is_dynamic=True, 
                    velocity_x=-0.08, velocity_y=0.12),
        ]
        
        # Restricted zones (no-fly zones)
        self.restricted_zones = [
            (10, 1, 3, 3),  # (x, y, width, height)
            (3, 10, 2, 2),
        ]
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        # Reset drone state
        self.drone_state = DroneState(
            x=self.depot_location[0],
            y=self.depot_location[1],
            battery=100.0,
            has_package=False,
            velocity_x=0.0,
            velocity_y=0.0,
            altitude=10.0
        )
        
        # Reset environment
        self.time_step = 0
        self.weather = Weather.CLEAR
        self.wind_vector = (0.0, 0.0)
        self.path_history = [(self.drone_state.x, self.drone_state.y)]
        
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
        
        # Update path history
        self.path_history.append((int(self.drone_state.x), int(self.drone_state.y)))
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
        """Execute drone action and return reward and done status"""
        reward = 0
        done = False
        
        # Movement actions (0-8)
        if action <= 8:
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
            # action == 8 is hover (dx=0, dy=0)
            
            # Apply movement with bounds checking
            new_x = np.clip(self.drone_state.x + dx, 0, self.grid_size - 1)
            new_y = np.clip(self.drone_state.y + dy, 0, self.grid_size - 1)
            
            self.drone_state.x = new_x
            self.drone_state.y = new_y
            self.drone_state.velocity_x = dx * 0.5
            self.drone_state.velocity_y = dy * 0.5
        
        # Pick up package (action 9)
        elif action == 9:
            if self._at_depot() and not self.drone_state.has_package:
                self.drone_state.has_package = True
                reward = 50
            else:
                reward = -5  # Invalid action penalty
        
        # Deliver package (action 10)
        elif action == 10:
            if self.drone_state.has_package and self._at_delivery_point():
                self.drone_state.has_package = False
                reward = self.current_delivery_target.reward
                self.successful_deliveries += 1
                self.total_deliveries += 1
                done = True
            else:
                reward = -5  # Invalid action penalty
        
        # Return to base (action 11)
        elif action == 11:
            if self._at_depot():
                # Recharge at depot
                self.drone_state.battery = min(100, self.drone_state.battery + 20)
                reward = 20
            else:
                # Just a regular move towards depot
                reward = -1
        
        return reward, done
    
    def _apply_physics(self):
        """Apply wind and physics effects"""
        # Apply wind effect
        wind_effect_x = self.wind_vector[0] * 0.1
        wind_effect_y = self.wind_vector[1] * 0.1
        
        self.drone_state.x += wind_effect_x
        self.drone_state.y += wind_effect_y
        
        # Keep within bounds
        self.drone_state.x = np.clip(self.drone_state.x, 0, self.grid_size - 1)
        self.drone_state.y = np.clip(self.drone_state.y, 0, self.grid_size - 1)
        
        # Apply drag (reduce velocity over time)
        self.drone_state.velocity_x *= 0.8
        self.drone_state.velocity_y *= 0.8
    
    def _update_dynamic_obstacles(self):
        """Update positions of dynamic obstacles"""
        for obstacle in self.dynamic_obstacles:
            obstacle.x += obstacle.velocity_x
            obstacle.y += obstacle.velocity_y
            
            # Bounce off boundaries
            if obstacle.x <= 0 or obstacle.x >= self.grid_size - 1:
                obstacle.velocity_x *= -1
            if obstacle.y <= 0 or obstacle.y >= self.grid_size - 1:
                obstacle.velocity_y *= -1
    
    def _update_wind(self):
        """Update wind vector based on weather"""
        if self.weather == Weather.CLEAR:
            self.wind_vector = (0.0, 0.0)
        elif self.weather == Weather.WINDY:
            angle = self.np_random.random() * 2 * math.pi
            magnitude = self.np_random.random() * 0.5 + 0.3
            self.wind_vector = (math.cos(angle) * magnitude, 
                              math.sin(angle) * magnitude)
        elif self.weather == Weather.RAINY:
            self.wind_vector = (self.np_random.random() * 0.4 - 0.2, 
                              self.np_random.random() * 0.3)
        elif self.weather == Weather.STORMY:
            angle = self.np_random.random() * 2 * math.pi
            magnitude = self.np_random.random() * 1.0 + 0.5
            self.wind_vector = (math.cos(angle) * magnitude, 
                              math.sin(angle) * magnitude)
    
    def _calculate_battery_consumption(self, action: int) -> float:
        """Calculate battery consumption based on action and weather"""
        base_consumption = self.battery_consumption_base
        
        # Movement actions consume more battery
        if action <= 7:  # Movement actions
            base_consumption *= 1.5
        elif action == 8:  # Hover
            base_consumption *= 0.5
        
        # Weather effects
        if self.weather == Weather.WINDY:
            base_consumption *= 1.5
        elif self.weather == Weather.RAINY:
            base_consumption *= 2.0
        elif self.weather == Weather.STORMY:
            base_consumption *= 3.0
        
        # Package weight effect
        if self.drone_state.has_package:
            base_consumption *= 1.2
        
        return base_consumption
    
    def _check_collision(self) -> bool:
        """Check if drone collides with any obstacle"""
        drone_x, drone_y = int(self.drone_state.x), int(self.drone_state.y)
        
        # Check static obstacles
        for obstacle in self.static_obstacles:
            if (obstacle.x <= drone_x < obstacle.x + obstacle.width and
                obstacle.y <= drone_y < obstacle.y + obstacle.height):
                return True
        
        # Check dynamic obstacles
        for obstacle in self.dynamic_obstacles:
            if (abs(drone_x - obstacle.x) < 1 and 
                abs(drone_y - obstacle.y) < 1):
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
        """Get current observation"""
        return {
            'position': np.array([self.drone_state.x, self.drone_state.y], dtype=np.float32),
            'battery': np.array([self.drone_state.battery], dtype=np.float32),
            'package_status': int(self.drone_state.has_package),
            'weather': self.weather.value,
            'time': np.array([self.time_step], dtype=np.float32),
            'velocity': np.array([self.drone_state.velocity_x, self.drone_state.velocity_y], dtype=np.float32),
            'wind': np.array(self.wind_vector, dtype=np.float32),
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
        """Draw drone's path history"""
        if len(self.path_history) < 2:
            return
        
        for i in range(len(self.path_history) - 1):
            x1, y1 = self.path_history[i]
            x2, y2 = self.path_history[i + 1]
            
            start_pos = (offset_x + x1 * CELL_SIZE + CELL_SIZE // 2,
                        offset_y + y1 * CELL_SIZE + CELL_SIZE // 2)
            end_pos = (offset_x + x2 * CELL_SIZE + CELL_SIZE // 2,
                      offset_y + y2 * CELL_SIZE + CELL_SIZE // 2)
            
            # Fade older path segments
            alpha = int(255 * (i / len(self.path_history)))
            color = (alpha, alpha, 255)
            pygame.draw.line(self.window, color, start_pos, end_pos, 2)
    
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
        """Draw the drone"""
        center_x = offset_x + int(self.drone_state.x * CELL_SIZE) + CELL_SIZE // 2
        center_y = offset_y + int(self.drone_state.y * CELL_SIZE) + CELL_SIZE // 2
        
        # Draw drone shadow
        shadow_offset = 5
        pygame.draw.circle(self.window, (0, 0, 0, 50), 
                         (center_x + shadow_offset, center_y + shadow_offset), 
                         CELL_SIZE // 3)
        
        # Draw drone body
        pygame.draw.circle(self.window, COLORS['drone'], (center_x, center_y), CELL_SIZE // 3)
        pygame.draw.circle(self.window, (139, 0, 0), (center_x, center_y), CELL_SIZE // 3, 2)
        
        # Draw propellers
        propeller_distance = CELL_SIZE // 2
        propeller_positions = [
            (center_x - propeller_distance, center_y - propeller_distance),
            (center_x + propeller_distance, center_y - propeller_distance),
            (center_x - propeller_distance, center_y + propeller_distance),
            (center_x + propeller_distance, center_y + propeller_distance),
        ]
        
        for pos in propeller_positions:
            pygame.draw.circle(self.window, (50, 50, 50), pos, 5)
            # Rotating propeller effect
            rotation = (self.time_step * 20) % 360
            for angle in [rotation, rotation + 90]:
                rad = math.radians(angle)
                blade_end = (pos[0] + int(8 * math.cos(rad)),
                           pos[1] + int(8 * math.sin(rad)))
                pygame.draw.line(self.window, (100, 100, 100), pos, blade_end, 2)
        
        # Draw package if carrying
        if self.drone_state.has_package:
            package_rect = pygame.Rect(center_x - 8, center_y + 10, 16, 12)
            pygame.draw.rect(self.window, COLORS['package'], package_rect)
            pygame.draw.rect(self.window, (0, 0, 0), package_rect, 1)
        
        # Draw velocity vector
        if abs(self.drone_state.velocity_x) > 0.1 or abs(self.drone_state.velocity_y) > 0.1:
            vel_end_x = center_x + int(self.drone_state.velocity_x * 30)
            vel_end_y = center_y + int(self.drone_state.velocity_y * 30)
            pygame.draw.line(self.window, (0, 255, 0), 
                           (center_x, center_y), (vel_end_x, vel_end_y), 2)
    
    def _draw_ui(self):
        """Draw user interface elements"""
        # Background panel
        panel_rect = pygame.Rect(10, 10, 300, 200)
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
        
        # Package status
        package_text = f"Package: {'Carrying' if self.drone_state.has_package else 'None'}"
        package_color = (0, 255, 0) if self.drone_state.has_package else (255, 255, 255)
        text_surface = self.small_font.render(package_text, True, package_color)
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 30
        
        # Weather
        weather_text = f"Weather: {self.weather.name}"
        text_surface = self.small_font.render(weather_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 30
        
        # Time step
        time_text = f"Time: {self.time_step}/{self.max_steps}"
        text_surface = self.small_font.render(time_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 30
        
        # Position
        pos_text = f"Pos: ({self.drone_state.x:.1f}, {self.drone_state.y:.1f})"
        text_surface = self.small_font.render(pos_text, True, (255, 255, 255))
        self.window.blit(text_surface, (20, y_offset))
        
        y_offset += 30
        
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