

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pygame
import time
from pygame import gfxdraw

# Define order types and their probabilities
ORDER_TYPES = {
    'frozen_food': 1,
    'prepared_meals': 0,
    'meal_kits': 0,
    'fast_food': 0
}

COLORS = {
    'background': (240, 242, 245),
    'grid': (220, 225, 230),
    'van': (52, 73, 94),
    'agent': (41, 128, 185),
    'frozen_food': (26, 188, 156),
    'prepared_meals': (46, 204, 113),
    'meal_kits': (241, 196, 15),
    'fast_food': (231, 76, 60),
    'text': (44, 62, 80),
    'panel': (255, 255, 255),
    'border': (189, 195, 199)
}


class DeliveryEnv(gym.Env):
    def __init__(self, grid_size=20, n_agents=4, max_items=1, max_demands=50, episode_timeout=100, render_size=1000):
        super(DeliveryEnv, self).__init__()
        
        # Initialize environment parameters
        self.grid_size = grid_size
        self.n_agents = n_agents
        self.max_items = max_items
        self.max_demands = max_demands
        self.render_size = render_size
        self.grid_render_size = int(render_size * 0.8)
        self.cell_size = self.grid_render_size // grid_size
        self.dashboard_width = render_size - self.grid_render_size
            
        # Initialize van inventory (25 of each type)
        self.van_inventory = {
            'frozen_food': 100,
            # 'prepared_meals': 25,
            # 'meal_kits': 25,
            # 'fast_food': 25
        }
        
        # Initialize Pygame only if not already initialized
        if not pygame.get_init():
            pygame.init()
        
        # Initialize display properties but don't create screen yet
        self.screen = None
        self.font = pygame.font.Font(None, 32)
        self.small_font = pygame.font.Font(None, 24)
        
        # Action space now only includes movement (no pickup/deliver actions)
        self.action_space = gym.spaces.Discrete(4)
        
        # Calculate observation space size
        obs_size = 8 + 4 + (self.grid_size * self.grid_size * 4)  # Base features + van inventory + demand grid
        
        # Initialize observation space
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )
        
        # Initialize other attributes
        self.active_demands = []
        self.fulfilled_demands = 0
        self.total_demands = 0
        self.current_episode = 0
        self.last_checkpoint = 0
        self.total_rewards = 0
        self.episode_steps = 0
        self.start_time = time.time()
        self.episode_start_time = None
        self.episode_timeout = episode_timeout
        self.last_demand_time = None
        self.edge_hits = 0

        # Add new tracking for agent targets and movement
        self.agent_targets = [None] * n_agents
        self.target_hold_time = [0] * n_agents
        self.previous_distances = [float('inf')] * n_agents
        self.last_action = [None] * n_agents
        self.prev_positions = [None] * n_agents

    def step(self, actions):
        current_time = time.time()
        rewards = np.zeros(self.n_agents)
        done = False
        info = {}
        
        # Store previous positions for all agents
        for i in range(self.n_agents):
            self.prev_positions[i] = self.agents[i].copy()
        
        # Check episode end conditions
        if current_time - self.episode_start_time > self.episode_timeout:
            done = True
            info['timeout'] = True
            return self._get_obs(), rewards, done, info
        
        if sum(self.van_inventory.values()) == 0:
            done = True
            info['out_of_stock'] = True
            return self._get_obs(), rewards, done, info
        
        # Always check if we need to generate new demands
        active_unfulfilled = [d for d in self.active_demands if not d['fulfilled']]
        if len(active_unfulfilled) == 0:
            self.generate_new_demand()

        # Update agent targets
        for i in range(self.n_agents):
            current_target = self.agent_targets[i]
            if (current_target is None or 
                current_target.get('fulfilled', True) or
                not self.is_target_in_demands(current_target, self.active_demands)):
                self.agent_targets[i] = self.get_best_target(i)
                self.target_hold_time[i] = 0
            else:
                self.target_hold_time[i] += 1
        
        # Process actions for each agent
        for i, action in enumerate(actions):
            agent_pos = self.agents[i]
            target = self.agent_targets[i]
            
            # Handle movement
            direction = {
                0: [-1, 0],  # up
                1: [1, 0],   # down
                2: [0, -1],  # left
                3: [0, 1]    # right
            }[action]
            # print(direction)
            new_pos = agent_pos + np.array(direction)
            
            # Base movement penalty
            rewards[i] -= 0.2  # Small penalty for any movement
            
            # Check if agent is staying in the same position
            # if np.array_equal(new_pos, agent_pos):
            #     rewards[i] -= 200.0  # Larger penalty for staying still
            
            # Define corner zones more strictly
            corner_zones = [
                (0, 0), (0, 1), (1, 0),  # Top-left
                (0, self.grid_size-1), (0, self.grid_size-2), (1, self.grid_size-1),  # Top-right
                (self.grid_size-1, 0), (self.grid_size-2, 0), (self.grid_size-1, 1),  # Bottom-left
                (self.grid_size-1, self.grid_size-1), (self.grid_size-2, self.grid_size-1), (self.grid_size-1, self.grid_size-2)  # Bottom-right
            ]
            
            # Apply corner penalty if agent is in or moving to a corner zone
            # print (tuple(new_pos))
            # print (corner_zones)
            if tuple(agent_pos) in corner_zones:
                rewards[i] -= 20.0  # Significant penalty for corner camping
                self.edge_hits += 1

            # Check if the move is valid and process it
            if (0 <= new_pos[0] < self.grid_size and 
                0 <= new_pos[1] < self.grid_size):
                
                was_at_van = np.array_equal(agent_pos, self.van_pos)
                self.agents[i] = new_pos
                
                # Automatic pickup when leaving van
                if was_at_van and not np.array_equal(new_pos, self.van_pos):
                    if len(self.agent_inventories[i]) < self.max_items:
                        if target is not None:
                            item_type = target['type']
                            if self.van_inventory[item_type] > 0:
                                self.agent_inventories[i].append(item_type)
                                self.van_inventory[item_type] -= 1
                                rewards[i] += 200.0
                            else:
                                available_items = [item for item, count in self.van_inventory.items() if count > 0]
                                if available_items:
                                    item = np.random.choice(available_items)
                                    self.agent_inventories[i].append(item)
                                    self.van_inventory[item] -= 1
                                    rewards[i] += 150.0
                
                # Process deliveries at new position
                if len(self.agent_inventories[i]) > 0:
                    delivered_item = self.agent_inventories[i][0]
                    for demand in self.active_demands:
                        if (not demand['fulfilled'] and 
                            np.array_equal(new_pos, demand['position']) and 
                            demand['type'] == delivered_item):
                            
                            delivery_time = current_time - demand['creation_time']
                            time_bonus = max(1000 - delivery_time * 10, 200)
                            
                            # Higher reward for targeted delivery
                            if target is not None and np.array_equal(demand['position'], target['position']):
                                rewards[i] += 2000.0 + time_bonus
                            else:  # Smaller reward for opportunistic delivery
                                rewards[i] += 1000.0 + time_bonus
                            
                            demand['fulfilled'] = True
                            self.agent_inventories[i].remove(delivered_item)
                            self.fulfilled_demands += 1
                            
                            # Clear target if we just delivered to it
                            if target is not None and np.array_equal(demand['position'], target['position']):
                                self.agent_targets[i] = None
                                self.agent_targets[i] = self.get_best_target(i)
                            break
            else:
                # Move is not valid
                rewards[i] -= 20.0  # Significant penalty for edge actions
                self.edge_hits += 1
        # Update episode stats
        self.total_rewards += sum(rewards)
        self.episode_steps += 1
        
        # Update info dictionary
        info.update({
            'fulfilled_demands': self.fulfilled_demands,
            'total_demands': self.total_demands,
            'active_demands': len([d for d in self.active_demands if not d['fulfilled']]),
            'episode_time': current_time - self.episode_start_time,
            'van_inventory': self.van_inventory,
            'edge_hits': self.edge_hits
        })
        
        return self._get_obs(), rewards, done, info
    
    def get_manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_best_target(self, agent_idx):
        """Determine best target for an agent considering distance and other agents' targets"""
        agent_pos = self.agents[agent_idx]
        best_target = None
        best_score = float('-inf')
        
        for demand in self.active_demands:
            if demand['fulfilled']:
                continue
                
            # Skip if another agent is already targeting this demand
            # Fix: Compare demand IDs or positions instead of the entire demand object
            is_targeted = False
            for i in range(self.n_agents):
                if i != agent_idx and self.agent_targets[i] is not None:
                    # Compare positions instead of the entire demand object
                    if np.array_equal(self.agent_targets[i]['position'], demand['position']):
                        is_targeted = True
                        break
            
            if is_targeted:
                continue
            
            # Calculate base score based on distance
            distance = self.get_manhattan_distance(agent_pos, demand['position'])
            
            # Add time pressure
            waiting_time = time.time() - demand['creation_time']
            urgency_factor = min(waiting_time / 5.0, 2.0)  # Cap at 2x multiplier
            
            # Calculate score (higher is better)
            score = 100 * urgency_factor - distance
            
            # Bonus if agent has matching item
            if (len(self.agent_inventories[agent_idx]) > 0 and 
                self.agent_inventories[agent_idx][0] == demand['type']):
                score += 50
            
            if score > best_score:
                best_score = score
                best_target = demand
        
        return best_target
    
    def is_target_in_demands(self, target, active_demands):
        """Helper method to check if a target exists in active demands"""
        if target is None:
            return False
        
        for demand in active_demands:
            if np.array_equal(target['position'], demand['position']) and target['type'] == demand['type']:
                return True
        return False
    def render(self):
        # Make sure we have a valid display
        if self.screen is None:
            self.initialize_display()

        self.screen.fill(COLORS['background'])

        # Draw main simulation area
        simulation_surface = pygame.Surface((self.grid_render_size, self.grid_render_size))
        simulation_surface.fill(COLORS['background'])

        # Add action probability visualization colors
        ACTION_COLORS = {
            0: (255, 0, 0, 128),    # Up - Red
            1: (0, 255, 0, 128),    # Down - Green
            2: (0, 0, 255, 128),    # Left - Blue
            3: (255, 255, 0, 128)   # Right - Yellow
        }

        # Draw grid with action probabilities
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_rect = pygame.Rect(
                    j * self.cell_size, 
                    i * self.cell_size,
                    self.cell_size, 
                    self.cell_size
                )
                
                # Draw base grid cell
                pygame.draw.rect(simulation_surface, COLORS['grid'], cell_rect, 1)
                
                # If we have action probabilities, visualize them
                if hasattr(self, 'current_action_probs') and self.current_action_probs is not None:
                    # Get probabilities for this cell
                    flat_index = i * self.grid_size + j
                    if flat_index < len(self.current_action_probs):
                        probs = self.current_action_probs[flat_index]
                        # Split cell into quarters for each action
                        quarter_size = self.cell_size // 2
                        for action in range(4):
                            if action < len(probs) and probs[action] > 0.1:  # Only show significant probabilities
                                color = list(ACTION_COLORS[action])
                                color[3] = int(255 * probs[action])  # Adjust alpha based on probability
                                
                                quarter_surface = pygame.Surface((quarter_size, quarter_size), pygame.SRCALPHA)
                                quarter_surface.fill(color)
                                
                                # Position quarters based on action
                                if action == 0:  # Up
                                    pos = (j * self.cell_size, i * self.cell_size)
                                elif action == 1:  # Down
                                    pos = (j * self.cell_size, i * self.cell_size + quarter_size)
                                elif action == 2:  # Left
                                    pos = (j * self.cell_size, i * self.cell_size)
                                else:  # Right
                                    pos = (j * self.cell_size + quarter_size, i * self.cell_size)
                                
                                simulation_surface.blit(quarter_surface, pos)

        # Rest of the rendering code remains the same...
        # Draw van with inventory counts
        van_x = self.van_pos[1] * self.cell_size + self.cell_size // 2
        van_y = self.van_pos[0] * self.cell_size + self.cell_size // 2
        gfxdraw.aacircle(simulation_surface, van_x, van_y, 
                        self.cell_size // 2, COLORS['van'])
        gfxdraw.filled_circle(simulation_surface, van_x, van_y, 
                            self.cell_size // 2, COLORS['van'])

        # Draw van inventory counts
        inventory_y = van_y - self.cell_size // 2 - 15
        for item, count in self.van_inventory.items():
            color = COLORS[item]
            text = self.small_font.render(str(count), True, color)
            text_rect = text.get_rect(center=(van_x, inventory_y))
            simulation_surface.blit(text, text_rect)
            inventory_y -= 15

        # Draw active demands
        current_time = time.time()
        for demand in self.active_demands:
            if not demand['fulfilled']:
                color = COLORS[demand['type']]
                pos = demand['position']
                x = pos[1] * self.cell_size + self.cell_size // 2
                y = pos[0] * self.cell_size + self.cell_size // 2
                
                age = current_time - demand['creation_time']
                size = min(self.cell_size // 3 + int(age), self.cell_size // 2)
                
                gfxdraw.aacircle(simulation_surface, x, y, size, color)
                gfxdraw.filled_circle(simulation_surface, x, y, size, color)
                
                wait_text = self.small_font.render(f"{int(age)}s", True, COLORS['text'])
                text_rect = wait_text.get_rect(center=(x, y))
                simulation_surface.blit(wait_text, text_rect)

        # Draw agents and their inventories
        for i, (agent_pos, inventory) in enumerate(zip(self.agents, self.agent_inventories)):
            x = agent_pos[1] * self.cell_size + self.cell_size // 2
            y = agent_pos[0] * self.cell_size + self.cell_size // 2
            
            gfxdraw.aacircle(simulation_surface, x, y, 
                            self.cell_size // 4, COLORS['agent'])
            gfxdraw.filled_circle(simulation_surface, x, y, 
                                self.cell_size // 4, COLORS['agent'])
            
            text = self.small_font.render(str(i+1), True, (255, 255, 255))
            text_rect = text.get_rect(center=(x, y))
            simulation_surface.blit(text, text_rect)

            for j, item in enumerate(inventory):
                color = COLORS[item]
                inv_x = x + (j - len(inventory)/2) * 15
                inv_y = y - self.cell_size//2 - 5
                gfxdraw.aacircle(simulation_surface, int(inv_x), int(inv_y), 5, color)
                gfxdraw.filled_circle(simulation_surface, int(inv_x), int(inv_y), 5, color)

        # Draw dashboard
        dashboard_surface = pygame.Surface((self.dashboard_width, self.render_size))
        dashboard_surface.fill(COLORS['panel'])
        
        # Update metrics display
        y_offset = 20
        metrics = [
            ("Episode", f"{self.current_episode}"),
            ("Time", f"{int(time.time() - self.episode_start_time)}s"),
            ("Rewards", f"{self.total_rewards}"),
            ("Total Demands", f"{self.total_demands}"),
            ("Fulfilled", f"{self.fulfilled_demands}"),
            ("Active Demands", f"{len([d for d in self.active_demands if not d['fulfilled']])}"),
            ("Van Inventory", ""),
            ("Edge Hits", f"{self.edge_hits}")
        ]
        
        for label, value in metrics:
            text = self.small_font.render(f"{label}: {value}", True, COLORS['text'])
            dashboard_surface.blit(text, (20, y_offset))
            y_offset += 30
        
        # Display van inventory
        for item, count in self.van_inventory.items():
            text = self.small_font.render(f"  {item}: {count}", True, COLORS[item])
            dashboard_surface.blit(text, (20, y_offset))
            y_offset += 20

        # Blit surfaces to screen
        self.screen.blit(simulation_surface, (0, 0))
        self.screen.blit(dashboard_surface, (self.grid_render_size, 0))
        
        pygame.display.flip()
    
    def reset(self):
        """Reset the environment state"""
        # Reset van inventory
        self.van_inventory = {
            'frozen_food': 25,
            'prepared_meals': 25,
            'meal_kits': 25,
            'fast_food': 25
        }
        
        # Place van in center
        self.van_pos = np.array([self.grid_size // 2, self.grid_size // 2])
        
        # Initialize agents near van
        self.agents = []
        self.agent_inventories = []
        for _ in range(self.n_agents):
            pos = self.van_pos + np.random.randint(-2, 3, size=2)
            pos = np.clip(pos, 0, self.grid_size - 1)
            self.agents.append(pos)
            self.agent_inventories.append([])
        
        # Reset demand tracking
        self.active_demands = []
        self.fulfilled_demands = 0
        self.total_demands = 0
        
        # Reset timing
        self.episode_start_time = time.time()
        self.last_demand_time = self.episode_start_time
        self.episode_steps = 0
        
        # Reset agent-specific tracking
        self.edge_hits = 0
        self.total_rewards = 0
        self.agent_targets = [None] * self.n_agents
        self.target_hold_time = [0] * self.n_agents
        self.previous_distances = [float('inf')] * self.n_agents
        self.last_action = [None] * self.n_agents
        
        return self._get_obs()

    def close(self):
        """Properly close Pygame display"""
        if pygame.display.get_init():
            pygame.display.quit()
        if pygame.get_init():
            pygame.quit()
        self.screen = None
    
    def initialize_display(self):
        """Initialize the Pygame display if it hasn't been created yet"""
        if self.screen is None:
            self.screen = pygame.display.set_mode((self.render_size, self.render_size))
            pygame.display.set_caption("Logistics Delivery Simulation")
        pygame.event.pump()

    # def generate_new_demand(self):
    #         """Generate a new demand if conditions are met, avoiding boundary positions"""
    #         current_time = time.time()
            
    #         # Only generate new demand if we haven't exceeded max_demands
    #         active_unfulfilled = [d for d in self.active_demands if not d['fulfilled']]
    #         if len(active_unfulfilled) < self.max_demands:
    #             # Get all currently occupied positions
    #             occupied_positions = {tuple(d['position']) for d in active_unfulfilled}
    #             occupied_positions.add(tuple(self.van_pos))
                
    #             # Define safe zone boundaries (avoid edge positions)
    #             safe_min = 1  # Minimum safe position
    #             safe_max = self.grid_size - 2  # Maximum safe position
                
    #             # Try to find an unoccupied position
    #             max_attempts = 100  # Prevent infinite loop
    #             attempts = 0
    #             while attempts < max_attempts:
    #                 # Generate position within safe zone
    #                 pos = np.array([
    #                     np.random.randint(safe_min, safe_max + 1),
    #                     np.random.randint(safe_min, safe_max + 1)
    #                 ])
    #                 pos_tuple = tuple(pos)
                    
    #                 if pos_tuple not in occupied_positions:
    #                     # Found a valid position, create the demand
    #                     order_type = np.random.choice(
    #                         list(ORDER_TYPES.keys()),
    #                         p=list(ORDER_TYPES.values())
    #                     )
                        
    #                     self.active_demands.append({
    #                         'position': pos,
    #                         'type': order_type,
    #                         'creation_time': current_time,
    #                         'fulfilled': False
    #                     })
                        
    #                     self.total_demands += 1
    #                     break
                    
    #                 attempts += 1

    def generate_new_demand(self):
        """Generate a new demand if conditions are met, or immediately if no demands exist"""
        current_time = time.time()
        
        # Get count of active unfulfilled demands
        active_unfulfilled = [d for d in self.active_demands if not d['fulfilled']]
        
        # Generate new demand if we have no active demands or haven't exceeded max_demands
        if len(active_unfulfilled) == 0 or (
            len(active_unfulfilled) < self.max_demands and 
            current_time - self.last_demand_time >= 2
        ):
            # Get all currently occupied positions
            occupied_positions = {tuple(d['position']) for d in active_unfulfilled}
            occupied_positions.add(tuple(self.van_pos))
            
            # Define safe zone boundaries (avoid edge positions)
            safe_min = 1  # Minimum safe position
            safe_max = self.grid_size - 2  # Maximum safe position
            
            # Try to find an unoccupied position
            max_attempts = 100  # Prevent infinite loop
            attempts = 0
            while attempts < max_attempts:
                # Generate position within safe zone
                pos = np.array([
                    np.random.randint(safe_min, safe_max + 1),
                    np.random.randint(safe_min, safe_max + 1)
                ])
                pos_tuple = tuple(pos)
                
                if pos_tuple not in occupied_positions:
                    # Found a valid position, create the demand
                    order_type = np.random.choice(
                        list(ORDER_TYPES.keys()),
                        p=list(ORDER_TYPES.values())
                    )
                    
                    self.active_demands.append({
                        'position': pos,
                        'type': order_type,
                        'creation_time': current_time,
                        'fulfilled': False
                    })
                    
                    self.total_demands += 1
                    self.last_demand_time = current_time
                    break
                
                attempts += 1

    def update_metrics(self, episode, last_checkpoint, rewards, steps):
        """Update the dashboard metrics"""
        self.current_episode = episode
        self.last_checkpoint = last_checkpoint
        self.total_rewards = rewards
        self.episode_steps = steps
    
    def _get_obs(self):
            """Create observation for each agent"""
            # Calculate grid features size
            demand_grid_size = self.grid_size * self.grid_size * 4
            
            # Initialize observation array with correct size
            obs = np.zeros(12 + demand_grid_size, dtype=np.float32)
            
            # Add van position (normalized)
            obs[0] = self.van_pos[0] / self.grid_size
            obs[1] = self.van_pos[1] / self.grid_size
            
            # Add agent positions (normalized)
            obs[2] = self.agents[0][0] / self.grid_size
            obs[3] = self.agents[0][1] / self.grid_size
            
            # Add agent's inventory (one-hot)
            if len(self.agent_inventories[0]) > 0:
                item_idx = list(ORDER_TYPES.keys()).index(self.agent_inventories[0][0])
                obs[4 + item_idx] = 1
            
            # Add van inventory information (normalized)
            base_idx = 8
            for i, count in enumerate(self.van_inventory.values()):
                obs[base_idx + i] = count / 25.0  # Normalize by initial quantity
            
            # Add demands information to the grid
            demands_base_idx = base_idx + 4
            for demand in self.active_demands:
                if not demand['fulfilled']:
                    pos = demand['position']
                    type_idx = list(ORDER_TYPES.keys()).index(demand['type'])
                    grid_idx = (pos[0] * self.grid_size + pos[1]) * 4 + type_idx
                    obs[demands_base_idx + grid_idx] = 1
            
            return obs

# DQN Agent
class DQN(nn.Module):
    def __init__(self, observation_space, n_actions):
        super(DQN, self).__init__()
        
        # Calculate total input size from observation space
        input_size = int(np.prod(observation_space.shape))  # Multiply all dimensions together
        
        # Create a fully connected network
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
    
    def forward(self, x):
        # Ensure input is properly flattened
        batch_size = x.size(0) if len(x.size()) > 1 else 1
        x = x.view(batch_size, -1)  # Flatten all dimensions except batch
        return self.network(x)

class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_states.append(None)
            self.dones.append(None)
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        states = np.array([self.states[idx] for idx in indices])
        actions = np.array([self.actions[idx] for idx in indices])
        rewards = np.array([self.rewards[idx] for idx in indices])
        next_states = np.array([self.next_states[idx] for idx in indices])
        dones = np.array([self.dones[idx] for idx in indices])
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.states)

class FastExperienceBuffer:
    def __init__(self, capacity, observation_shape):
        self.capacity = capacity
        shape = (capacity,) + observation_shape
        
        # Pre-allocate numpy arrays for all components
        self.states = np.zeros(shape, dtype=np.float32)
        self.next_states = np.zeros(shape, dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices]),
            torch.from_numpy(self.rewards[indices]),
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )
    
    def __len__(self):
        return self.size

class ExperienceBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.states) < self.capacity:
            self.states.append(None)
            self.actions.append(None)
            self.rewards.append(None)
            self.next_states.append(None)
            self.dones.append(None)
        
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size):
        indices = np.random.choice(len(self.states), batch_size, replace=False)
        
        states = np.array([self.states[idx] for idx in indices])
        actions = np.array([self.actions[idx] for idx in indices])
        rewards = np.array([self.rewards[idx] for idx in indices])
        next_states = np.array([self.next_states[idx] for idx in indices])
        dones = np.array([self.dones[idx] for idx in indices])
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )
    
    def __len__(self):
        return len(self.states)

def evaluate_models(env, models, n_episodes=10, render=True, render_delay=0.1):
    """Run trained models in evaluation mode with proper state processing"""
    device = next(models[0].parameters()).device
    total_rewards = []
    try:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            while not done:
                if render:
                    env.render()
                    time.sleep(render_delay)
                actions = []
                # Process raw state for each agent
                for i, model in enumerate(models):
                    with torch.no_grad():
                        # Extract state components (adjust indices as needed based on your state structure)
                        van_pos = state[0:2]
                        agent_pos = state[2:4]
                        agent_inventory_raw = state[4:8]
                        van_inventory = state[8:12]
                        grid_size = int(np.sqrt((len(state) - 12) / 4))
                        demand_grid = state[12:].reshape(grid_size, grid_size, 4)
                        # Format state as dictionary
                        tensor_state = {
                            'agent_pos': torch.FloatTensor([agent_pos]).to(device),
                            'van_pos': torch.FloatTensor([van_pos]).to(device)
                        }
                        # Extract demands from grid
                        demands = []
                        for i in range(grid_size):
                            for j in range(grid_size):
                                if demand_grid[i, j].any():
                                    for type_idx, has_demand in enumerate(demand_grid[i, j]):
                                        if has_demand:
                                            demands.append({
                                                'position': np.array([i, j]),
                                                'type': list(ORDER_TYPES.keys())[type_idx],
                                                'fulfilled': False
                                            })
                        # Get action
                        q_values = model(tensor_state, demands, agent_inventory_raw.tolist())
                        
                        # Check if q_values is a tuple and handle accordingly
                        if isinstance(q_values, tuple):
                            # If q_values is a tuple, use the first element (assuming it contains the action values)
                            # Adjust this based on what your model actually returns
                            action_values = q_values[0]
                            actions.append(action_values.max(1)[1].item())
                        else:
                            # Original behavior if q_values is a tensor
                            actions.append(q_values.max(1)[1].item())
                
                # Take step in environment
                next_state, rewards, done, _ = env.step(actions)
                total_reward += sum(rewards)
                state = next_state
            total_rewards.append(total_reward)
            print(f"Evaluation Episode {episode + 1}/{n_episodes}, Total Reward: {total_reward:.1f}")
    finally:
        env.close()
    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\nAverage Reward over {n_episodes} episodes: {avg_reward:.1f}")
    return total_rewards
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AttentionDeliveryNetwork(nn.Module):
    def __init__(self, grid_size, n_item_types=4, hidden_dim=256):
        super(AttentionDeliveryNetwork, self).__init__()
        self.grid_size = grid_size
        self.n_item_types = n_item_types
        self.hidden_dim = hidden_dim
        
        self.position_embedding = nn.Linear(2, hidden_dim)
        self.item_embedding = nn.Linear(n_item_types, hidden_dim)
        self.demand_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.van_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.loading_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        self.delivery_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 4)
        )
        
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.phase_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def process_demands(self, demands, agent_pos):
        """Process demand locations with attention relative to agent position"""
        # Ensure agent_pos is 2D [batch_size, 2]
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.unsqueeze(0)
        batch_size = agent_pos.size(0)
        device = agent_pos.device
        
        # Create demand embeddings
        demand_positions = []
        demand_items = []
        
        for demand in demands:
            if not demand['fulfilled']:
                # Calculate relative position to agent (maintaining batch dimension)
                rel_pos = torch.tensor(demand['position'], device=device).float()
                rel_pos = rel_pos.unsqueeze(0).expand(batch_size, -1) - agent_pos
                demand_positions.append(rel_pos)
                
                # One-hot encode item type
                item = torch.zeros(self.n_item_types, device=device)
                item[list(ORDER_TYPES.keys()).index(demand['type'])] = 1
                item = item.unsqueeze(0).expand(batch_size, -1)
                demand_items.append(item)
        
        if not demand_positions:  # No active demands
            # Return zero tensor with correct dimensions [batch_size, 1, hidden_dim]
            return torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        # Stack demands along a new dimension [batch_size, n_demands, features]
        demand_positions = torch.stack(demand_positions, dim=1)
        demand_items = torch.stack(demand_items, dim=1)
        
        # Create embeddings
        pos_embeds = self.position_embedding(demand_positions)  # [batch_size, n_demands, hidden_dim]
        item_embeds = self.item_embedding(demand_items)  # [batch_size, n_demands, hidden_dim]
        
        # Combine embeddings
        demand_embeds = pos_embeds + item_embeds  # [batch_size, n_demands, hidden_dim]
        
        # Apply attention
        attended_demands, _ = self.demand_attention(
            demand_embeds, demand_embeds, demand_embeds
        )
        
        return attended_demands

    def forward(self, state, demands, agent_inventory):
        """Forward pass considering both loading and delivery phases"""
        # Ensure agent_pos is 2D [batch_size, 2]
        agent_pos = state['agent_pos']
        if len(agent_pos.shape) == 1:
            agent_pos = agent_pos.unsqueeze(0)
        
        # Ensure van_pos is 2D [batch_size, 2]
        van_pos = state['van_pos']
        if len(van_pos.shape) == 1:
            van_pos = van_pos.unsqueeze(0)
            
        batch_size = agent_pos.size(0)
        device = agent_pos.device
        
        # Process demands with attention
        demand_features = self.process_demands(demands, agent_pos)
        
        # Process van position with attention
        van_rel_pos = van_pos - agent_pos  # Relative position
        van_embed = self.position_embedding(van_rel_pos)
        
        # Add sequence dimension [batch_size, 1, hidden_dim]
        van_embed = van_embed.unsqueeze(1)
        
        van_features, _ = self.van_attention(
            van_embed, van_embed, van_embed
        )
        
        # Combine features - ensure proper dimensions
        demand_features = demand_features.mean(dim=1)  # [batch_size, hidden_dim]
        van_features = van_features.squeeze(1)  # [batch_size, hidden_dim]
        combined_features = torch.cat([demand_features, van_features], dim=-1)
        
        # Process through networks
        state_encoding = self.state_encoder(combined_features)
        phase_prob = self.phase_detector(state_encoding)
        loading_logits = self.loading_policy(state_encoding)
        delivery_logits = self.delivery_policy(state_encoding)
        
        # Combine policies
        policy_logits = phase_prob * loading_logits + (1 - phase_prob) * delivery_logits
        
        # Get value estimate
        value = self.value_head(state_encoding)
        
        return policy_logits, value

class DeliveryActorCritic(nn.Module):
    def __init__(self, grid_size, n_item_types=4, hidden_dim=256):
        super(DeliveryActorCritic, self).__init__()
        self.network = AttentionDeliveryNetwork(grid_size, n_item_types, hidden_dim)
    
    def forward(self, state, demands, agent_inventory):
        return self.network(state, demands, agent_inventory)
    
    def act(self, state, demands, agent_inventory):
        """Select an action and get value estimate"""
        policy_logits, value = self.forward(state, demands, agent_inventory)
        
        # Convert logits to probabilities and sample action
        action_probs = F.softmax(policy_logits, dim=-1)
        action_dist = Categorical(action_probs)
        action = action_dist.sample()
        
        # Get action log probability for training
        log_prob = action_dist.log_prob(action)
        
        return action, value, log_prob, action_probs

def save_checkpoint(models, optimizers, episode, metrics, filename_prefix='delivery_ac_checkpoint'):
    """Save models, optimizers, and training progress to disk"""
    checkpoint = {
        'episode': episode,
        'models_state_dict': [model.state_dict() for model in models],
        'optimizers_state_dict': [opt.state_dict() for opt in optimizers],
        'metrics': metrics
    }
    path = f'{filename_prefix}_episode_{episode}.pt'
    torch.save(checkpoint, path)
    print(f"Checkpoint saved at episode {episode}: {path}")

def load_checkpoint(checkpoint_path, grid_size, device=None):
    """Load models and optimizers from a checkpoint file"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load with weights_only=False since we need the full checkpoint data
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Initialize models and optimizers
    models = []
    optimizers = []
    
    for model_state in checkpoint['models_state_dict']:
        model = DeliveryActorCritic(grid_size=grid_size).to(device)
        model.load_state_dict(model_state)
        models.append(model)
    
    for opt_state in checkpoint['optimizers_state_dict']:
        optimizer = optim.Adam(models[len(optimizers)].parameters())
        optimizer.load_state_dict(opt_state)
        optimizers.append(optimizer)
    
    return models, optimizers, checkpoint['episode'], checkpoint['metrics']
def train_agents_attention_ac(env, n_episodes=1000, render=False, render_delay=0.1, 
                            checkpoint_freq=5, load_from=None):
    """Train multiple agents using attention-based actor-critic"""
    n_agents = env.n_agents
    grid_size = env.grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize or load models and optimizers
    start_episode = 0
    metrics = {'episode_rewards': [], 'mean_rewards': []}
    
    if load_from:
        models, optimizers, start_episode, metrics = load_checkpoint(load_from, grid_size, device)
        print(f"Loaded checkpoint from episode {start_episode}")
    else:
        models = [DeliveryActorCritic(grid_size=grid_size).to(device) for _ in range(n_agents)]
        optimizers = [optim.Adam(model.parameters(), lr=1e-4) for model in models]
    
    for episode in range(start_episode, n_episodes):
        state = env.reset()
        done = False
        episode_rewards = [0] * n_agents
        episode_values = [[] for _ in range(n_agents)]
        episode_log_probs = [[] for _ in range(n_agents)]
        episode_rewards_list = [[] for _ in range(n_agents)]
        
        while not done:
            if render:
                env.render()
                time.sleep(render_delay)
            
            # Convert state to structured format
            van_pos = state[0:2]
            agent_pos = state[2:4]
            agent_inventory = state[4:8]
            van_inventory = state[8:12]
            demand_grid = state[12:].reshape(grid_size, grid_size, 4)
            
            tensor_state = {
                'agent_pos': torch.FloatTensor([agent_pos]).to(device),
                'van_pos': torch.FloatTensor([van_pos]).to(device)
            }
            
            # Process demands
            demands = []
            for i in range(grid_size):
                for j in range(grid_size):
                    if demand_grid[i, j].any():
                        for type_idx, has_demand in enumerate(demand_grid[i, j]):
                            if has_demand:
                                demands.append({
                                    'position': np.array([i, j]),
                                    'type': list(ORDER_TYPES.keys())[type_idx],
                                    'fulfilled': False
                                })
            
            # Get actions and probabilities for all agents
            actions = []
            all_action_probs = []
            
            for i in range(n_agents):
                action, value, log_prob, action_probs = models[i].act(
                    state=tensor_state,
                    demands=demands,
                    agent_inventory=agent_inventory.tolist()
                )
                actions.append(action.item())
                episode_values[i].append(value)
                episode_log_probs[i].append(log_prob)
                all_action_probs.append(action_probs.detach().cpu().numpy())
            
            # Store action probabilities in environment for visualization
            env.current_action_probs = all_action_probs[0]  # For single agent, use first agent's probs
            
            # Take step in environment
            next_state, rewards, done, _ = env.step(actions)
            
            # Store rewards
            for i in range(n_agents):
                episode_rewards[i] += rewards[i]
                episode_rewards_list[i].append(rewards[i])
            
            state = next_state
        
        # Update models at episode end
        for i in range(n_agents):
            # Convert episode data to tensors
            rewards = torch.FloatTensor(episode_rewards_list[i]).to(device)
            values = torch.cat(episode_values[i])
            log_probs = torch.stack(episode_log_probs[i])
            
            # Calculate returns and advantages
            returns = rewards
            advantages = returns - values.detach()
            
            # Ensure proper shapes
            returns = returns.unsqueeze(-1)
            advantages = advantages.unsqueeze(-1)
            
            # Calculate losses with correct shapes
            actor_loss = -(log_probs * advantages.detach()).mean()
            critic_loss = F.smooth_l1_loss(values, returns)
            
            # Total loss
            total_loss = actor_loss + 0.5 * critic_loss
            
            # Update model
            optimizers[i].zero_grad()
            total_loss.backward()
            optimizers[i].step()
        
        # Store metrics
        metrics['episode_rewards'].append(episode_rewards)
        metrics['mean_rewards'].append(np.mean(episode_rewards))
        
        # Print episode stats
        if episode % 10 == 0:
            mean_reward = np.mean(episode_rewards)
            print(f"Episode {episode}, Average Rewards: {mean_reward:.2f}")
        
        # Save checkpoint
        if episode % checkpoint_freq == 0:
            save_checkpoint(models, optimizers, episode, metrics)
    
    # Save final checkpoint
    save_checkpoint(models, optimizers, n_episodes-1, metrics, filename_prefix='delivery_ac_final')
    
    return models, metrics

def main():
    env = DeliveryEnv(grid_size=5, n_agents=1)
    trained_models = train_agents_attention_ac(env, n_episodes=1000, render=True, render_delay=0.1)

if __name__ == "__main__":
    main()