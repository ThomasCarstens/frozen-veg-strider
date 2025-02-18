

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
    'frozen_food': 0.25,
    'prepared_meals': 0.25,
    'meal_kits': 0.25,
    'fast_food': 0.25
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
    def __init__(self, grid_size=20, n_agents=4, max_items=4, max_demands=50, episode_timeout=100, render_size=1000):
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
            'frozen_food': 25,
            'prepared_meals': 25,
            'meal_kits': 25,
            'fast_food': 25
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
        
        # Generate new demands
        if current_time - self.last_demand_time >= 0.5:
            self.generate_new_demand()
            self.last_demand_time = current_time
        
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
        
        # Process actions
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
            
            new_pos = self.agents[i] + np.array(direction)
            rewards[i] -= 0.5 # penalty over time
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
                                rewards[i] += 20.0
                            else:
                                available_items = [item for item, count in self.van_inventory.items() if count > 0]
                                if available_items:
                                    item = np.random.choice(available_items)
                                    self.agent_inventories[i].append(item)
                                    self.van_inventory[item] -= 1
                                    rewards[i] += 15.0
                
                # Check for deliveries at new position (including opportunistic deliveries)
                if len(self.agent_inventories[i]) > 0:
                    delivered_item = self.agent_inventories[i][0]
                    delivery_made = False
                    
                    # Check all unfulfilled demands at current position
                    for demand in self.active_demands:
                        if (not demand['fulfilled'] and 
                            np.array_equal(new_pos, demand['position']) and 
                            demand['type'] == delivered_item):
                            
                            delivery_time = current_time - demand['creation_time']
                            time_bonus = max(100 - delivery_time * 10, 20)
                            
                            # Higher reward if this was the targeted demand
                            if target is not None and np.array_equal(demand['position'], target['position']):
                                rewards[i] += 100.0 + time_bonus
                            else:  # Smaller reward for opportunistic delivery
                                rewards[i] += 50.0 + time_bonus
                            
                            demand['fulfilled'] = True
                            self.agent_inventories[i].remove(delivered_item)
                            self.fulfilled_demands += 1
                            delivery_made = True
                            
                            # Clear target if we just delivered to it
                            if target is not None and np.array_equal(demand['position'], target['position']):
                                self.agent_targets[i] = None
                            
                            break  # Only one delivery per move
                    
                    # Update agent target if delivery was made
                    if delivery_made and self.agent_targets[i] is None:
                        self.agent_targets[i] = self.get_best_target(i)
                        self.target_hold_time[i] = 0
                
                # Movement rewards based on target
                # ASK CLAUDE: DO AGENTS LEARN TO ADAPT THEIR TARGETS BY RL? BY EXTENSION DRONES MIGHT DO THE SAME
                if target is not None:
                    new_distance = self.get_manhattan_distance(new_pos, target['position'])
                    old_distance = self.get_manhattan_distance(agent_pos, target['position'])
                    
                    if new_distance < old_distance:
                        commitment_bonus = min(self.target_hold_time[i] * 0.1, 1.0)
                        rewards[i] += 0.5 * (1 + commitment_bonus)
                        rewards[i] += 0
                    elif new_distance > old_distance:
                        rewards[i] -= 0.7
                
                # if len(self.agent_inventories[i]) > 0:
                #     if action != self.last_action[i]:
                #         rewards[i] -= 0.2
            
            self.last_action[i] = action
        
        # Update episode stats
        self.total_rewards += sum(rewards)
        self.episode_steps += 1
        
        # Update info
        info.update({
            'fulfilled_demands': self.fulfilled_demands,
            'total_demands': self.total_demands,
            'active_demands': len([d for d in self.active_demands if not d['fulfilled']]),
            'episode_time': current_time - self.episode_start_time,
            'van_inventory': self.van_inventory
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

        # Draw grid
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                pygame.draw.rect(simulation_surface, COLORS['grid'],
                            (i * self.cell_size, j * self.cell_size,
                                self.cell_size, self.cell_size), 1)

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
                
                # Draw demand circle with size based on age
                age = current_time - demand['creation_time']
                size = min(self.cell_size // 3 + int(age), self.cell_size // 2)
                
                gfxdraw.aacircle(simulation_surface, x, y, size, color)
                gfxdraw.filled_circle(simulation_surface, x, y, size, color)
                
                # Draw waiting time
                wait_text = self.small_font.render(f"{int(age)}s", True, COLORS['text'])
                text_rect = wait_text.get_rect(center=(x, y))
                simulation_surface.blit(wait_text, text_rect)

        # Draw agents and their inventories
        for i, (agent_pos, inventory) in enumerate(zip(self.agents, self.agent_inventories)):
            x = agent_pos[1] * self.cell_size + self.cell_size // 2
            y = agent_pos[0] * self.cell_size + self.cell_size // 2
            
            # Draw agent
            gfxdraw.aacircle(simulation_surface, x, y, 
                            self.cell_size // 4, COLORS['agent'])
            gfxdraw.filled_circle(simulation_surface, x, y, 
                                self.cell_size // 4, COLORS['agent'])
            
            # Draw agent number
            text = self.small_font.render(str(i+1), True, (255, 255, 255))
            text_rect = text.get_rect(center=(x, y))
            simulation_surface.blit(text, text_rect)

            # Draw inventory
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
            ("Edge Hits", f"{self.edge_hits}"),
            
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

    def generate_new_demand(self):
            """Generate a new demand if conditions are met, avoiding boundary positions"""
            current_time = time.time()
            
            # Only generate new demand if we haven't exceeded max_demands
            active_unfulfilled = [d for d in self.active_demands if not d['fulfilled']]
            if len(active_unfulfilled) < self.max_demands:
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
            # Calculate the correct observation space size
            # Base features: van position (2), agent position (2), agent inventory one-hot (4)
            base_size = 8
            # Van inventory features (4 types)
            van_inventory_size = 4
            # Demand grid features (grid_size * grid_size cells * 4 order types)
            demand_grid_size = self.grid_size * self.grid_size * 4
            
            # Total observation space size
            total_size = base_size + van_inventory_size + demand_grid_size
            
            obs = np.zeros(total_size, dtype=np.float32)
            
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
            
            # Add demands information
            demands_base_idx = base_idx + 4
            for demand in self.active_demands:
                if not demand['fulfilled']:
                    pos = demand['position']
                    type_idx = list(ORDER_TYPES.keys()).index(demand['type'])
                    idx = demands_base_idx + (pos[0] * self.grid_size + pos[1]) * 4 + type_idx
                    if idx < total_size:  # Add bounds check
                        obs[idx] = 1
            
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


def save_checkpoint(models, episode, filename_prefix='abb_checkpoint'):
    """Save models and training progress to disk"""
    checkpoint = {
        'episode': episode,
        'models': [model.state_dict() for model in models]
    }
    torch.save(checkpoint, f'{filename_prefix}_episode_{episode}.pt')
    print(f"Checkpoint saved at episode {episode}")

def load_checkpoint(env, filename, device=None):
    """Load models from a checkpoint file with architecture compatibility checks"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint = torch.load(filename, map_location=device)
    models = []
    
    # Get observation shape from environment
    obs_shape = env.observation_space
    
    # Check if the checkpoint was saved with DQNWithTarget
    uses_target_network = any('target_network' in key for key in checkpoint['models'][0].keys())
    
    for state_dict in checkpoint['models']:
        if uses_target_network:
            # Create DQNWithTarget model
            model = DQNWithTarget(obs_shape, env.action_space.n).to(device)
        else:
            # Create regular DQN model
            model = DQN(obs_shape, env.action_space.n).to(device)
        
        # Filter state dict to match current model architecture
        filtered_state_dict = {}
        model_state_dict = model.state_dict()
        
        for key, value in state_dict.items():
            if key in model_state_dict:
                if model_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Warning: Skipping parameter {key} due to shape mismatch. "
                          f"Checkpoint: {value.shape}, Model: {model_state_dict[key].shape}")
        
        # Load compatible weights
        model.load_state_dict(filtered_state_dict, strict=False)
        model.eval()
        models.append(model)
    
    return models, checkpoint['episode']

def evaluate_models(env, models, n_episodes=10, render=True, render_delay=0.1):
    """Run trained models in evaluation mode"""
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
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                
                # Get actions from all agents
                for i, model in enumerate(models):
                    with torch.no_grad():
                        q_values = model(state_tensor)
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

def train_agents(env, n_episodes=1000, render=True, render_delay=0.01, checkpoint_freq=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models and optimizers for each agent
    models = []
    optimizers = []
    memory = []
    
    # Hyperparameters
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 0.97
    batch_size = 64
    learning_rate = 3e-4
    update_frequency = 4
    min_replay_size = 1000
    
    # Calculate total size for experience buffer
    obs_size = int(np.prod(env.observation_space.shape))
    
    for _ in range(env.n_agents):
        model = DQN(env.observation_space, env.action_space.n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        models.append(model)
        optimizers.append(optimizer)
        memory.append(FastExperienceBuffer(5000, (obs_size,)))
    
    total_steps = 0
    steps = 0
    
    try:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            episode_steps = 0
            
            while not done:
                if render and episode % 10 == 0:
                    env.render()
                    time.sleep(render_delay)

                actions = []
                epsilon = max(epsilon_final, epsilon_start * (epsilon_decay ** episode))
                state_tensor = torch.FloatTensor(state).to(device)
                
                # Get actions from all agents
                for i in range(env.n_agents):
                    if random.random() < epsilon:
                        actions.append(random.randint(0, env.action_space.n - 1))
                    else:
                        with torch.no_grad():
                            q_values = models[i](state_tensor.unsqueeze(0))
                            actions.append(q_values.max(1)[1].item())
                
                # Take step in environment
                next_state, rewards, done, info = env.step(actions)
                total_reward += sum(rewards)
                
                # Store experience for each agent
                for i in range(env.n_agents):
                    memory[i].push(state, actions[i], rewards[i], next_state, done)
                
                # Training step
                if total_steps % update_frequency == 0 and all(len(m) >= min_replay_size for m in memory):
                    for i in range(env.n_agents):
                        batch = memory[i].sample(batch_size)
                        batch = [b.to(device) for b in batch]
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
                        
                        current_q_values = models[i](batch_states).gather(1, batch_actions.unsqueeze(1))
                        with torch.no_grad():
                            max_next_q_values = models[i](batch_next_states).max(1)[0]
                            target_q_values = batch_rewards + (1 - batch_dones) * 0.99 * max_next_q_values
                        
                        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                        optimizers[i].zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=1.0)
                        optimizers[i].step()
                
                state = next_state
                total_steps += 1
                episode_steps += 1
                steps += 1
                
                env.update_metrics(
                    episode=episode,
                    last_checkpoint=episode - (episode % checkpoint_freq),
                    rewards=total_reward,
                    steps=steps
                )
            
            if done and info.get('timeout', False):
                print(f"Episode {episode} timeout after {info.get('elapsed_time', 0):.1f} seconds")
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Steps: {episode_steps}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
            
            if (episode + 1) % checkpoint_freq == 0:
                save_checkpoint(models, episode + 1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(models, episode + 1, filename_prefix='interrupt')
        print("You can resume training or evaluate the models using the saved checkpoint.")
    
    finally:
        env.close()
    
    return models

class SharedExperienceBuffer:
    def __init__(self, capacity, observation_shape, n_agents):
        self.capacity = capacity
        shape = (capacity,) + observation_shape
        
        # Pre-allocate numpy arrays for all components
        self.states = np.zeros(shape, dtype=np.float32)
        self.next_states = np.zeros(shape, dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents), dtype=np.int64)  # Store actions for all agents
        self.rewards = np.zeros((capacity, n_agents), dtype=np.float32)  # Store rewards for all agents
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.position = 0
        self.size = 0
        
    def push(self, state, actions, rewards, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = actions
        self.rewards[self.position] = rewards
        self.next_states[self.position] = next_state
        self.dones[self.position] = done
        
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        
    def sample(self, batch_size, agent_idx):
        """Sample experience for a specific agent"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            torch.from_numpy(self.states[indices]),
            torch.from_numpy(self.actions[indices, agent_idx]),  # Get actions for specific agent
            torch.from_numpy(self.rewards[indices, agent_idx]),  # Get rewards for specific agent
            torch.from_numpy(self.next_states[indices]),
            torch.from_numpy(self.dones[indices])
        )
    
    def __len__(self):
        return self.size

def train_agents_shared_experience(env, n_episodes=1000, render=True, render_delay=0.01, checkpoint_freq=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize models and optimizers for each agent
    models = []
    optimizers = []
    
    # Hyperparameters
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 0.97
    batch_size = 64
    learning_rate = 3e-4
    update_frequency = 4
    min_replay_size = 1000
    
    # Calculate total size for experience buffer
    obs_size = int(np.prod(env.observation_space.shape))
    
    # Create single shared memory buffer for all agents
    shared_memory = SharedExperienceBuffer(10000, (obs_size,), env.n_agents)
    
    for _ in range(env.n_agents):
        model = DQN(env.observation_space, env.action_space.n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        models.append(model)
        optimizers.append(optimizer)
    
    total_steps = 0
    steps = 0
    
    try:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            episode_steps = 0
            
            while not done:
                if render and episode % 10 == 0:
                    env.render()
                    time.sleep(render_delay)

                actions = []
                epsilon = max(epsilon_final, epsilon_start * (epsilon_decay ** episode))
                state_tensor = torch.FloatTensor(state).to(device)
                
                # Get actions from all agents
                for i in range(env.n_agents):
                    if random.random() < epsilon:
                        actions.append(random.randint(0, env.action_space.n - 1))
                    else:
                        with torch.no_grad():
                            q_values = models[i](state_tensor.unsqueeze(0))
                            actions.append(q_values.max(1)[1].item())
                
                # Take step in environment
                next_state, rewards, done, info = env.step(actions)
                total_reward += sum(rewards)
                
                # Store experience in shared buffer
                shared_memory.push(state, actions, rewards, next_state, done)
                
                # Training step
                if total_steps % update_frequency == 0 and len(shared_memory) >= min_replay_size:
                    # Each agent learns from shared experience
                    for i in range(env.n_agents):
                        batch = shared_memory.sample(batch_size, i)
                        batch = [b.to(device) for b in batch]
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
                        
                        current_q_values = models[i](batch_states).gather(1, batch_actions.unsqueeze(1))
                        with torch.no_grad():
                            max_next_q_values = models[i](batch_next_states).max(1)[0]
                            target_q_values = batch_rewards + (1 - batch_dones) * 0.99 * max_next_q_values
                        
                        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
                        optimizers[i].zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=1.0)
                        optimizers[i].step()
                
                state = next_state
                total_steps += 1
                episode_steps += 1
                steps += 1
                
                env.update_metrics(
                    episode=episode,
                    last_checkpoint=episode - (episode % checkpoint_freq),
                    rewards=total_reward,
                    steps=steps
                )
            
            if episode % 10 == 0:
                print(f"Episode {episode}, Steps: {episode_steps}, Total Reward: {total_reward:.1f}, Epsilon: {epsilon:.3f}")
            
            if (episode + 1) % checkpoint_freq == 0:
                save_checkpoint(models, episode + 1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(models, episode + 1, filename_prefix='interrupt')
        print("You can resume training or evaluate the models using the saved checkpoint.")
    
    finally:
        env.close()
    
    return models

def train_agents_stable(env, n_episodes=1000, render=True, render_delay=0.01, checkpoint_freq=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Checkpointing every {checkpoint_freq} episodes")
    
    # Initialize models and optimizers
    models = []
    optimizers = []
    
    # Hyperparameters
    epsilon_start = 1.0
    epsilon_final = 0.1
    epsilon_decay = 0.995  # Slower decay
    batch_size = 128      # Larger batch size
    learning_rate = 1e-4  # Lower learning rate
    update_frequency = 4
    target_update_frequency = 1000  # Update target network every 1000 steps
    min_replay_size = 5000  # Larger minimum replay size
    gamma = 0.99  # Discount factor
    
    # Calculate observation size and initialize shared memory
    obs_size = int(np.prod(env.observation_space.shape))
    shared_memory = SharedExperienceBuffer(50000, (obs_size,), env.n_agents)  # Larger buffer
    
    # Initialize models and optimizers for each agent
    for _ in range(env.n_agents):
        model = DQNWithTarget(env.observation_space, env.action_space.n).to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        models.append(model)
        optimizers.append(optimizer)
    
    total_steps = 0
    episode_rewards = []  # Track rewards for reporting
    best_avg_reward = float('-inf')  # Track best performance for model saving
    
    try:
        for episode in range(n_episodes):
            state = env.reset()
            total_reward = 0
            done = False
            episode_steps = 0
            
            while not done:
                if render and episode % 10 == 0:
                    env.render()
                    time.sleep(render_delay)

                actions = []
                epsilon = max(epsilon_final, epsilon_start * (epsilon_decay ** episode))
                state_tensor = torch.FloatTensor(state).to(device)
                
                # Get actions from all agents
                for i in range(env.n_agents):
                    if random.random() < epsilon:
                        actions.append(random.randint(0, env.action_space.n - 1))
                    else:
                        with torch.no_grad():
                            q_values = models[i](state_tensor.unsqueeze(0))
                            actions.append(q_values.max(1)[1].item())
                
                # Take step in environment
                next_state, rewards, done, info = env.step(actions)
                total_reward += sum(rewards)
                
                # Store experience in shared buffer
                shared_memory.push(state, actions, rewards, next_state, done)
                
                # Training step
                if total_steps % update_frequency == 0 and len(shared_memory) >= min_replay_size:
                    for i in range(env.n_agents):
                        batch = shared_memory.sample(batch_size, i)
                        batch = [b.to(device) for b in batch]
                        batch_states, batch_actions, batch_rewards, batch_next_states, batch_dones = batch
                        
                        # Scale rewards for training stability while preserving relative magnitudes
                        batch_rewards = batch_rewards / 100.0  # Scale down rewards but maintain ratios
                        
                        # Get current Q values
                        current_q_values = models[i](batch_states).gather(1, batch_actions.unsqueeze(1))
                        
                        # Get next Q values from target network
                        with torch.no_grad():
                            next_q_values = models[i].get_target_values(batch_next_states)
                            max_next_q_values = next_q_values.max(1)[0]
                            target_q_values = batch_rewards + (1 - batch_dones) * gamma * max_next_q_values
                        
                        # Huber loss for robustness to outliers
                        loss = nn.SmoothL1Loss()(current_q_values.squeeze(), target_q_values)
                        
                        # Optimization step
                        optimizers[i].zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(models[i].parameters(), max_norm=1.0)
                        optimizers[i].step()
                
                # Update target networks periodically
                if total_steps % target_update_frequency == 0:
                    for model in models:
                        model.update_target_network()
                
                state = next_state
                total_steps += 1
                episode_steps += 1
                
                # Update environment metrics
                env.update_metrics(
                    episode=episode,
                    last_checkpoint=episode - (episode % checkpoint_freq),
                    rewards=total_reward,
                    steps=total_steps
                )
            
            # End of episode processing
            episode_rewards.append(total_reward)
            
            # Calculate running averages
            if len(episode_rewards) >= 10:
                avg_reward = sum(episode_rewards[-10:]) / 10
                if avg_reward > best_avg_reward:
                    print(avg_reward, 'is better than the best avg reward, ', best_avg_reward)
                    best_avg_reward = avg_reward
                    # Save best performing models
                    save_checkpoint(models, episode + 1, filename_prefix='best_model')
            else:
                avg_reward = total_reward
            
            # Print progress
            if episode % 10 == 0:
                print(f"Episode {episode}")
                print(f"  Steps: {episode_steps}")
                print(f"  Reward: {total_reward:.1f}")
                print(f"  Avg Reward (last 10): {avg_reward:.1f}")
                print(f"  Epsilon: {epsilon:.3f}")
                print(f"  Edge Hits: {env.edge_hits}")
            
            # Regular checkpointing
            if (episode + 1) % checkpoint_freq == 0:
                save_checkpoint(models, episode + 1)
    
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        save_checkpoint(models, episode + 1, filename_prefix='interrupt')
        print("Checkpoint saved. You can resume training or evaluate the models using the saved checkpoint.")
    
    finally:
        env.close()
    
    return models

class DQNWithTarget(nn.Module):
    def __init__(self, observation_space, n_actions):
        super(DQNWithTarget, self).__init__()
        input_size = int(np.prod(observation_space.shape))
        
        # Main network without batch normalization for better stability
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        # Target network with the same architecture
        self.target_network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )
        
        self.update_target_network()
    
    def forward(self, x):
        batch_size = x.size(0) if len(x.size()) > 1 else 1
        x = x.view(batch_size, -1)
        return self.network(x)
    
    def get_target_values(self, x):
        batch_size = x.size(0) if len(x.size()) > 1 else 1
        x = x.view(batch_size, -1)
        return self.target_network(x)
    
    def update_target_network(self):
        self.target_network.load_state_dict(self.network.state_dict())

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class AttentionDeliveryNetwork(nn.Module):
    def __init__(self, grid_size, n_item_types=4, hidden_dim=256):
        super(AttentionDeliveryNetwork, self).__init__()
        # Previous initialization code remains the same
        self.grid_size = grid_size
        self.n_item_types = n_item_types
        self.hidden_dim = hidden_dim
        
        self.position_embedding = nn.Linear(2, hidden_dim)
        self.item_embedding = nn.Linear(n_item_types, hidden_dim)
        self.demand_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        self.van_attention = nn.MultiheadAttention(hidden_dim, 8, batch_first=True)
        
        # Rest of the initialization remains the same
        self.state_encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.loading_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
        )
        
        self.delivery_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 5)
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
        batch_size = agent_pos.size(0)
        device = agent_pos.device
        
        # Create demand embeddings
        demand_positions = []
        demand_items = []
        
        for demand in demands:
            if not demand['fulfilled']:
                # Calculate relative position to agent
                rel_pos = torch.tensor(demand['position'], device=device).float() - agent_pos
                demand_positions.append(rel_pos)
                
                # One-hot encode item type
                item = torch.zeros(self.n_item_types, device=device)
                item[list(ORDER_TYPES.keys()).index(demand['type'])] = 1
                demand_items.append(item)
        
        if not demand_positions:  # No active demands
            # Return zero tensor with correct batch dimension
            return torch.zeros(batch_size, 1, self.hidden_dim, device=device)
        
        # Convert to tensors and expand batch dimension
        demand_positions = torch.stack(demand_positions).unsqueeze(0).expand(batch_size, -1, -1)
        demand_items = torch.stack(demand_items).unsqueeze(0).expand(batch_size, -1, -1)
        
        # Create embeddings
        pos_embeds = self.position_embedding(demand_positions)
        item_embeds = self.item_embedding(demand_items)
        
        # Combine embeddings
        demand_embeds = pos_embeds + item_embeds
        
        # Apply attention
        attended_demands, _ = self.demand_attention(
            demand_embeds, demand_embeds, demand_embeds
        )
        
        return attended_demands

    def forward(self, state, demands, agent_inventory):
        """Forward pass considering both loading and delivery phases"""
        batch_size = state['agent_pos'].size(0)
        device = state['agent_pos'].device
        
        # Process demands with attention
        demand_features = self.process_demands(demands, state['agent_pos'])
        
        # Process van position with attention
        van_pos = state['van_pos'] - state['agent_pos']  # Relative position
        van_embed = self.position_embedding(van_pos)
        
        # Expand van embedding to match batch size and add sequence dimension
        van_embed = van_embed.unsqueeze(1)  # Add sequence dimension
        
        van_features, _ = self.van_attention(
            van_embed, van_embed, van_embed
        )
        
        # Combine features
        combined_features = torch.cat([demand_features, van_features], dim=-1)
        state_encoding = self.state_encoder(combined_features)
        
        # Get policy outputs
        phase_prob = self.phase_detector(state_encoding)
        loading_logits = self.loading_policy(state_encoding)
        delivery_logits = self.delivery_policy(state_encoding)
        
        # Combine policies based on detected phase
        policy_logits = phase_prob * loading_logits + (1 - phase_prob) * delivery_logits
        
        # Calculate value estimate
        value = self.value_head(state_encoding)
        
        return policy_logits, value

class DeliveryActorCritic(nn.Module):
    def __init__(self, grid_size, n_item_types=4, hidden_dim=256):
        super(DeliveryActorCritic, self).__init__()
        self.network = AttentionDeliveryNetwork(grid_size, n_item_types, hidden_dim)
    
    def forward(self, state, demands, agent_inventory):
        """
        Forward pass of the actor-critic network
        
        Args:
            state (dict): Current state including agent and van positions
            demands (list): List of active demands
            agent_inventory: Current items being carried
            
        Returns:
            tuple: (action_logits, value)
                - action_logits: Policy logits for action selection
                - value: Value estimate for current state
        """
        return self.network(state, demands, agent_inventory)
    
    def act(self, state, demands, agent_inventory):
        """
        Select an action and get value estimate
        
        Args:
            state (dict): Current state including agent and van positions
            demands (list): List of active demands
            agent_inventory: Current items being carried
            
        Returns:
            tuple: (action, value)
                - action: Selected action index
                - value: Value estimate for current state
        """
        with torch.no_grad():
            policy_logits, value = self.forward(state, demands, agent_inventory)
            
            # Convert logits to probabilities and sample action
            action_probs = F.softmax(policy_logits, dim=-1)
            action_dist = Categorical(action_probs)
            action = action_dist.sample()
            
            # Get action log probability for training
            log_prob = action_dist.log_prob(action)
            
            return action, value, log_prob
    
    def evaluate_actions(self, state, demands, agent_inventory, actions):
        """
        Evaluate actions for training
        
        Args:
            state (dict): Current state including agent and van positions
            demands (list): List of active demands
            agent_inventory: Current items being carried
            actions: Tensor of actions that were taken
            
        Returns:
            tuple: (log_probs, values, entropy)
                - log_probs: Log probabilities of taken actions
                - values: Value estimates
                - entropy: Policy entropy for exploration
        """
        policy_logits, values = self.forward(state, demands, agent_inventory)
        
        action_probs = F.softmax(policy_logits, dim=-1)
        action_dist = Categorical(action_probs)
        
        log_probs = action_dist.log_prob(actions)
        entropy = action_dist.entropy()
        
        return log_probs, values, entropy

def train_agents_attention_ac(env, n_episodes=1000, render=False, render_delay=0.1):
    """
    Train multiple agents using attention-based actor-critic
    """
    n_agents = env.n_agents
    grid_size = env.grid_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize models for each agent
    models = [DeliveryActorCritic(grid_size=grid_size).to(device) for _ in range(n_agents)]
    optimizers = [optim.Adam(model.parameters(), lr=1e-4) for model in models]
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_rewards = [0] * n_agents
        
        while not done:
            if render:
                env.render()
                time.sleep(render_delay)
            
            # Convert state to tensor format - handle different possible state structures
            tensor_state = {
                'agent_pos': torch.FloatTensor(state['agent_positions']).to(device),
                'van_pos': torch.FloatTensor(state['van_positions']).to(device)
            }
            
            # Get actions for all agents
            actions = []
            values = []
            log_probs = []
            
            for i in range(n_agents):
                # Create individual agent observation
                agent_state = {
                    'agent_pos': tensor_state['agent_pos'][i].unsqueeze(0),  # Add batch dimension
                    'van_pos': tensor_state['van_pos'][i].unsqueeze(0)       # Add batch dimension
                }
                
                action, value, log_prob = models[i].act(
                    state=agent_state,
                    demands=state['demands'],
                    agent_inventory=state['agent_inventories'][i]
                )
                actions.append(action.item())
                values.append(value)
                log_probs.append(log_prob)
            
            # Take actions in environment
            next_state, rewards, done, _ = env.step(actions)
            
            # Store rewards
            for i in range(n_agents):
                episode_rewards[i] += rewards[i]
            
            # Update state
            state = next_state
            
            # If episode is done, update models
            if done:
                for i in range(n_agents):
                    reward_tensor = torch.FloatTensor([episode_rewards[i]]).to(device)
                    value_tensor = values[i]
                    advantage = reward_tensor - value_tensor.detach()
                    
                    # Calculate losses
                    actor_loss = -log_probs[i] * advantage
                    critic_loss = F.smooth_l1_loss(value_tensor, reward_tensor)
                    
                    # Total loss
                    total_loss = actor_loss + 0.5 * critic_loss
                    
                    # Update model
                    optimizers[i].zero_grad()
                    total_loss.backward()
                    optimizers[i].step()
        
        # Print episode stats
        if episode % 10 == 0:
            print(f"Episode {episode}, Average Rewards: {np.mean(episode_rewards):.2f}")
    
    return models

def main():
    env = DeliveryEnv(grid_size=20, n_agents=4)
    trained_models = train_agents_attention_ac(env, n_episodes=1000, render=True, render_delay=0.1)

if __name__ == "__main__":
    main()