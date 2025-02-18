from delivery_env import DeliveryEnv, load_checkpoint, evaluate_models, train_agents_attention_ac

# Initialize environment
env = DeliveryEnv(grid_size=5, n_agents=1)

# Load models from a checkpoint
loaded_models, optimizers, episode, metrics = load_checkpoint(
    'delivery_ac_checkpoint_episode_25.pt', 
    grid_size=5
)
print(f"Loaded models from episode {episode}")

# Evaluate the loaded models
rewards = evaluate_models(env, loaded_models, n_episodes=5, render=True)

# Continue training if needed
# models = train_agents_attention_ac(env, n_episodes=1000-episode, load_from='delivery_ac_checkpoint_episode_5.pt')