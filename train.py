from delivery_env import DeliveryEnv, load_checkpoint, evaluate_models, train_agents_attention_ac

# Train with visualization
env = DeliveryEnv(grid_size=5, n_agents=1)
models = train_agents_attention_ac(env, n_episodes=1000, render=True, render_delay=0.1)
# models = train_agents_stable(env, n_episodes=1000, render=True, render_delay=0.1)



# Or run without training to just watch the agents

# env = DeliveryEnv()
# state = env.reset()
# done = False
# while not done:
#     env.render()
#     time.sleep(0.1)
#     actions = [env.action_space.sample() for _ in range(env.n_agents)]
#     state, rewards, done, _ = env.step(actions)
# env.close()


# Load models from a checkpoint

# env = DeliveryEnv(grid_size=20, n_agents=10)
# loaded_models, episode = load_checkpoint(env, '10s_checkpoint_episode_5.pt')
# print(f"Loaded models from episode {episode}")
# rewards = evaluate_models(env, loaded_models, n_episodes=5, render=True)

# # Continue training from the loaded checkpoint

# env = DeliveryEnv(grid_size=20, n_agents=4)
# loaded_models, episode = load_checkpoint(env, 'interrupt_episode_22.pt')
# models = train_agents(env, n_episodes=1000-episode)  # Train remaining episodes