import gym
import torch
import wandb
from agent import Agent, Policy
from env.custom_hopper import *


def main():
    # 🔄 Avvia wandb
    wandb.init(
        project="reinforce-hopper",
        config={
            "env_id": "CustomHopper-source-v0",
            "n_episodes": 100_000,
            "print_every": 50,
            "gamma": 0.99,
            "lr": 1e-3,
            "baseline": 20
        }
    )
    config = wandb.config

    # Ambiente
    env = gym.make(config.env_id)
    print("State space:", env.observation_space)
    print("Action space:", env.action_space)
    print("Dynamics parameters:", env.get_parameters())

    observation_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    # Inizializza policy e agente
    policy = Policy(observation_dim, action_dim)
    agent = Agent(policy, device="cpu")  # usa config.device se vuoi

    train_reward_avg = 0.0

    for episode in range(config.n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, done, _ = env.step(action.detach().cpu().numpy())

            agent.store_outcome(state, next_state, log_prob, reward, done)
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Aggiorna la policy dopo l’episodio
        agent.update_policy()

        # Log su wandb
        wandb.log({
            "train/episode_return": episode_reward,
			"train/episode_length": episode_length,
            "train/episode": episode
        })

        train_reward_avg += episode_reward

        if (episode + 1) % config.print_every == 0:
            avg = train_reward_avg / config.print_every
            print(f"Episode {episode+1} | Avg Return: {avg:.2f}")
            wandb.log({"train/avg_return": avg})
            train_reward_avg = 0.0

    # Salva modello
    torch.save(agent.policy.state_dict(), "reinforce_model.mdl")
    wandb.save("reinforce_model.mdl")
    wandb.finish()


if __name__ == '__main__':
    main()