"""Train an RL agent on the OpenAI Gym Hopper environment using
   REINFORCE and Actor-critic algorithms
"""
import argparse
import torch
import gym
import wandb

from env.custom_hopper import *
from agent import Agent, Policy


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-episodes', default=40000, type=int, help='Number of training episodes')
    parser.add_argument('--print-every', default=10, type=int, help='Print info every <> episodes')
    parser.add_argument('--device', default='cpu', type=str, help='network device [cpu, cuda]')

    return parser.parse_args()


args = parse_args()


def main():
    # 🌐 Start WandB run
    wandb.init(
        project="actor-critic-hopper",
        config={
            "algorithm": "ActorCritic",
            "env_id": "CustomHopper-source-v0",
            "n_episodes": args.n_episodes,
            "print_every": args.print_every,
            "gamma": 0.99,
            "lr": 1e-3
        }
    )

    env = gym.make(wandb.config.env_id)
    print('Action space:', env.action_space)
    print('State space:', env.observation_space)
    print('Dynamics parameters:', env.get_parameters())

    render = True

    observation_space_dim = env.observation_space.shape[-1]
    action_space_dim = env.action_space.shape[-1]

    policy = Policy(observation_space_dim, action_space_dim)
    agent = Agent(policy, device=args.device)

    train_reward_avg = 0.0

    for episode in range(wandb.config.n_episodes):
        done = False
        train_reward = 0
        episode_length = 0
        episode_reward = 0


        state = env.reset()
        while not done:
            action, action_probabilities = agent.get_action(state)
            previous_state = state
            state, reward, done, info = env.step(action.detach().cpu().numpy())

            # Update policy after each timestep (BIG difference with REINFORCE, which updates the policy after each episode is over)
            agent.update_policy()

            if render:
                env.render()

            agent.store_outcome(previous_state, state, action_probabilities, reward, done)

            episode_reward += reward
            episode_length += 1

            train_reward += reward



        # Log to wandb every episode
        wandb.log({"train/episode_return": train_reward, "train/episode": episode})
        wandb.log({
        "train/episode_return": episode_reward,
        "train/episode_length": episode_length,
    })

        # Optional print
        if (episode + 1) % wandb.config.print_every == 0:
            avg = train_reward_avg / wandb.config.print_every
            print(f"Episode {episode+1} | Average return: {avg:.2f}")
            wandb.log({"train/avg_return": avg})
            train_reward_avg = 0.0

    # Save model
    torch.save(agent.policy.state_dict(), "model.mdl")
    wandb.save("model.mdl")

    wandb.finish()


if __name__ == '__main__':
    main()