# visualize.py

import matplotlib.pyplot as plt

def visualize_results(env, save_path, reward=None, epsilon=None, policy_loss=None, value_loss=None, folder_name=None):
    plt.figure(figsize=(10, 10))
    missions = env.missions.cpu().numpy()
    plt.scatter(missions[:, 0], missions[:, 1], c='blue', marker='o', label='Mission')
    plt.scatter(missions[0, 0], missions[0, 1], c='green', marker='s', s=100, label='Start/End Point')
    
    colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
    for i, path in enumerate(env.paths):
        path_coords = missions[path]
        color = colors[i % len(colors)]
        plt.plot(path_coords[:, 0], path_coords[:, 1], marker='x', color=color, label=f'UAV {i} Path (Velocity: {env.speeds[i].item():.2f})')
    
    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'UAV MTSP - {folder_name}')
    
    # 텍스트 추가
    text = ""
    if reward is not None:
        text += f"Reward: {reward:.2f}\n"
    if epsilon is not None:
        text += f"Epsilon: {epsilon:.4f}\n"
    if policy_loss is not None:
        text += f"Policy Loss: {policy_loss:.4f}\n"
    if value_loss is not None:
        text += f"Value Loss: {value_loss:.4f}\n"
    
    if text:
        plt.text(0.05, 0.95, text, transform=plt.gca().transAxes, fontsize=12,
                 verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
