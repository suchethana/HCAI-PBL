import torch
import numpy as np
from .reinforce import PolicyNetwork, state_to_tensor
from .mouse import initialize_grid_with_cheese_types, move, ACTIONS, CHEESE, ORGANIC_CHEESE, MOUSE, TRAP, WALL, EMPTY, \
    GRID_SIZE



def simulate_full_trajectory(policy_net, max_steps):
    grid, mouse_pos, _, _ = initialize_grid_with_cheese_types()
    trajectory = []

    for _ in range(max_steps):
        state_tensor = state_to_tensor(grid, mouse_pos)

        with torch.no_grad():
            action_probs = policy_net(state_tensor)
        action_index = torch.distributions.Categorical(action_probs).sample()
        action = ACTIONS[action_index.item()]

        old_mouse_pos = mouse_pos
        grid_copy = grid.copy()
        grid = move(action, grid)
        new_mouse_pos = tuple(np.argwhere(grid == MOUSE)[0])

        trajectory.append({
            'state': state_to_tensor(grid_copy, old_mouse_pos),
            'action': action_index,
            'grid': grid.copy()
        })


        if grid[new_mouse_pos] in [CHEESE, ORGANIC_CHEESE, TRAP]:
            break

    return trajectory


# A fake user that prefers the trajectory with the organic cheese first, or the shortest number of steps if no organic cheese.
def get_feedback_from_fake_user(trajectory1, trajectory2):
    has_organic_cheese_1 = any(
        step['grid'][tuple(np.argwhere(step['grid'] == MOUSE)[0])] == ORGANIC_CHEESE for step in trajectory1)
    has_organic_cheese_2 = any(
        step['grid'][tuple(np.argwhere(step['grid'] == MOUSE)[0])] == ORGANIC_CHEESE for step in trajectory2)


    if has_organic_cheese_1 and not has_organic_cheese_2:
        return 2
    elif not has_organic_cheese_1 and has_organic_cheese_2:
        return 1
    else:

        if len(trajectory1) < len(trajectory2):
            return 1
        elif len(trajectory2) < len(trajectory1):
            return 2
        else:
            return np.random.choice([1, 2])