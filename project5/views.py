import io
import os
import base64
import joblib
import numpy as np
import torch

from django.conf import settings
from django.shortcuts import render, redirect

# Matplotlib (headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# === RL imports from your package ===
from .rl_agent.reinforce import (
    PolicyNetwork,
    REINFORCE,
    compute_loss,
    compute_loss_with_penalty,
    state_to_tensor,
    get_reward_from_project_spec,
)
from .rl_agent.mouse import (
    MOUSE,
    ACTIONS,
    print_grid_with_cheese_types,
    initialize_grid_with_cheese_types,
    move,
    GRID_SIZE,
    WALL,
    CHEESE,
    ORGANIC_CHEESE,
    TRAP,
    EMPTY,
)
from .rl_agent.reward_model import RewardModel, train_reward_model


# ========== Globals ==========
feedback_data = {}
CURRENT_POLICY_NET = None  # trained baseline after Task 1 will be stored here


# ========== Utilities ==========
def fig_to_base64(fig) -> str:
    """Encode a Matplotlib fig as base64 PNG and close it."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


from django.conf import settings

from django.templatetags.static import static

from django.templatetags.static import static

from django.conf import settings

from django.conf import settings
import numpy as np

def grid_to_html(grid_np: np.ndarray) -> str:
    """
    Responsive, image-first grid that expands to the width of its parent
    (the trajectory card). Cells stay square via aspect-ratio and scale
    automatically. Uses static icons if present; falls back to emojis.
    """
    rows, cols = grid_np.shape

    icon_base = (settings.STATIC_URL.rstrip("/") + "/project5/icons").rstrip("/")

    # Map cell values -> (filename, emoji fallback, accent color)
    from .rl_agent.mouse import (
        MOUSE, CHEESE, ORGANIC_CHEESE, TRAP, WALL, EMPTY
    )

    def cell_spec(val):
        if val == MOUSE:
            return ("mouse.png", "🐭", "#2563eb")
        if val == CHEESE:
            return ("cheese.png", "🧀", "#f59e0b")
        if val == ORGANIC_CHEESE:
            return ("organic.png", "🥦", "#10b981")
        if val == TRAP:
            return ("trap.png", "☠️", "#ef4444")
        if val == WALL:
            return ("wall.png", "⬛", "#9ca3af")
        return ("empty.svg", "", "#e5e7eb")  # EMPTY

    # CONTAINER: fills parent width, keeps some padding, and a subtle card look
    container_style = (
        "width:100%;"
        "display:grid;"
        f"grid-template-columns:repeat({cols}, 1fr);"
        "gap:10px;"
        "padding:14px;"
        "background:#f8fafc;"
        "border:1px solid #e5e7eb;"
        "border-radius:14px;"
        "box-shadow:0 1px 3px rgba(0,0,0,0.06) inset;"
        "box-sizing:border-box;"
    )

    # TILE: squares that stretch with the card
    tile_style = (
        "position:relative;"
        "aspect-ratio:1/1;"               # <— keeps perfect square cells
        "border-radius:10px;"
        "background:linear-gradient(180deg,#fff,#f3f4f6);"
        "box-shadow:inset 0 0 0 1px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.05);"
        "overflow:hidden;"
        "display:flex;align-items:center;justify-content:center;"
    )

    # ICON / EMOJI scales with cell size
    img_style   = "width:66%;height:66%;object-fit:contain;user-select:none;pointer-events:none;"
    emoji_style = "font-size:clamp(16px,5.2cqw,36px);line-height:1;"  # cqw follows container width

    # small dot for empty cells
    dot_html = '<span style="width:10%;height:10%;border-radius:50%;background:#d1d5db;display:block;"></span>'

    parts = [f'<div class="grid-responsive" style="{container_style}">']

    for r in range(rows):
        for c in range(cols):
            fname, emoji, accent = cell_spec(grid_np[r, c])

            # subtle highlight ring for non-empty tiles
            ring = f"outline:2px solid {accent}2A;" if (fname != "empty.svg" or emoji) else ""

            parts.append(f'<div style="{tile_style}{ring}">')

            if fname and fname != "empty.svg":
                src = f"{icon_base}/{fname}"
                # graceful fallback to emoji if file missing
                parts.append(
                    f'<img src="{src}" alt="" style="{img_style}" '
                    "onerror=\"this.replaceWith(Object.assign(document.createElement('span'),"
                    f"{{textContent:'{emoji}', style:'{emoji_style}'}}))\">"
                )
            elif emoji:
                parts.append(f'<span style="{emoji_style}">{emoji}</span>')
            else:
                parts.append(dot_html)

            parts.append("</div>")  # /tile

    parts.append("</div>")  # /grid
    return "".join(parts)
# ========== Views ==========
def index(request):
    return render(request, "index.html")


def simulate_trajectory(policy_net: PolicyNetwork, max_steps: int):
    """
    Roll a single trajectory with the current policy.
    Returns a list of step dicts, each including:
      - state tensor
      - action_name / action_index
      - immediate reward
      - grid_output (ASCII for debugging)
      - grid_html (pretty HTML tiles; INLINE styles)
      - step number (1-based)
    """
    grid, mouse_pos, _, _ = initialize_grid_with_cheese_types()
    trajectory_data = []

    for step in range(max_steps):
        state_tensor = state_to_tensor(grid, mouse_pos)

        # sample an action from the policy
        action_probs = policy_net(state_tensor)
        action_index = torch.distributions.Categorical(action_probs).sample()
        action_name = ACTIONS[action_index.item()]

        # compute immediate reward for the *attempted* move
        old_mouse_pos = mouse_pos
        delta = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[action_name]
        new_pos_potential = (old_mouse_pos[0] + delta[0], old_mouse_pos[1] + delta[1])

        if 0 <= new_pos_potential[0] < GRID_SIZE and 0 <= new_pos_potential[1] < GRID_SIZE:
            cell_content_before_move = grid[new_pos_potential]
            reward = get_reward_from_project_spec(
                cell_content_before_move if cell_content_before_move != WALL else WALL
            )
        else:
            reward = get_reward_from_project_spec(WALL)

        # execute the move in the environment
        grid = move(action_name, grid)
        new_mouse_pos = tuple(np.argwhere(grid == MOUSE)[0])

        # if mouse didn't move (bumped wall), keep wall penalty
        if new_mouse_pos == old_mouse_pos:
            reward = get_reward_from_project_spec(WALL)

        # capture ASCII grid (for debugging/logs)
        out = io.StringIO()
        print_grid_with_cheese_types(grid, file=out)
        grid_output = out.getvalue()

        # ALWAYS build robust pretty HTML (inline styles; no CSS dependency)
        grid_html = grid_to_html(grid)

        trajectory_data.append(
            {
                "state": state_tensor,
                "action_name": action_name,
                "action_index": action_index,
                "reward": reward,
                "grid_output": grid_output,  # debugging
                "grid_html": grid_html,      # UI uses this
                "step": step + 1,
            }
        )

        # optional terminate upon landing on terminal cells
        if grid[new_mouse_pos] in [CHEESE, ORGANIC_CHEESE, TRAP]:
            break

        mouse_pos = new_mouse_pos

    return trajectory_data


def evaluate_policy(policy_net: PolicyNetwork, num_episode: int = 50, max_steps: int = 20):
    """
    Roll out several episodes and compute total return statistics.
    Returns (avg, std, all_returns_list).
    """
    returns = []
    for _ in range(num_episode):
        traj = simulate_trajectory(policy_net, max_steps)
        total = sum(step["reward"] for step in traj)
        returns.append(total)
    returns = np.array(returns, dtype=float)
    return float(np.mean(returns)), float(np.std(returns)), returns.tolist()


def get_feedback_from_fake_user(traj1, traj2):
    """(Optional helper; kept for completeness)"""
    score1 = sum(1 for step in traj1 if step["grid_output"].find("C") != -1)
    score2 = sum(1 for step in traj2 if step["grid_output"].find("C") != -1)
    if score1 < score2:
        return 1
    if score2 < score1:
        return 2
    return np.random.choice([1, 2])


def train_agent_view(request):
    """
    Task 1: Train a baseline policy with REINFORCE.
    Also produce a learning curve and a histogram of post-training returns.
    """
    policy_net = PolicyNetwork()
    agent = REINFORCE(policy_net)

    num_episodes = 10
    num_trajectories = 2
    max_steps = 20

    # For the learning curve we collect avg reward per episode using quick rollouts
    learning_curve = []

    for _ in range(num_episodes):
        total_loss = 0.0
        for _ in range(num_trajectories):
            trajectory_data = simulate_trajectory(agent.policy_net, max_steps)
            states = [s["state"] for s in trajectory_data]
            actions_taken = [s["action_index"] for s in trajectory_data]
            rewards = [s["reward"] for s in trajectory_data]

            loss = compute_loss(states, actions_taken, rewards, agent.policy_net, agent.gamma)
            total_loss += loss

        agent.optimizer.zero_grad()
        total_loss.backward()
        agent.optimizer.step()

        # quick estimate of average reward after this episode
        avg_ep, _, _ = evaluate_policy(agent.policy_net, num_episode=10, max_steps=20)
        learning_curve.append(avg_ep)

    # Build trajectories to display
    trajectories_to_show = [simulate_trajectory(agent.policy_net, max_steps=10) for _ in range(2)]

    # Evaluate final policy to make a histogram
    avg_final, std_final, returns = evaluate_policy(agent.policy_net, num_episode=32, max_steps=20)

    # --- Figures ---
    # 1) Learning curve
    fig1 = plt.figure(figsize=(6, 4))
    xs = np.arange(1, len(learning_curve) + 1)
    plt.plot(xs, learning_curve, marker="o")
    plt.title("REINFORCE Learning Curve")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward (avg over rollouts)")
    learning_curve_img = fig_to_base64(fig1)

    # 2) Return histogram after training
    fig2 = plt.figure(figsize=(6, 4))
    plt.hist(returns, bins=12, edgecolor="black")
    plt.title("Reward Distribution (Trained Policy)")
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    reward_hist_img = fig_to_base64(fig2)

    # Store baseline so Task 2 can use as reference if user proceeds later
    global CURRENT_POLICY_NET
    CURRENT_POLICY_NET = agent.policy_net

    context = {
        "message": "REINFORCE algorithm training complete!",
        "trajectories": trajectories_to_show,
        "learning_curve_img": learning_curve_img,
        "reward_hist_img": reward_hist_img,
        "avg_final": avg_final,
        "std_final": std_final,
    }
    return render(request, "training_complete.html", context)


def start_feedback_study_view(request):
    """Show two trajectories and ask the user which is better."""
    global CURRENT_POLICY_NET
    policy_net = CURRENT_POLICY_NET if CURRENT_POLICY_NET is not None else PolicyNetwork()

    global feedback_data
    feedback_data = {
        "traj1": simulate_trajectory(policy_net, max_steps=5),
        "traj2": simulate_trajectory(policy_net, max_steps=5),
    }

    context = {
        "trajectory1": feedback_data["traj1"],
        "trajectory2": feedback_data["traj2"],
        "total_reward1": sum(step["reward"] for step in feedback_data["traj1"]),
        "total_reward2": sum(step["reward"] for step in feedback_data["traj2"]),
    }
    return render(request, "feedback_study.html", context)


def collect_feedback_view(request):
    """
    Receive user's preference and train a small reward model.
    Persist the learned model so retraining can find it.
    """
    if request.method == "POST":
        preference = int(request.POST.get("preference"))

        traj1_log = feedback_data.get("traj1", [])
        traj2_log = feedback_data.get("traj2", [])

        feedbacks = [
            {
                "trajectory1_states": [step["state"] for step in traj1_log],
                "trajectory2_states": [step["state"] for step in traj2_log],
                "preference": preference,
            }
        ]

        reward_model = RewardModel()
        train_reward_model(reward_model, feedbacks)

        # Save the reward model to disk for RLHF retraining page
        model_dir = os.path.join(settings.BASE_DIR, "model_data")
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(reward_model, os.path.join(model_dir, "reward_model.joblib"))

        context = {"message": "Reward model trained using human feedback!"}
        return render(request, "reward_model_complete.html", context)

    return redirect("project5:start_feedback_study")


def retrain_with_feedback_view(request):
    """
    Task 2: RLHF-style fine-tuning.
    - Use CURRENT_POLICY_NET as the KL reference (baseline) if available
    - Load the saved reward model (from collect_feedback_view)
    - Train a new policy with reward model + KL penalty to the baseline
    - Show trajectories and comparison graphs
    """
    # 1) Baseline (KL anchor)
    global CURRENT_POLICY_NET
    original_policy_net = CURRENT_POLICY_NET if CURRENT_POLICY_NET is not None else PolicyNetwork()

    # Example trajectories from baseline
    original_trajectories = [simulate_trajectory(original_policy_net, max_steps=5) for _ in range(2)]

    # 2) Load learned reward model
    model_dir = os.path.join(settings.BASE_DIR, "model_data")
    os.makedirs(model_dir, exist_ok=True)
    reward_model_path = os.path.join(model_dir, "reward_model.joblib")
    if os.path.exists(reward_model_path):
        reward_model = joblib.load(reward_model_path)
    else:
        reward_model = RewardModel()  # fallback (no preferences collected)

    # 3) Fine-tune a new policy with KL penalty
    new_policy_net = PolicyNetwork()
    retrain_agent = REINFORCE(new_policy_net)

    num_episodes = 10
    num_trajectories = 2
    max_steps = 20

    for _ in range(num_episodes):
        total_loss = 0.0
        for _ in range(num_trajectories):
            traj = simulate_trajectory(new_policy_net, max_steps=max_steps)
            states = [s["state"] for s in traj]
            actions_taken = [s["action_index"] for s in traj]

            loss = compute_loss_with_penalty(
                states,
                actions_taken,
                new_policy_net,
                original_policy_net,
                retrain_agent.gamma,
                reward_model,
            )
            total_loss += loss

        retrain_agent.optimizer.zero_grad()
        total_loss.backward()
        retrain_agent.optimizer.step()

    # Display trajectories from new policy
    retrained_trajectories = [simulate_trajectory(new_policy_net, max_steps=5) for _ in range(2)]

    # Evaluate both policies for comparison figures
    orig_avg, orig_std, orig_returns = evaluate_policy(original_policy_net, num_episode=32, max_steps=20)
    new_avg, new_std, new_returns = evaluate_policy(new_policy_net, num_episode=32, max_steps=20)

    # Promote the retrained policy to current (optional)
    CURRENT_POLICY_NET = new_policy_net

    # --- Figures ---
    # 1) Bar with error bars (mean ± std)
    fig_bar = plt.figure(figsize=(6, 4))
    x = np.arange(2)
    means = [orig_avg, new_avg]
    stds = [orig_std, new_std]
    labels = ["Baseline", "RLHF"]
    plt.bar(x, means, yerr=stds, capsize=6)
    plt.xticks(x, labels)
    plt.ylabel("Total Reward")
    plt.title("Baseline vs RLHF (Mean ± Std)")
    comparison_bar_img = fig_to_base64(fig_bar)

    # 2) Overlaid histograms
    fig_hist = plt.figure(figsize=(6, 4))
    plt.hist(orig_returns, bins=12, alpha=0.6, label="Baseline", edgecolor="black")
    plt.hist(new_returns, bins=12, alpha=0.6, label="RLHF", edgecolor="black")
    plt.legend()
    plt.xlabel("Total Reward")
    plt.ylabel("Frequency")
    plt.title("Return Distributions")
    compare_hist_img = fig_to_base64(fig_hist)

    # Totals for the first two display trajectories (optional display values)
    original_total_reward1 = sum(step["reward"] for step in original_trajectories[0])
    original_total_reward2 = sum(step["reward"] for step in original_trajectories[1])
    retrained_total_reward1 = sum(step["reward"] for step in retrained_trajectories[0])
    retrained_total_reward2 = sum(step["reward"] for step in retrained_trajectories[1])

    context = {
        "message": "Policy retrained with human feedback!",
        "original_avg_reward": orig_avg,
        "original_std": orig_std,
        "retrained_avg_reward": new_avg,
        "retrained_std": new_std,
        "original_trajectories": original_trajectories,
        "retrained_trajectories": retrained_trajectories,
        "original_total_reward1": original_total_reward1,
        "original_total_reward2": original_total_reward2,
        "retrained_total_reward1": retrained_total_reward1,
        "retrained_total_reward2": retrained_total_reward2,
        "comparison_bar_img": comparison_bar_img,
        "compare_hist_img": compare_hist_img,
    }
    return render(request, "retrained_policy_complete.html", context)
