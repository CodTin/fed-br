import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

# --- 绘图风格设置 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2
})


def find_latest_log_file(search_dir: Path, pattern: str = "client_metrics_*.jsonl") -> Path:
    """
    在指定目录中查找符合模式的最新文件。
    由于时间戳格式为 YYYYMMDD_HHMMSS，按文件名排序即可找到最新的。
    """
    if not search_dir.exists():
        print(f"Warning: Directory '{search_dir}' does not exist.")
        return None

    # 查找所有匹配的文件
    # 使用 rglob 可以递归查找子目录，或者使用 glob 仅查找当前目录
    # 这里为了稳健性，先在当前目录找，找不到再去子目录找
    files = list(search_dir.glob(pattern))
    if not files:
        files = list(search_dir.rglob(pattern))

    if not files:
        return None

    # 按文件名排序（因为时间戳在文件名中且格式固定，字典序即时间序）
    # 也可以使用 key=lambda p: p.stat().st_mtime 按修改时间排序
    latest_file = sorted(files)[-1]
    return latest_file


def load_records(path: Path):
    """加载 JSONL 日志文件"""
    records = []
    print(f"Reading log file: {path}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    records.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        return []
    return records


def aggregate_rounds(records):
    """
    聚合每一轮的全局指标。
    """
    rounds_data = defaultdict(lambda: {
        "clients": {},
        "global_metrics": defaultdict(float)
    })

    # -------------------------------------------------------------
    # [重要] 权重参数设置
    # 请根据您 common/const.py 或 system_model.py 中的实际值进行调整
    # -------------------------------------------------------------
    ALPHA = 1.0  # Energy weight
    BETA = 1.0  # Privacy weight
    GAMMA = 100.0  # Error weight
    # Global Error 常数 A (Eq 26), 这是一个问题相关的常数
    CONST_A = 50.0

    for r in records:
        round_num = r.get("round")
        cid = r.get("client_id")
        metrics = r.get("metrics", {})

        # 仅处理包含有效指标的记录
        if round_num is not None and cid is not None:
            rounds_data[round_num]["clients"][cid] = metrics

    sorted_rounds = sorted(rounds_data.keys())
    aggregated_history = []

    for r in sorted_rounds:
        clients = rounds_data[r]["clients"]
        if not clients:
            continue

        # 1. 聚合 Total Energy
        total_energy = sum(c.get("total_energy", 0) for c in clients.values())

        # 2. 聚合 Privacy Cost
        total_privacy = sum(c.get("privacy_cost", 0) for c in clients.values())

        # 3. 计算 Global Error (Surrogate J_err)
        # J_err = A / (Sum e_i) + Sum (Impact_i)
        total_epochs = sum(c.get("local_epochs", 1) for c in clients.values())
        sum_impacts = sum(c.get("local_impact_factor", 0) for c in clients.values())

        # 防止除以零
        safe_total_epochs = total_epochs if total_epochs > 0 else 1
        j_err = (CONST_A / safe_total_epochs) + sum_impacts

        # 4. 计算势函数 Potential Function Phi (Theorem 1)
        phi = (ALPHA * total_energy) + (BETA * total_privacy) + (GAMMA * j_err)

        aggregated_history.append({
            "round": r,
            "phi": phi,
            "j_err": j_err,
            "total_energy": total_energy,
            "total_privacy": total_privacy,
            "client_strategies": {
                cid: {
                    "e": c.get("local_epochs"),
                    "sigma": c.get("noise_multiplier")
                }
                for cid, c in clients.items()
            }
        })

    return aggregated_history


def plot_paper_figures(history, output_dir: Path):
    if not history:
        print("No data to plot.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    rounds = [h["round"] for h in history]

    # --- Figure 1: Potential Function Descent ---
    phis = [h["phi"] for h in history]

    plt.figure(figsize=(8, 6))
    plt.plot(rounds, phis, 'b-o', linewidth=2, label=r'Potential Function $\Phi(a)$')
    plt.xlabel('Communication Round')
    plt.ylabel(r'Potential Value $\Phi$')
    plt.title('Convergence of Potential Function (Fed-BR)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_dir / "fig_potential_convergence.png", dpi=300)
    plt.close()
    print(f"Generated: {output_dir / 'fig_potential_convergence.png'}")

    # --- Figure 2: Cost Components ---
    energies = [h["total_energy"] for h in history]
    privacies = [h["total_privacy"] for h in history]
    errors = [h["j_err"] * 100 for h in history]  # Scale for visibility if needed

    fig, ax1 = plt.subplots(figsize=(8, 6))

    ax1.set_xlabel('Communication Round')
    ax1.set_ylabel('Cost Components (Energy & Privacy)', color='tab:blue')
    l1, = ax1.plot(rounds, energies, color='tab:blue', linestyle='--', label='Total Energy')
    l2, = ax1.plot(rounds, privacies, color='tab:green', linestyle='-.', label='Total Privacy Cost')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel(r'Global Error Cost ($J^{err} \times 100$)', color='tab:red')
    l3, = ax2.plot(rounds, errors, color='tab:red', marker='x', label='Global Error')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    lines = [l1, l2, l3]
    labels = [l.get_label() for l in lines]
    plt.legend(lines, labels, loc='upper right')
    plt.title('Evolution of System Cost Components')
    plt.tight_layout()
    plt.savefig(output_dir / "fig_cost_components.png", dpi=300)
    plt.close()
    print(f"Generated: {output_dir / 'fig_cost_components.png'}")

    # --- Figure 3: Client Strategy Evolution ---
    # 自动选择数据最全的 3 个客户端
    # 统计每个客户端出现的次数
    client_counts = defaultdict(int)
    for h in history:
        for cid in h["client_strategies"]:
            client_counts[cid] += 1
    # 选出现次数最多的前3个
    target_clients = sorted(client_counts, key=client_counts.get, reverse=True)[:3]

    if target_clients:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10), sharex=True)
        markers = ['o', 's', '^', 'd', 'v']
        colors = ['r', 'g', 'b', 'c', 'm']

        for idx, cid in enumerate(target_clients):
            c_epochs = []
            c_noises = []
            valid_rounds = []

            for h in history:
                if cid in h["client_strategies"]:
                    st = h["client_strategies"][cid]
                    if st["e"] is not None:
                        valid_rounds.append(h["round"])
                        c_epochs.append(st["e"])
                        c_noises.append(st["sigma"])

            m = markers[idx % len(markers)]
            c = colors[idx % len(colors)]

            if valid_rounds:
                ax1.plot(valid_rounds, c_epochs, marker=m, color=c,
                         label=f'Client {cid}', linestyle='-')
                ax2.plot(valid_rounds, c_noises, marker=m, color=c,
                         label=f'Client {cid}', linestyle='-')

        ax1.set_ylabel(r'Local Epochs ($e_i$)')
        ax1.set_title('Evolution of Local Epochs')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        ax2.set_ylabel(r'Noise Multiplier ($\sigma_i$)')
        ax2.set_xlabel('Communication Round')
        ax2.set_title('Evolution of Privacy Noise')
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(output_dir / "fig_strategy_equilibrium.png", dpi=300)
        plt.close()
        print(f"Generated: {output_dir / 'fig_strategy_equilibrium.png'}")


def main():
    parser = argparse.ArgumentParser(description="Generate Paper Figures for Fed-BR (Auto-detect latest log)")

    # 默认去 final_model 文件夹找，如果找不到会尝试当前文件夹
    parser.add_argument("--search-dir", type=Path, default=Path("../outputs"),
                        help="Directory to search for log files (default: outputs)")
    parser.add_argument("--input", type=Path, default=None,
                        help="Path to specific log file (overrides auto-detection)")
    parser.add_argument("--output-dir", type=Path, default=Path("paper_plots"),
                        help="Directory for output plots")

    args = parser.parse_args()

    input_path = args.input

    # 如果没有指定具体文件，则自动查找最新文件
    if input_path is None:
        print(f"Searching for latest log in '{args.search_dir}'...")
        input_path = find_latest_log_file(args.search_dir)

        # 如果在 final_model 没找到，尝试在当前目录找
        if input_path is None and args.search_dir != Path("."):
            print(f"No logs in '{args.search_dir}', searching in current directory...")
            input_path = find_latest_log_file(Path("."))

        if input_path is None:
            print("Error: No 'client_metrics_*.jsonl' files found.")
            return

    if not input_path.exists():
        print(f"Error: File {input_path} does not exist.")
        return

    records = load_records(input_path)

    if not records:
        print("No valid records found in the log file.")
        return

    print("Aggregating metrics...")
    history = aggregate_rounds(records)

    print(f"Plotting figures to '{args.output_dir}'...")
    plot_paper_figures(history, args.output_dir)
    print("All plots generated successfully.")


if __name__ == "__main__":
    main()