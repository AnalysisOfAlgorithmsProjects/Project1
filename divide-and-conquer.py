import random, time
import matplotlib.pyplot as plt

from statistics import mean



def recursive_load_balancer(tasks, servers, depth=0):

    indent = "  " * depth 

    print(f"{indent}Level {depth}: Tasks = {tasks}, Servers = {servers}")

    # Base Case: only one server left
    if len(servers) == 1:
        print(f"{indent}→ Assign all tasks {tasks} to {servers[0]}")
        return {servers[0]: tasks}
    
    # Divide step
    mid_t = len(tasks) // 2
    mid_s = len(servers) // 2

    left_tasks = tasks[:mid_t]
    right_tasks = tasks[mid_t:]
    left_servers = servers[:mid_s]
    right_servers = servers[mid_s:]

    print(f"{indent}Dividing into:")
    print(f"{indent}  Left:  Tasks {left_tasks}, Servers {left_servers}")
    print(f"{indent}  Right: Tasks {right_tasks}, Servers {right_servers}")

    # Recursive conquer
    left_mapping = recursive_load_balancer(left_tasks, left_servers, depth + 1)
    right_mapping = recursive_load_balancer(right_tasks, right_servers, depth + 1)

    # Combine step
    left_load = sum(sum(v) for v in left_mapping.values())
    right_load = sum(sum(v) for v in right_mapping.values())
    threshold = 0.1 * (left_load + right_load)

    print(f"{indent}Combining results at Level {depth}:")
    print(f"{indent}  Left total = {left_load}, Right total = {right_load}")

    if abs(left_load - right_load) > threshold:
        print(f"{indent}  ⚖️  Rebalancing: difference {abs(left_load - right_load)} > threshold {threshold:.2f}")
        
        # Identify heavy vs light sides
        if left_load > right_load:
            heavy_mapping, light_mapping = left_mapping, right_mapping
        else:
            heavy_mapping, light_mapping = right_mapping, left_mapping

        # Record original difference
        original_diff = abs(left_load - right_load)

        # Find the heaviest task on the heavy side
        heavy_server = max(heavy_mapping, key=lambda k: sum(heavy_mapping[k]))
        task_to_move = max(heavy_mapping[heavy_server])

        # Find the lightest server on the light side
        light_server = min(light_mapping, key=lambda k: sum(light_mapping[k]))

        print(f"{indent}  Attempting to move task {task_to_move} from {heavy_server} → {light_server}")

        # Perform tentative move
        heavy_mapping[heavy_server].remove(task_to_move)
        light_mapping[light_server].append(task_to_move)

        # Recalculate new difference
        new_left = sum(sum(v) for v in left_mapping.values())
        new_right = sum(sum(v) for v in right_mapping.values())
        new_diff = abs(new_left - new_right)

        # Decide whether to keep or revert
        if new_diff < original_diff:
            print(f"{indent}  ✅ Improved balance: {original_diff:.2f} → {new_diff:.2f}. Move kept.")
        else:
            print(f"{indent}  ⚠️ Move worsened balance: {original_diff:.2f} → {new_diff:.2f}. Reverting.")
            # Revert move
            light_mapping[light_server].remove(task_to_move)
            heavy_mapping[heavy_server].append(task_to_move)
    else:
        print(f"{indent}  ✅ Balanced within threshold.")


    merged = {**left_mapping, **right_mapping}
    print(f"{indent}→ Final mapping at Level {depth}: {merged}")
    return merged




if __name__ == "__main__":
    # tasks = [12, 9]   # 7 tasks
    # servers = ["S1", "S2", "S3", "S4", "S5"]

    # print("\n===== Divide and Conquer Load Balancing Debug Run =====")
    # result = recursive_load_balancer_debug(tasks, servers)
    # print("\n===== Final Assignment =====")
    # for s, t in result.items():
    #     print(f"{s}: total load = {sum(t):2d}, tasks = {t}")


    task_sizes = [100, 500, 1000, 5000, 10000]
    runtimes = []

    for n in task_sizes:
        tasks = [random.randint(1, 100) for _ in range(n)]
        servers = [f"S{i}" for i in range(16)]  # fixed number of servers
        
        start = time.time()
        recursive_load_balancer(tasks, servers)
        end = time.time()
        
        runtimes.append(end - start)



    plt.figure(figsize=(6,4))  # good aspect ratio for report
    plt.plot(task_sizes, runtimes, marker='o', linewidth=2, markersize=6)

    plt.title("Runtime vs Number of Tasks for Divide and Conquer Load Balancer", fontsize=11)
    plt.xlabel("Number of Tasks (n)", fontsize=10)
    plt.ylabel("Runtime (seconds)", fontsize=10)
    plt.grid(False)
    plt.box(False)
    plt.tight_layout()

    plt.savefig("runtime_plot.png", dpi=300, bbox_inches="tight")
    plt.show()
