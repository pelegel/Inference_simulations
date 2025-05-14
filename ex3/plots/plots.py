import matplotlib.pyplot as plt

def plot_requests_status(running, waiting, total_context_length):
    x = total_context_length
    plt.plot(x, running, label="Running", color="blue")
    plt.plot(x, waiting, label="Waiting", color="orange")
    plt.xlabel("Context Length")
    plt.ylabel("# Requests")
    plt.title("Runing and Waiting Requests")
    plt.legend()
    plt.grid(True)
    plt.savefig("Requests_Status_vs_Context_Length.png", dpi=300, bbox_inches="tight")
    plt.clf()  # Clear the current figure



def plot_queue_times(queue_times):
    x = [i for i in range(len(queue_times))]
    plt.bar(x, queue_times)
    plt.xlabel("Request")
    plt.ylabel("Queue time")
    plt.title("Requests Queue Time")
    plt.grid(True)
    plt.savefig("Requests_Queue_Time.png", dpi=300, bbox_inches="tight")
    plt.clf()  # Clear the current figure


    
def plot_generation_speed(generation_speeds, total_context_length, queue_times):
    x = total_context_length
    fig, ax1 = plt.subplots()

    # Plot generation speeds on primary y-axis
    ax1.plot(x, generation_speeds, label="Generation Speed (T/s)", color="blue")
    ax1.set_xlabel("Context Length")
    ax1.set_ylabel("Generation Speed (T/s)", color="blue")
    ax1.tick_params(axis='y', labelcolor="blue")

    # Create a second y-axis for queue times
    ax2 = ax1.twinx()
    ax2.plot(x, queue_times, label="Queue Time", color="orange")
    ax2.set_ylabel("Queue Time (s)", color="orange")
    ax2.tick_params(axis='y', labelcolor="orange")

    # Title and grid
    plt.title("Generation Speed and Queue Time vs Context Length")
    ax1.grid(True)

    # Save and clear
    plt.savefig("Generation_Speed_vs_Context_Length.png", dpi=300, bbox_inches="tight")
    plt.clf()




if __name__ == "__main__":


    with open("running.txt", "r") as f:
        running = [float(line.strip()) for line in f if line.strip()]

    with open("waiting.txt", "r") as f:
        waiting = [float(line.strip()) for line in f if line.strip()]

    with open("generation_speeds.txt", "r") as f:
        generation_speeds = [float(line.strip()) for line in f if line.strip()]

    with open("durations.txt", "r") as f:
        durations = [float(line.strip()) for line in f if line.strip()]

    with open("queue_times.txt", "r") as f:
        queue_times = [float(line.strip()) for line in f if line.strip()]

    with open("total_context_length.txt", "r") as f:
        total_context_length = [float(line.strip()) for line in f if line.strip()]

    plot_requests_status(running, waiting, total_context_length)
    plot_queue_times(queue_times)
    plot_generation_speed(generation_speeds, total_context_length, queue_times)