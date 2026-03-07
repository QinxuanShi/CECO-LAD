import psutil
import time
import json
import os
import argparse

def get_current_processes():
    processes = {}
    pids = psutil.pids()
    for pid in pids:
        if psutil.pid_exists(int(pid)):
            p = psutil.Process(pid)
            if p != None:
                processes[p.pid] = p.name()
    return processes
 
def get_new_processes(old_processes):
    # get new processes (pid, name)
    current_processes = get_current_processes()
    new_processes = {pid: name for pid, name in current_processes.items() if pid not in old_processes}
 
    # print number of new processes, and the info of each new process
    print(f"Number of new processes: {len(new_processes)}")

    for pid, name in new_processes.items():
        print(f"New process detected: PID={pid}, Name={name}")
    
    print("-" * 50)
 
    return current_processes, new_processes

def monitor_process(file_name, data_source, iteration):
    monitor_list = {}
    
    # Path to the directory and file
    folder_path = './consumption_monitor/' + data_source
    process_store = os.path.join(folder_path, 'process_'+ file_name + '.ndjson')
    system_store = os.path.join(folder_path, 'system_'+ file_name + '.ndjson')
    
    print(process_store)
    print(system_store)

    old_processes = get_current_processes()

    try:
        while True:
            # start time in one loop
            start = time.time()
            
            update_processes, new_processes = get_new_processes(old_processes)
            # update the old processes
            old_processes = update_processes

            matches = ["python", "pt_main_thread", "executor_runner", "py", "python.exe"]

            # search all the new processes name can be matched, if matched, add to moniter list keep monitoring
            for pid, name in new_processes.items():
                if any(x in name.lower() for x in matches):
                    print(f"start monitoring new process: PID={pid}, Name={name}")
                    monitor_list[pid] = name
            
            # monitor processes
            current_time = time.time()

            record_p = {}
            record_p[current_time] = {}
            record_s = {}
            record_s.setdefault(current_time, {})

            # record the process info
            for pid in list(monitor_list.keys()):
                # check if the process is still running
                if not psutil.pid_exists(int(pid)):
                    print(f"process {pid}, name: {monitor_list[pid]} had stopped.")
                    monitor_list.pop(pid)
                    continue
                else:
                    # if exists, check if pid be reused by another program
                    process = psutil.Process(pid)
                    # add all process info into one json which key is timestamp
                    record_p[current_time][pid] = {"Name": process.name(), "CPU": process.cpu_percent(interval=1),
                                    "Memory": process.memory_percent(), "Mem_use": process.memory_info().rss, "io_counters": process.io_counters()}
        
            # record system info
            cpu = psutil.cpu_percent(interval=1)
            cpu_logi = psutil.cpu_percent(interval=1,  percpu=True)
            memory = psutil.virtual_memory().percent
            temperature = psutil.sensors_temperatures()
            record_s[current_time] = {"CPU_total": cpu, "Memory": memory, "logi_cpu": cpu_logi, "temperature": temperature}

            print(record_p)
            print(record_s)

            # Write the dictionary to a JSON file
            with open(process_store, 'a') as json_file:
                json_file.write(json.dumps(record_p) + '\n')
            
            with open(system_store, 'a') as json_file:
                json_file.write(json.dumps(record_s) + '\n')

            # loop every 3 seconds, if the loop takes more than 3 seconds, no wait and next loop
            duration = time.time() - start
            print(f"duration: {duration}")
            sleep_time = 2 - duration
            if sleep_time > 0:
                time.sleep(sleep_time)
            print("=" * 60)
    except psutil.NoSuchProcess:
        print(f"Process has terminated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--file_name', type=str, default='executorch_os')
    parser.add_argument('--data_source', type=str, default='os')
    parser.add_argument('--iteration', type=str, default="1")

    config = parser.parse_args()

    monitor_process(config.file_name, config.data_source, config.iteration)
