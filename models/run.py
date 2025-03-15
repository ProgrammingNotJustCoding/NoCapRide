import os
import sys
import time
import logging
import subprocess
import signal
import argparse
import traceback
import shutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("run_log.txt"), logging.StreamHandler()],
)
logger = logging.getLogger("run")

# Global variables
processes = {}
MAX_RESTART_ATTEMPTS = 3
restart_counts = {}


def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        # Check Python packages
        required_packages = [
            "pandas",
            "numpy",
            "sklearn",
            "flask",
            "schedule",
            "holidays",
            "playwright",
            "requests",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            logger.error(f"Missing required packages: {', '.join(missing_packages)}")
            logger.error("Please install them using: pip install -r requirements.txt")
            return False

        # Check if playwright is installed
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                pass
        except Exception as e:
            logger.error(f"Error with Playwright: {e}")
            logger.error("Please run: playwright install")
            return False

        # Check if required directories exist
        for directory in ["data", "models"]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

        return True

    except Exception as e:
        logger.error(f"Error checking dependencies: {e}")
        logger.error(traceback.format_exc())
        return False


def start_process(name, command):
    """Start a subprocess and return the process object"""
    try:
        logger.info(f"Starting {name} process: {command}")

        # Create log file
        log_file = open(f"{name}_output.log", "w")

        # Start the process
        process = subprocess.Popen(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            shell=True,
            preexec_fn=os.setsid,  # Use process group for clean termination
        )

        logger.info(f"{name} process started with PID {process.pid}")

        # Store process info
        processes[name] = {
            "process": process,
            "log_file": log_file,
            "command": command,
            "start_time": time.time(),
        }

        # Initialize restart count
        if name not in restart_counts:
            restart_counts[name] = 0

        return process

    except Exception as e:
        logger.error(f"Error starting {name} process: {e}")
        logger.error(traceback.format_exc())
        return None


def stop_process(name):
    """Stop a running process"""
    if name in processes:
        process_info = processes[name]
        process = process_info["process"]

        try:
            logger.info(f"Stopping {name} process (PID {process.pid})")

            # Send SIGTERM to the process group
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)

            # Wait for process to terminate
            process.wait(timeout=5)

            # Close log file
            if process_info["log_file"]:
                process_info["log_file"].close()

            logger.info(f"{name} process stopped")

            # Remove from processes dict
            del processes[name]

            return True

        except subprocess.TimeoutExpired:
            logger.warning(f"{name} process did not terminate, sending SIGKILL")
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except Exception as e:
                logger.error(f"Error sending SIGKILL to {name} process: {e}")

            # Close log file
            if process_info["log_file"]:
                process_info["log_file"].close()

            # Remove from processes dict
            del processes[name]

            return True

        except Exception as e:
            logger.error(f"Error stopping {name} process: {e}")
            logger.error(traceback.format_exc())

            # Close log file if it exists
            if "log_file" in process_info and process_info["log_file"]:
                try:
                    process_info["log_file"].close()
                except:
                    pass

            # Remove from processes dict
            try:
                del processes[name]
            except:
                pass

            return False

    else:
        logger.warning(f"{name} process not found")
        return False


def check_process(name):
    """Check if a process is still running and restart if needed"""
    if name in processes:
        process_info = processes[name]
        process = process_info["process"]

        # Check if process is still running
        if process.poll() is not None:
            # Process has terminated
            exit_code = process.returncode
            logger.warning(f"{name} process has terminated with code {exit_code}")

            # Close log file
            if process_info["log_file"]:
                process_info["log_file"].close()

            # Remove from processes dict
            del processes[name]

            # Check restart count
            restart_counts[name] += 1
            if restart_counts[name] > MAX_RESTART_ATTEMPTS:
                logger.error(f"Too many restart attempts for {name}, giving up")
                return False

            # Print the last few lines of the log file
            log_file_path = f"{name}_output.log"
            if os.path.exists(log_file_path):
                logger.warning(f"Last lines of {name} log:")
                try:
                    with open(log_file_path, "r") as f:
                        lines = f.readlines()
                        for line in lines[-10:]:  # Print last 10 lines
                            logger.warning(f"  {line.strip()}")
                except Exception as e:
                    logger.error(f"Error reading log file: {e}")

            # Restart the process
            logger.info(f"Restarting {name} process (attempt {restart_counts[name]})")
            start_process(name, process_info["command"])

            return False

        # Process is still running
        return True

    return False


def start_all():
    """Start all required processes"""
    try:
        # Start fetch process
        start_process("fetch", "python fetch.py")

        # Wait for initial data to be fetched
        logger.info("Waiting for initial data to be fetched...")
        time.sleep(10)

        # Check if fetch process is still running
        if "fetch" not in processes:
            logger.error("Fetch process failed to start or terminated early")
            logger.error("Please check fetch_output.log for details")
            return False

        # Start train process
        start_process("train", "python train.py")

        # Wait for initial model to be trained
        logger.info("Waiting for initial model to be trained...")
        time.sleep(30)

        # Check if train process is still running
        if "train" not in processes:
            logger.error("Train process failed to start or terminated early")
            logger.error("Please check train_output.log for details")
            return False

        # Start serve process
        start_process("serve", "python serve.py")

        # Check if serve process is still running
        if "serve" not in processes:
            logger.error("Serve process failed to start or terminated early")
            logger.error("Please check serve_output.log for details")
            return False

        logger.info("All processes started")
        return True

    except Exception as e:
        logger.error(f"Error starting all processes: {e}")
        logger.error(traceback.format_exc())
        return False


def stop_all():
    """Stop all running processes"""
    for name in list(processes.keys()):
        stop_process(name)

    logger.info("All processes stopped")


def monitor_processes():
    """Monitor all processes and restart if needed"""
    while True:
        try:
            for name in list(processes.keys()):
                check_process(name)

            # If all processes have failed too many times, exit
            if all(count > MAX_RESTART_ATTEMPTS for count in restart_counts.values()):
                logger.error("All processes have failed too many times, exiting")
                return

            time.sleep(5)

        except Exception as e:
            logger.error(f"Error monitoring processes: {e}")
            logger.error(traceback.format_exc())
            time.sleep(5)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run the demand prediction system")

    parser.add_argument(
        "--fetch-only", action="store_true", help="Run only the fetch process"
    )
    parser.add_argument(
        "--train-only", action="store_true", help="Run only the train process"
    )
    parser.add_argument(
        "--serve-only", action="store_true", help="Run only the serve process"
    )
    parser.add_argument(
        "--reset", action="store_true", help="Reset all data and models before starting"
    )

    return parser.parse_args()


def reset_data():
    """Reset all data and models"""
    try:
        logger.info("Resetting all data and models")

        # Remove data directory
        if os.path.exists("data"):
            shutil.rmtree("data")
            logger.info("Removed data directory")

        # Remove models directory
        if os.path.exists("models"):
            shutil.rmtree("models")
            logger.info("Removed models directory")

        # Create empty directories
        os.makedirs("data")
        os.makedirs("models")

        logger.info("Reset complete")
        return True

    except Exception as e:
        logger.error(f"Error resetting data: {e}")
        logger.error(traceback.format_exc())
        return False


def main():
    """Main function"""
    args = parse_args()

    try:
        logger.info("Starting demand prediction system")

        # Check dependencies
        if not check_dependencies():
            logger.error("Dependency check failed, exiting")
            return

        # Reset data if requested
        if args.reset:
            if not reset_data():
                logger.error("Reset failed, exiting")
                return

        # Handle specific process requests
        if args.fetch_only:
            start_process("fetch", "python fetch.py")
            monitor_processes()
        elif args.train_only:
            start_process("train", "python train.py")
            monitor_processes()
        elif args.serve_only:
            start_process("serve", "python serve.py")
            monitor_processes()
        else:
            # Start all processes
            if start_all():
                # Monitor processes
                monitor_processes()
            else:
                logger.error("Failed to start all processes, exiting")
                stop_all()

    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt, shutting down")
        stop_all()

    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        stop_all()


if __name__ == "__main__":
    main()
