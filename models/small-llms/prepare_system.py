#!/usr/bin/env python3
"""
System Preparation Script for GPT-OSS 20B
Automatically prepares your system for running large language models safely.
"""

import os
import sys
import subprocess
import psutil
import platform
import time
from pathlib import Path


class SystemPreparation:
    def __init__(self):
        self.killed_processes = []
        self.original_state = {}

    def check_permissions(self):
        """Check if we have necessary permissions"""
        if platform.system() == "Darwin":
            # Check if we can run system commands
            try:
                subprocess.run(
                    ["sysctl", "vm.swapusage"], capture_output=True, check=True
                )
                return True
            except:
                return False
        return True

    def get_memory_hog_processes(self, threshold_mb=500):
        """Find processes using more than threshold MB of memory"""
        print("   🔍 Scanning for memory-heavy processes...")
        memory_hogs = []

        try:
            # Add timeout protection
            start_time = time.time()
            count = 0

            for proc in psutil.process_iter(["pid", "name", "memory_info", "cmdline"]):
                count += 1
                # Timeout after 10 seconds or 1000 processes
                if time.time() - start_time > 10 or count > 1000:
                    print(f"   ⚠️  Process scan timeout (checked {count} processes)")
                    break

                try:
                    memory_mb = proc.info["memory_info"].rss / (1024 * 1024)
                    if memory_mb > threshold_mb:
                        # Skip system processes and our own Python
                        if proc.info["name"] not in [
                            "kernel_task",
                            "Python",
                            "python3",
                        ] and not any(
                            sys_name in proc.info["name"].lower()
                            for sys_name in [
                                "system",
                                "kernel",
                                "launchd",
                                "windowserver",
                            ]
                        ):
                            memory_hogs.append(
                                {
                                    "pid": proc.info["pid"],
                                    "name": proc.info["name"],
                                    "memory_mb": round(memory_mb, 1),
                                    "cmdline": (
                                        " ".join(proc.info["cmdline"][:3])
                                        if proc.info["cmdline"]
                                        else ""
                                    ),
                                }
                            )
                except (
                    psutil.NoSuchProcess,
                    psutil.AccessDenied,
                    psutil.ZombieProcess,
                ):
                    continue
                except Exception as e:
                    # Skip problematic processes
                    continue

            print(
                f"   ✅ Scanned {count} processes, found {len(memory_hogs)} memory hogs"
            )

        except Exception as e:
            print(f"   ❌ Error scanning processes: {e}")
            return []

        return sorted(memory_hogs, key=lambda x: x["memory_mb"], reverse=True)

    def identify_safe_to_kill_processes(self):
        """Identify processes that are generally safe to terminate"""
        safe_patterns = [
            "chrome",
            "firefox",
            "safari",
            "edge",  # Browsers
            "slack",
            "discord",
            "zoom",
            "teams",  # Communication
            "spotify",
            "music",
            "vlc",
            "quicktime",  # Media
            "photoshop",
            "illustrator",
            "sketch",  # Creative apps
            "code",
            "vscode",
            "sublime",
            "atom",  # Editors (if not current)
            "docker desktop",  # Docker
            "steam",
            "epic games",  # Games
        ]

        memory_hogs = self.get_memory_hog_processes()
        safe_to_kill = []

        for proc in memory_hogs:
            if any(pattern in proc["name"].lower() for pattern in safe_patterns):
                safe_to_kill.append(proc)

        return safe_to_kill

    def clear_system_caches(self):
        """Clear system caches to free memory"""
        print("🧹 Clearing system caches...")

        if platform.system() == "Darwin":
            try:
                # Clear memory pressure
                subprocess.run(["sudo", "purge"], check=True, capture_output=True)
                print("   ✅ System memory purged")
            except subprocess.CalledProcessError:
                print("   ⚠️  Could not purge memory (sudo required)")

            try:
                # Clear user caches
                cache_dirs = [
                    "~/Library/Caches",
                    "/System/Library/Caches",
                    "/Library/Caches",
                ]
                for cache_dir in cache_dirs:
                    expanded_dir = os.path.expanduser(cache_dir)
                    if os.path.exists(expanded_dir):
                        size_before = self.get_dir_size(expanded_dir)
                        # Only clear safe cache directories
                        subprocess.run(
                            ["find", expanded_dir, "-name", "*.cache", "-delete"],
                            capture_output=True,
                        )
                        print(f"   ✅ Cleared caches from {cache_dir}")
            except Exception as e:
                print(f"   ⚠️  Cache clearing issue: {e}")

        elif platform.system() == "Linux":
            try:
                # Drop caches
                subprocess.run(["sudo", "sync"], check=True)
                subprocess.run(
                    ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                    check=True,
                )
                print("   ✅ System caches dropped")
            except subprocess.CalledProcessError:
                print("   ⚠️  Could not drop caches (sudo required)")

    def get_dir_size(self, path):
        """Get directory size in MB"""
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    try:
                        total += os.path.getsize(filepath)
                    except:
                        continue
        except:
            pass
        return total / (1024 * 1024)

    def check_ollama_status(self):
        """Check if Ollama is ready"""
        print("🔍 Checking Ollama status...")

        try:
            # Check if Ollama is installed
            result = subprocess.run(
                ["ollama", "--version"], capture_output=True, text=True
            )
            if result.returncode == 0:
                print(f"   ✅ Ollama installed: {result.stdout.strip()}")
            else:
                print("   ❌ Ollama not found. Install with: brew install ollama")
                return False
        except FileNotFoundError:
            print("   ❌ Ollama not found. Install with: brew install ollama")
            return False

        try:
            # Check if Ollama service is running
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                print("   ✅ Ollama service is running")

                # Check if GPT-OSS model is available
                if "gpt-oss:20b" in result.stdout:
                    print("   ✅ GPT-OSS 20B model is downloaded")
                    return True
                else:
                    print("   ⚠️  GPT-OSS 20B model not found")
                    print("   💡 Download with: ollama pull gpt-oss:20b")
                    return False
            else:
                print("   ⚠️  Ollama service not responding")
                print("   💡 Start with: brew services start ollama")
                return False
        except subprocess.TimeoutExpired:
            print("   ⚠️  Ollama service timeout")
            return False
        except Exception as e:
            print(f"   ❌ Error checking Ollama: {e}")
            return False

    def interactive_process_killer(self):
        """Interactively kill memory-heavy processes"""
        safe_processes = self.identify_safe_to_kill_processes()

        if not safe_processes:
            print("✅ No obvious memory-heavy processes found to terminate.")
            return

        print(f"\n🎯 Found {len(safe_processes)} memory-heavy processes:")
        print("   (These are generally safe to terminate)")

        for i, proc in enumerate(safe_processes[:10], 1):
            print(
                f"   {i:2d}. {proc['name']} (PID: {proc['pid']}) - {proc['memory_mb']:.1f}MB"
            )

        print("\nOptions:")
        print("   'a' - Kill all listed processes")
        print("   '1,2,3' - Kill specific processes by number")
        print("   'n' - Skip process termination")

        choice = input("\nYour choice: ").strip().lower()

        if choice == "n":
            print("⏭️  Skipping process termination.")
            return
        elif choice == "a":
            processes_to_kill = safe_processes[:10]
        else:
            try:
                indices = [int(x.strip()) - 1 for x in choice.split(",")]
                processes_to_kill = [
                    safe_processes[i] for i in indices if 0 <= i < len(safe_processes)
                ]
            except (ValueError, IndexError):
                print("❌ Invalid selection. Skipping process termination.")
                return

        if not processes_to_kill:
            print("⏭️  No processes selected.")
            return

        print(f"\n🔄 Terminating {len(processes_to_kill)} processes...")
        for proc in processes_to_kill:
            try:
                psutil.Process(proc["pid"]).terminate()
                self.killed_processes.append(proc)
                print(f"   ✅ Terminated {proc['name']} (PID: {proc['pid']})")
            except psutil.NoSuchProcess:
                print(f"   ⚠️  Process {proc['name']} already terminated")
            except psutil.AccessDenied:
                print(f"   ❌ Access denied for {proc['name']} (may require sudo)")
            except Exception as e:
                print(f"   ❌ Error terminating {proc['name']}: {e}")

        # Wait for processes to terminate
        time.sleep(2)

    def display_resource_status(self):
        """Display current resource usage"""
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        print(f"\n📊 Current Resource Status:")
        print(
            f"   RAM: {memory.available/1024**3:.1f}GB available of {memory.total/1024**3:.1f}GB ({100-memory.percent:.1f}% free)"
        )
        print(
            f"   Swap: {swap.free/1024**3:.1f}GB available of {swap.total/1024**3:.1f}GB ({100-swap.percent:.1f}% free)"
        )
        print(f"   CPU: {psutil.cpu_percent(interval=1):.1f}% usage")

        # Temperature (macOS only)
        if platform.system() == "Darwin":
            try:
                result = subprocess.run(
                    ["pmset", "-g", "therm"], capture_output=True, text=True
                )
                if "CPU_Speed_Limit" in result.stdout:
                    print("   🌡️  Temperature: System is thermal throttling")
                else:
                    print("   🌡️  Temperature: Normal")
            except:
                pass

    def run_preparation(self):
        """Run the complete preparation process"""
        print("🚀 GPT-OSS 20B System Preparation")
        print("=" * 50)

        # Check permissions
        if not self.check_permissions():
            print("⚠️  Limited permissions. Some optimizations may not work.")

        # Initial resource check
        print("\n📊 Initial System Status:")
        self.display_resource_status()

        # Check Ollama
        if not self.check_ollama_status():
            print("\n❌ Ollama setup incomplete. Please resolve Ollama issues first.")
            return False

        # Clear caches
        print("\n🧹 System Cleanup:")
        self.clear_system_caches()

        # Process management
        print("\n🎯 Process Management:")
        self.interactive_process_killer()

        # Final status
        print("\n📊 Final System Status:")
        self.display_resource_status()

        if self.killed_processes:
            print(f"\n📝 Terminated Processes:")
            for proc in self.killed_processes:
                print(f"   • {proc['name']} (freed {proc['memory_mb']:.1f}MB)")

        print("\n✅ System preparation complete!")
        print("\n🎯 Next Steps:")
        print("   1. Save all important work")
        print('   2. Run: python run_small_model_demo.py --model gpt-oss-20b "Hello!"')
        print("   3. Monitor system resources during execution")
        print("   4. Be ready to force-quit if system becomes unresponsive")

        return True


def main():
    try:
        preparer = SystemPreparation()
        success = preparer.run_preparation()
        return 0 if success else 1

    except KeyboardInterrupt:
        print("\n\n⏹️  Preparation interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Error during preparation: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
