#!/usr/bin/env python3
"""
System Assessment for GPT-OSS 20B Local Execution
Analyzes hardware capabilities and provides recommendations for running large language models locally.
"""

import os
import sys
import subprocess
import json
import psutil
import platform
from pathlib import Path
import time


class SystemAssessment:
    def __init__(self):
        self.requirements = {
            "min_ram_gb": 24,  # Minimum RAM for 20B model
            "recommended_ram_gb": 32,  # Recommended RAM
            "min_free_disk_gb": 50,  # For model files and swap
            "min_cpu_cores": 4,
            "swap_multiplier": 1.5,  # Recommended swap as multiple of RAM shortage
        }

    def get_system_info(self):
        """Gather comprehensive system information"""
        info = {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "architecture": platform.architecture(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        }

        # Memory information
        memory = psutil.virtual_memory()
        info.update(
            {
                "total_ram_gb": round(memory.total / (1024**3), 2),
                "available_ram_gb": round(memory.available / (1024**3), 2),
                "used_ram_gb": round(memory.used / (1024**3), 2),
                "ram_percent": memory.percent,
            }
        )

        # Swap information
        swap = psutil.swap_memory()
        info.update(
            {
                "total_swap_gb": round(swap.total / (1024**3), 2),
                "used_swap_gb": round(swap.used / (1024**3), 2),
                "swap_percent": swap.percent,
            }
        )

        # Disk information
        disk = psutil.disk_usage("/")
        info.update(
            {
                "total_disk_gb": round(disk.total / (1024**3), 2),
                "free_disk_gb": round(disk.free / (1024**3), 2),
                "used_disk_gb": round(disk.used / (1024**3), 2),
                "disk_percent": round((disk.used / disk.total) * 100, 2),
            }
        )

        # GPU information (if available)
        info["gpu_info"] = self.get_gpu_info()

        return info

    def get_gpu_info(self):
        """Get GPU information if available"""
        gpu_info = {"has_gpu": False, "details": []}

        # Try NVIDIA GPU
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                gpu_info["has_gpu"] = True
                gpu_info["type"] = "NVIDIA"
                for line in result.stdout.strip().split("\n"):
                    if line.strip():
                        parts = [p.strip() for p in line.split(",")]
                        if len(parts) >= 4:
                            gpu_info["details"].append(
                                {
                                    "name": parts[0],
                                    "total_memory_mb": int(parts[1]),
                                    "free_memory_mb": int(parts[2]),
                                    "used_memory_mb": int(parts[3]),
                                }
                            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # Try AMD GPU (basic check)
        if not gpu_info["has_gpu"]:
            try:
                result = subprocess.run(
                    ["rocm-smi", "--showmeminfo", "vram"],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                if result.returncode == 0:
                    gpu_info["has_gpu"] = True
                    gpu_info["type"] = "AMD"
                    gpu_info["details"] = [
                        {"name": "AMD GPU", "info": "Basic detection only"}
                    ]
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass

        # Check for Apple Silicon GPU
        if platform.system() == "Darwin":
            gpu_info["apple_silicon"] = self.check_apple_silicon()

        return gpu_info

    def check_apple_silicon(self):
        """Check if running on Apple Silicon with Metal support"""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
            )
            if "Apple" in result.stdout:
                return {"type": "Apple Silicon", "metal_support": True}
        except:
            pass
        return {"type": "Unknown", "metal_support": False}

    def get_resource_hungry_processes(self):
        """Find processes using significant CPU or memory"""
        processes = []
        for proc in psutil.process_iter(
            ["pid", "name", "cpu_percent", "memory_percent", "memory_info"]
        ):
            try:
                cpu_percent = proc.info["cpu_percent"] or 0
                memory_percent = proc.info["memory_percent"] or 0
                if cpu_percent > 5 or memory_percent > 2:
                    processes.append(
                        {
                            "pid": proc.info["pid"],
                            "name": proc.info["name"],
                            "cpu_percent": round(proc.info["cpu_percent"], 1),
                            "memory_percent": round(proc.info["memory_percent"], 1),
                            "memory_mb": round(
                                proc.info["memory_info"].rss / (1024 * 1024), 1
                            ),
                        }
                    )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue

        # Sort by memory usage
        processes.sort(key=lambda x: x["memory_percent"], reverse=True)
        return processes[:10]  # Top 10

    def assess_model_compatibility(self, info):
        """Assess if system can run GPT-OSS 20B"""
        assessment = {
            "can_run": True,
            "confidence": "high",
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # RAM Assessment
        if info["total_ram_gb"] < self.requirements["min_ram_gb"]:
            assessment["can_run"] = False
            assessment["confidence"] = "low"
            shortage = self.requirements["min_ram_gb"] - info["total_ram_gb"]
            assessment["issues"].append(
                f"Insufficient RAM: {info['total_ram_gb']}GB available, "
                f"{self.requirements['min_ram_gb']}GB minimum required. "
                f"Short by {shortage:.1f}GB."
            )
        elif info["total_ram_gb"] < self.requirements["recommended_ram_gb"]:
            assessment["confidence"] = "medium"
            assessment["warnings"].append(
                f"RAM below recommended: {info['total_ram_gb']}GB available, "
                f"{self.requirements['recommended_ram_gb']}GB recommended."
            )

        # Available RAM check
        if info["available_ram_gb"] < 16:
            assessment["warnings"].append(
                f"Low available RAM: {info['available_ram_gb']}GB free. "
                f"Consider closing other applications."
            )

        # Swap Assessment
        if info["total_swap_gb"] < 8:
            if info["total_ram_gb"] < self.requirements["recommended_ram_gb"]:
                recommended_swap = (
                    self.requirements["recommended_ram_gb"] - info["total_ram_gb"]
                ) * self.requirements["swap_multiplier"]
                assessment["recommendations"].append(
                    f"Increase swap space to {recommended_swap:.1f}GB to compensate for limited RAM."
                )

        # Disk Space
        if info["free_disk_gb"] < self.requirements["min_free_disk_gb"]:
            assessment["issues"].append(
                f"Insufficient disk space: {info['free_disk_gb']}GB free, "
                f"{self.requirements['min_free_disk_gb']}GB required."
            )
            assessment["can_run"] = False

        # CPU Assessment
        if info["cpu_count"] < self.requirements["min_cpu_cores"]:
            assessment["warnings"].append(
                f"Low CPU core count: {info['cpu_count']} cores, "
                f"{self.requirements['min_cpu_cores']} recommended."
            )

        # GPU Assessment
        if info["gpu_info"]["has_gpu"]:
            if info["gpu_info"]["type"] == "NVIDIA":
                total_gpu_memory = sum(
                    gpu["total_memory_mb"] for gpu in info["gpu_info"]["details"]
                )
                if total_gpu_memory < 12000:  # 12GB
                    assessment["warnings"].append(
                        f"Limited GPU memory: {total_gpu_memory/1024:.1f}GB. "
                        f"Model may need to run on CPU/System RAM."
                    )
                else:
                    assessment["recommendations"].append(
                        f"Good GPU memory: {total_gpu_memory/1024:.1f}GB available. "
                        f"Consider GPU acceleration."
                    )
            elif "apple_silicon" in info["gpu_info"]:
                assessment["recommendations"].append(
                    "Apple Silicon detected. Consider using Metal Performance Shaders (MPS) for acceleration."
                )
        else:
            assessment["warnings"].append(
                "No dedicated GPU detected. Model will run on CPU only (slower)."
            )

        return assessment

    def generate_preparation_steps(self, info, assessment):
        """Generate specific steps to prepare the system"""
        steps = []

        # Memory optimization steps
        if info["available_ram_gb"] < 16 or assessment["confidence"] != "high":
            steps.append(
                {
                    "category": "Memory Optimization",
                    "steps": [
                        "Close unnecessary applications (browsers, media players, IDEs)",
                        "Quit memory-intensive processes",
                        "Clear system caches: sudo purge (macOS) or echo 3 | sudo tee /proc/sys/vm/drop_caches (Linux)",
                        "Restart your computer for maximum available memory",
                    ],
                }
            )

        # Process management
        resource_processes = self.get_resource_hungry_processes()
        if resource_processes:
            process_steps = ["Consider stopping these resource-heavy processes:"]
            for p in resource_processes[:5]:
                memory_mb = p.get("memory_mb", 0)
                cpu_percent = p.get("cpu_percent", 0)
                process_steps.append(
                    f"  • {p['name']} (PID: {p['pid']}) - {memory_mb:.1f}MB, {cpu_percent}% CPU"
                )

            steps.append({"category": "Process Management", "steps": process_steps})

        # Swap recommendations
        if info["total_swap_gb"] < 16 and info["total_ram_gb"] < 32:
            if platform.system() == "Darwin":
                steps.append(
                    {
                        "category": "Swap Space (macOS)",
                        "steps": [
                            "macOS manages swap automatically",
                            "Ensure at least 50GB free disk space for dynamic swap",
                            "Consider increasing VM settings if experienced with terminal",
                        ],
                    }
                )
            else:
                steps.append(
                    {
                        "category": "Swap Space (Linux)",
                        "steps": [
                            "Create additional swap file:",
                            "  sudo fallocate -l 16G /swapfile",
                            "  sudo chmod 600 /swapfile",
                            "  sudo mkswap /swapfile",
                            "  sudo swapon /swapfile",
                        ],
                    }
                )

        # Model-specific preparations
        steps.append(
            {
                "category": "Model Preparation",
                "steps": [
                    "Ensure Ollama is installed and running: brew install ollama",
                    "Pull the model: ollama pull gpt-oss:20b",
                    "Close other AI/ML applications",
                    "Set GPU memory growth (if using TensorFlow/similar)",
                    "Consider running during low-activity hours",
                ],
            }
        )

        # Monitoring steps
        steps.append(
            {
                "category": "Monitoring",
                "steps": [
                    "Open Activity Monitor (macOS) or htop (Linux) to watch resource usage",
                    "Monitor temperature: sudo powermetrics -n 1 -s thermal (macOS)",
                    "Have 'Force Quit' ready in case of system freeze",
                    "Save all important work before starting",
                ],
            }
        )

        return steps

    def run_assessment(self):
        """Run complete system assessment"""
        print("🔍 GPT-OSS 20B System Assessment")
        print("=" * 50)

        print("\n📊 Gathering system information...")
        info = self.get_system_info()

        print(f"\n💻 System Overview:")
        print(f"   Platform: {info['platform']}")
        print(
            f"   CPU: {info['processor']} ({info['cpu_count']} cores, {info['cpu_count_logical']} logical)"
        )
        print(
            f"   RAM: {info['total_ram_gb']}GB total, {info['available_ram_gb']}GB available ({info['ram_percent']:.1f}% used)"
        )
        print(
            f"   Swap: {info['total_swap_gb']}GB total ({info['swap_percent']:.1f}% used)"
        )
        print(f"   Disk: {info['free_disk_gb']}GB free of {info['total_disk_gb']}GB")

        if info["gpu_info"]["has_gpu"]:
            print(f"   GPU: {info['gpu_info']['type']}")
            if info["gpu_info"]["details"]:
                for gpu in info["gpu_info"]["details"]:
                    if "total_memory_mb" in gpu:
                        print(
                            f"     • {gpu['name']}: {gpu['total_memory_mb']/1024:.1f}GB"
                        )
        else:
            print("   GPU: None detected")

        print("\n🔬 Compatibility Assessment...")
        assessment = self.assess_model_compatibility(info)

        # Display assessment results
        if assessment["can_run"]:
            confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴"}
            print(
                f"\n{confidence_emoji[assessment['confidence']]} Assessment Result: CAN RUN (Confidence: {assessment['confidence']})"
            )
        else:
            print(f"\n🔴 Assessment Result: CANNOT RUN")

        if assessment["issues"]:
            print(f"\n❌ Critical Issues:")
            for issue in assessment["issues"]:
                print(f"   • {issue}")

        if assessment["warnings"]:
            print(f"\n⚠️  Warnings:")
            for warning in assessment["warnings"]:
                print(f"   • {warning}")

        if assessment["recommendations"]:
            print(f"\n💡 Recommendations:")
            for rec in assessment["recommendations"]:
                print(f"   • {rec}")

        # Preparation steps
        print("\n🛠️  Preparation Steps:")
        steps = self.generate_preparation_steps(info, assessment)
        for step_group in steps:
            print(f"\n📋 {step_group['category']}:")
            for step in step_group["steps"]:
                print(f"   {step}")

        # Final recommendation
        print(f"\n🎯 Final Recommendation:")
        if assessment["can_run"]:
            if assessment["confidence"] == "high":
                print(
                    "   ✅ Your system should handle GPT-OSS 20B well. Follow preparation steps and monitor resources."
                )
            elif assessment["confidence"] == "medium":
                print(
                    "   ⚠️  Your system can run GPT-OSS 20B but may be slow. Consider the preparation steps carefully."
                )
            else:
                print(
                    "   🔴 Your system might struggle. Consider using smaller models or cloud alternatives."
                )
        else:
            print("   ❌ Your system cannot safely run GPT-OSS 20B. Consider:")
            print("      • Using smaller models (llama3.2:1b, qwen-0.5b, qwen3-4b-instruct)")
            print("      • Cloud solutions (Google Colab, AWS, etc.)")
            print("      • Upgrading hardware (more RAM)")

        return info, assessment


def main():
    try:
        assessor = SystemAssessment()
        info, assessment = assessor.run_assessment()

        # Save results
        results = {
            "timestamp": time.time(),
            "system_info": info,
            "assessment": assessment,
        }

        # Save results in assessments directory with timestamp
        assessments_dir = Path(__file__).parent / "assessments"
        assessments_dir.mkdir(exist_ok=True)

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        results_file = assessments_dir / f"system_assessment_{timestamp_str}.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

    except KeyboardInterrupt:
        print("\n\n⏹️  Assessment interrupted by user.")
    except Exception as e:
        print(f"\n❌ Error during assessment: {e}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
