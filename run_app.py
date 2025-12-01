#!/usr/bin/env python3
"""
Run Script for 3BodyProblem Streamlit Application

This script checks system requirements and launches the Streamlit app.
Designed for deployment environments like Vercel, Railway, Render, etc.
"""

import sys
import os
import subprocess
import importlib.util
from pathlib import Path

# Color codes for terminal output (if supported)
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_colored(message, color=Colors.RESET):
    """Print colored message if terminal supports it."""
    try:
        print(f"{color}{message}{Colors.RESET}")
    except:
        print(message)

def check_python_version():
    """Check if Python version meets requirements."""
    print_colored("🐍 Checking Python version...", Colors.BLUE)
    
    min_version = (3, 11)
    current_version = sys.version_info[:2]
    
    if current_version < min_version:
        print_colored(
            f"❌ Python {min_version[0]}.{min_version[1]}+ required. "
            f"Found Python {current_version[0]}.{current_version[1]}",
            Colors.RED
        )
        return False
    
    print_colored(
        f"✓ Python {current_version[0]}.{current_version[1]} detected",
        Colors.GREEN
    )
    return True

def check_package(package_name, import_name=None):
    """
    Check if a package is installed.
    
    Args:
        package_name: Package name as installed (e.g., 'streamlit')
        import_name: Import name if different (e.g., 'st' for streamlit)
    
    Returns:
        bool: True if package is available
    """
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            return True
    except (ImportError, ModuleNotFoundError):
        pass
    
    return False

def check_requirements():
    """Check if all required packages are installed."""
    print_colored("\n📦 Checking required packages...", Colors.BLUE)
    
    # Required packages with their import names
    required_packages = {
        'numpy': 'numpy',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'plotly': 'plotly',
        'pandas': 'pandas',
        'seaborn': 'seaborn',
        'streamlit': 'streamlit',
    }
    
    missing_packages = []
    installed_packages = []
    
    for package_name, import_name in required_packages.items():
        if check_package(package_name, import_name):
            print_colored(f"  ✓ {package_name}", Colors.GREEN)
            installed_packages.append(package_name)
        else:
            print_colored(f"  ✗ {package_name} (missing)", Colors.RED)
            missing_packages.append(package_name)
    
    if missing_packages:
        print_colored(
            f"\n❌ Missing {len(missing_packages)} required package(s): {', '.join(missing_packages)}",
            Colors.RED
        )
        print_colored("\n💡 To install missing packages, run:", Colors.YELLOW)
        print_colored(f"   pip install {' '.join(missing_packages)}", Colors.YELLOW)
        print_colored("\n   Or install all requirements:", Colors.YELLOW)
        print_colored("   pip install -r requirements.txt", Colors.YELLOW)
        print_colored("   # or", Colors.YELLOW)
        print_colored("   pip install -e .", Colors.YELLOW)
        return False
    
    print_colored(f"\n✓ All {len(installed_packages)} required packages are installed", Colors.GREEN)
    return True

def check_project_structure():
    """Check if project structure is correct."""
    print_colored("\n📁 Checking project structure...", Colors.BLUE)
    
    required_paths = [
        'app/app.py',
        'core/body.py',
        'core/simulation.py',
        'core/solver.py',
        'visualization/plot.py',
    ]
    
    base_path = Path(__file__).parent
    missing_paths = []
    
    for path_str in required_paths:
        path = base_path / path_str
        if path.exists():
            print_colored(f"  ✓ {path_str}", Colors.GREEN)
        else:
            print_colored(f"  ✗ {path_str} (missing)", Colors.RED)
            missing_paths.append(path_str)
    
    if missing_paths:
        print_colored(
            f"\n❌ Missing {len(missing_paths)} required file(s)",
            Colors.RED
        )
        return False
    
    print_colored("✓ Project structure is valid", Colors.GREEN)
    return True

def get_streamlit_command():
    """Get the command to run Streamlit."""
    app_path = Path(__file__).parent / 'app' / 'app.py'
    
    if not app_path.exists():
        print_colored(f"❌ Streamlit app not found at {app_path}", Colors.RED)
        return None
    
    # Check if we're in a deployment environment
    port = os.environ.get('PORT', '8501')
    host = os.environ.get('HOST', '0.0.0.0')
    
    # For Vercel and similar platforms, we might need different handling
    # Streamlit typically needs a persistent server, which Vercel doesn't support well
    
    command = [
        sys.executable,
        '-m', 'streamlit', 'run',
        str(app_path),
        '--server.port', str(port),
        '--server.address', host,
        '--server.headless', 'true',
        '--browser.gatherUsageStats', 'false',
    ]
    
    return command

def run_app():
    """Run the Streamlit application."""
    print_colored("\n🚀 Starting Streamlit application...", Colors.BLUE)
    
    command = get_streamlit_command()
    if command is None:
        return False
    
    print_colored(f"   Command: {' '.join(command)}", Colors.YELLOW)
    print_colored("\n" + "="*60, Colors.BLUE)
    print_colored("🌌 3BodyProblem N-Body Gravitational Simulation", Colors.BOLD)
    print_colored("="*60 + "\n", Colors.BLUE)
    
    try:
        # Run Streamlit
        subprocess.run(command, check=True)
    except KeyboardInterrupt:
        print_colored("\n\n⚠️  Application stopped by user", Colors.YELLOW)
        return True
    except subprocess.CalledProcessError as e:
        print_colored(f"\n❌ Error running Streamlit: {e}", Colors.RED)
        return False
    except Exception as e:
        print_colored(f"\n❌ Unexpected error: {e}", Colors.RED)
        return False

def main():
    """Main entry point."""
    print_colored("\n" + "="*60, Colors.BOLD)
    print_colored("3BodyProblem - Application Launcher", Colors.BOLD)
    print_colored("="*60, Colors.BOLD)
    
    # Run checks
    checks_passed = True
    
    if not check_python_version():
        checks_passed = False
    
    if not check_requirements():
        checks_passed = False
    
    if not check_project_structure():
        checks_passed = False
    
    if not checks_passed:
        print_colored("\n❌ Pre-flight checks failed. Please fix the issues above.", Colors.RED)
        print_colored("\n💡 For deployment help, see README.md", Colors.YELLOW)
        sys.exit(1)
    
    print_colored("\n✓ All checks passed!", Colors.GREEN)
    
    # Run the app
    success = run_app()
    
    if not success:
        print_colored("\n❌ Failed to start application", Colors.RED)
        sys.exit(1)

if __name__ == "__main__":
    main()

