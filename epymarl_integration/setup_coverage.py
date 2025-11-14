#!/usr/bin/env python3
"""
Automated Setup Script for Coverage Environment
Integrates coverage environment into EPyMARL installation.

Usage:
    python setup_coverage.py [path/to/epymarl]

If no path is provided, assumes EPyMARL is in ./epymarl/
"""

import os
import sys
import shutil
from pathlib import Path


def find_epymarl_root(start_path=None):
    """Find EPyMARL root directory."""
    if start_path:
        path = Path(start_path)
        if (path / "src" / "envs").exists():
            return path
        print(f"Error: {start_path} doesn't look like EPyMARL root")
        return None

    # Try common locations
    candidates = [
        Path("./epymarl"),
        Path("../epymarl"),
        Path("."),
    ]

    for candidate in candidates:
        if (candidate / "src" / "envs").exists():
            return candidate

    return None


def backup_file(file_path):
    """Create backup of existing file."""
    if file_path.exists():
        backup_path = Path(str(file_path) + ".backup")
        shutil.copy(file_path, backup_path)
        print(f"  ✓ Backed up {file_path.name} to {backup_path.name}")


def copy_with_backup(src, dst):
    """Copy file, backing up destination if it exists."""
    dst_path = Path(dst)
    if dst_path.exists():
        backup_file(dst_path)

    shutil.copy(src, dst)
    print(f"  ✓ Copied {Path(src).name} to {dst}")


def modify_envs_init(epymarl_root):
    """Modify src/envs/__init__.py to register coverage environment."""
    init_file = epymarl_root / "src" / "envs" / "__init__.py"

    if not init_file.exists():
        print(f"  ✗ Could not find {init_file}")
        return False

    # Read current content
    with open(init_file, 'r') as f:
        content = f.read()

    # Check if already modified
    if "from .coverage import CoverageEnv" in content:
        print("  ⚠ Coverage already registered in __init__.py")
        return True

    # Backup
    backup_file(init_file)

    # Add import
    lines = content.split('\n')
    import_index = -1

    # Find where to add import (after other env imports)
    for i, line in enumerate(lines):
        if "from .smacv2 import SMACv2" in line or "from .starcraft2" in line:
            import_index = i + 1

    if import_index == -1:
        print("  ✗ Could not find where to add import")
        return False

    lines.insert(import_index, "from .coverage import CoverageEnv")

    # Add to env_REGISTRY function
    registry_index = -1
    for i, line in enumerate(lines):
        if "def env_REGISTRY" in line:
            registry_index = i
            break

    if registry_index == -1:
        print("  ✗ Could not find env_REGISTRY function")
        return False

    # Find where to add coverage case (before else clause)
    else_index = -1
    for i in range(registry_index, len(lines)):
        if "else:" in lines[i] and "raise ValueError" in lines[i+1]:
            else_index = i
            break

    if else_index == -1:
        print("  ✗ Could not find where to add coverage case")
        return False

    # Add coverage case
    lines.insert(else_index, "    elif env_name == \"coverage\":")
    lines.insert(else_index + 1, "        return CoverageEnv")
    lines.insert(else_index + 2, "")

    # Write back
    with open(init_file, 'w') as f:
        f.write('\n'.join(lines))

    print("  ✓ Modified src/envs/__init__.py")
    return True


def setup_coverage_env(epymarl_root, integration_dir):
    """Set up coverage environment in EPyMARL."""

    print("\n" + "="*60)
    print("Setting up Coverage Environment in EPyMARL")
    print("="*60)

    # 1. Create coverage directory
    print("\n[1/4] Creating coverage directory...")
    coverage_dir = epymarl_root / "src" / "envs" / "coverage"
    coverage_dir.mkdir(exist_ok=True)
    print(f"  ✓ Created {coverage_dir}")

    # 2. Copy coverage environment files
    print("\n[2/4] Copying coverage environment files...")

    # Copy main environment
    copy_with_backup(
        integration_dir / "coverage_env.py",
        coverage_dir / "coverage_env.py"
    )

    # Copy __init__.py
    copy_with_backup(
        integration_dir / "coverage__init__.py",
        coverage_dir / "__init__.py"
    )

    # 3. Modify src/envs/__init__.py
    print("\n[3/4] Registering coverage environment...")
    if not modify_envs_init(epymarl_root):
        print("  ⚠ Warning: Could not automatically modify __init__.py")
        print("  Please manually add coverage registration (see envs__init__.py)")

    # 4. Copy config file
    print("\n[4/4] Copying configuration file...")
    config_dir = epymarl_root / "src" / "config" / "envs"
    config_dir.mkdir(parents=True, exist_ok=True)

    copy_with_backup(
        integration_dir / "coverage.yaml",
        config_dir / "coverage.yaml"
    )

    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)

    return True


def verify_installation(epymarl_root):
    """Verify that coverage environment is properly installed."""

    print("\nVerifying installation...")

    checks = [
        (epymarl_root / "src" / "envs" / "coverage" / "coverage_env.py",
         "Coverage environment file"),
        (epymarl_root / "src" / "envs" / "coverage" / "__init__.py",
         "Coverage __init__ file"),
        (epymarl_root / "src" / "config" / "envs" / "coverage.yaml",
         "Coverage config file"),
    ]

    all_good = True
    for path, name in checks:
        if path.exists():
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name} NOT FOUND")
            all_good = False

    return all_good


def print_usage_instructions(epymarl_root):
    """Print usage instructions."""

    print("\n" + "="*60)
    print("Next Steps")
    print("="*60)

    print("\n1. Test the environment:")
    print(f"   cd {epymarl_root}")
    print("   python -c \"from src.envs.coverage import CoverageEnv; env = CoverageEnv(); print('OK')\"")

    print("\n2. Train with QMIX:")
    print("   python src/main.py --config=qmix --env-config=coverage")

    print("\n3. Train with IQL:")
    print("   python src/main.py --config=iql --env-config=coverage")

    print("\n4. Custom parameters:")
    print("   python src/main.py --config=qmix --env-config=coverage \\")
    print("       with env_args.n_agents=4 env_args.grid_size=30")

    print("\n5. Monitor with TensorBoard:")
    print("   tensorboard --logdir results/tb_logs")

    print("\n" + "="*60)


def main():
    """Main setup function."""

    print("="*60)
    print("Coverage Environment Setup for EPyMARL")
    print("="*60)

    # Get EPyMARL path
    if len(sys.argv) > 1:
        epymarl_path = sys.argv[1]
    else:
        epymarl_path = None

    # Find EPyMARL
    print("\nSearching for EPyMARL installation...")
    epymarl_root = find_epymarl_root(epymarl_path)

    if epymarl_root is None:
        print("\n✗ Could not find EPyMARL installation")
        print("\nPlease either:")
        print("  1. Run this script from EPyMARL directory")
        print("  2. Provide EPyMARL path: python setup_coverage.py /path/to/epymarl")
        print("  3. Clone EPyMARL: git clone https://github.com/uoe-agents/epymarl.git")
        sys.exit(1)

    print(f"✓ Found EPyMARL at: {epymarl_root.absolute()}")

    # Get integration directory (where this script is)
    integration_dir = Path(__file__).parent

    # Setup
    try:
        success = setup_coverage_env(epymarl_root, integration_dir)

        if success:
            # Verify
            if verify_installation(epymarl_root):
                print("\n✓ Installation verified successfully!")
                print_usage_instructions(epymarl_root)
            else:
                print("\n⚠ Installation incomplete - some files missing")
                sys.exit(1)
        else:
            print("\n✗ Setup failed")
            sys.exit(1)

    except Exception as e:
        print(f"\n✗ Error during setup: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
