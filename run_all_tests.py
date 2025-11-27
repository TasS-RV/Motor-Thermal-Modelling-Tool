"""
Script to run all test configurations from backup log files in parallel.
Each test runs in a separate process with its own configuration.
"""
import os
import re
import subprocess
import sys
from pathlib import Path
from multiprocessing import Process

def parse_log_file(log_path):
    """Parse experiment log backup file to extract parameters."""
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract folder name from path
    folder_name = Path(log_path).parent.name
    
    # Parse description
    desc_match = re.search(r'EXPERIMENT DESCRIPTION:\n-+\n(.*?)\n\nKEY PARAMETERS:', content, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""
    
    # Parse geometry (in mm, convert to m)
    space_length = float(re.search(r'Space Length: ([\d.]+) mm', content).group(1)) / 1000
    L_air_inner = float(re.search(r'Air Inner.*?: ([\d.]+) mm', content).group(1)) / 1000
    t_insulation = float(re.search(r'Insulation Thickness.*?: ([\d.]+) mm', content).group(1)) / 1000
    t_oil_margin = float(re.search(r'Oil Margin.*?: ([\d.]+) mm', content).group(1)) / 1000
    D_depth = float(re.search(r'Depth.*?: ([\d.]+) mm', content).group(1)) / 1000
    L_heater_block = float(re.search(r'Heater Block Size.*?: ([\d.]+) mm', content).group(1)) / 1000
    Resolution = int(re.search(r'Grid Resolution: (\d+)×', content).group(1))
    
    # Parse temperatures (already in K)
    T_oil_setpoint = float(re.search(r'Oil Setpoint: ([\d.]+) K', content).group(1))
    T_ambient = float(re.search(r'Ambient Temperature: ([\d.]+) K', content).group(1))
    T_initial = float(re.search(r'Initial Temperature: ([\d.]+) K', content).group(1))
    
    # Parse oil configuration
    oil_placement = re.search(r'Oil Placement: (\w+)', content).group(1)
    oil_finite_str = re.search(r'Oil Type: (FINITE|INFINITE)', content).group(1)
    OIL_IS_FINITE = (oil_finite_str == "FINITE")
    
    # Parse heat source
    Q_input_watts = float(re.search(r'Heater Power: ([\d.]+) W', content).group(1))
    
    # Parse simulation settings
    Time_Total = float(re.search(r'Total Simulation Time: ([\d.]+) s', content).group(1))
    SNAPSHOT_INTERVAL_S = float(re.search(r'Snapshot Interval: ([\d.]+) s', content).group(1))
    Animation_Speedup = float(re.search(r'Animation Speedup: ([\d.]+)x', content).group(1))
    
    return {
        'folder_name': folder_name,
        'description': description,
        'space_length': space_length,
        'L_air_inner': L_air_inner,
        't_insulation': t_insulation,
        't_oil_margin': t_oil_margin,
        'D_depth': D_depth,
        'L_heater_block': L_heater_block,
        'Resolution': Resolution,
        'T_oil_setpoint': T_oil_setpoint,
        'T_ambient': T_ambient,
        'T_initial': T_initial,
        'OIL_PLACEMENT': oil_placement,
        'OIL_IS_FINITE': OIL_IS_FINITE,
        'Q_input_watts': Q_input_watts,
        'Time_Total': Time_Total,
        'SNAPSHOT_INTERVAL_S': SNAPSHOT_INTERVAL_S,
        'Animation_Speedup': Animation_Speedup,
    }

def create_test_script(params):
    """Create a complete standalone Python script for running a specific test."""
    script_name = f"run_{params['folder_name'].replace(' ', '_').replace('-', '_')}.py"
    
    # Read the base model file as a string first for description replacement
    with open('new_model.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace EXPERIMENT_DESCRIPTION block using regex (handles multi-line)
    desc_pattern = r'EXPERIMENT_DESCRIPTION = """.*?"""'
    # Get indent from original
    indent_match = re.search(r'^( *?)EXPERIMENT_DESCRIPTION = """', content, re.MULTILINE)
    desc_indent = indent_match.group(1) if indent_match else ""
    
    # Build replacement description
    desc_replacement = f'{desc_indent}EXPERIMENT_DESCRIPTION = """\n'
    if params['description']:
        for desc_line in params['description'].split('\n'):
            desc_replacement += f'{desc_indent}{desc_line}\n'
    desc_replacement += f'{desc_indent}"""'
    
    content = re.sub(desc_pattern, desc_replacement, content, flags=re.DOTALL)
    
    # Now convert to lines for other parameter replacements
    lines = content.splitlines(keepends=True)
    modified_lines = []
    
    for line in lines:
        modified_line = line
        
        # Replace single-line parameters
        replacements = {
            r'OIL_IS_FINITE = \w+': f'OIL_IS_FINITE = {params["OIL_IS_FINITE"]}',
            r'OIL_PLACEMENT = "[^"]*"': f'OIL_PLACEMENT = "{params["OIL_PLACEMENT"]}"',
            r'SNAPSHOT_FOLDER\s*=\s*"[^"]*"': f'SNAPSHOT_FOLDER = "{params["folder_name"]}"',
            r'SNAPSHOT_INTERVAL_S\s*=\s*[\d.]+': f'SNAPSHOT_INTERVAL_S = {params["SNAPSHOT_INTERVAL_S"]}',
            r'space_length = [\d.]+': f'space_length = {params["space_length"]}',
            r'L_air_inner\s*=\s*[\d.]+': f'L_air_inner = {params["L_air_inner"]}',
            r't_insulation\s*=\s*[\d.]+': f't_insulation = {params["t_insulation"]}',
            r't_oil_margin\s*=\s*[\d.]+': f't_oil_margin = {params["t_oil_margin"]}',
            r'D_depth\s*=\s*space_length[^\\n]*': f'D_depth = {params["D_depth"]}  # Set from backup log',
            r'L_heater_block\s*=\s*[\d.]+': f'L_heater_block = {params["L_heater_block"]}',
            r'Resolution\s*=\s*\d+': f'Resolution = {params["Resolution"]}',
            r'Q_input_watts\s*=\s*[\d.]+': f'Q_input_watts = {params["Q_input_watts"]}',
            r'T_oil_setpoint\s*=\s*273.15[^\\n]*': f'T_oil_setpoint = {params["T_oil_setpoint"]}  # From backup log',
            r'T_ambient\s*=\s*273.15[^\\n]*': f'T_ambient = {params["T_ambient"]}  # From backup log',
            r'T_initial\s*=\s*293.15': f'T_initial = {params["T_initial"]}',
            r'Time_Total\s*=\s*[\d.*+\-()\s]+': f'Time_Total = {params["Time_Total"]}  # From backup log',
            r'Animation_Speedup\s*=\s*[\d.]+': f'Animation_Speedup = {params["Animation_Speedup"]}',
        }
        
        for pattern, replacement in replacements.items():
            if re.search(pattern, modified_line):
                indent = modified_line[:len(modified_line) - len(modified_line.lstrip())]
                modified_line = indent + replacement + '\n'
                break
        
        modified_lines.append(modified_line)
    
    # Verify that EXPERIMENT_DESCRIPTION was set and fix if missing
    script_content = ''.join(modified_lines)
    if 'EXPERIMENT_DESCRIPTION = """' not in script_content:
        # Fallback: add it after the comment if missing
        for i, line in enumerate(modified_lines):
            if '# --- Experiment Description ---' in line or ('Experiment Description' in line and '---' in line):
                # Find the indent (usually no indent for these comments)
                indent = ""
                # Build the description block
                desc_block = f'{indent}EXPERIMENT_DESCRIPTION = """\n'
                if params['description']:
                    for desc_line in params['description'].split('\n'):
                        if desc_line.strip():  # Only add non-empty lines
                            desc_block += f'{indent}{desc_line}\n'
                desc_block += f'{indent}"""\n'
                # Insert after the comment line (skip blank line if present)
                insert_idx = i + 1
                if insert_idx < len(modified_lines) and not modified_lines[insert_idx].strip():
                    insert_idx += 1  # Skip the blank line
                modified_lines.insert(insert_idx, desc_block)
                break
    
    # Write the modified script
    with open(script_name, 'w', encoding='utf-8') as f:
        f.writelines(modified_lines)
    
    return script_name

def run_test_in_terminal(script_name, test_name):
    """Launch a test script in a new terminal window (Windows)."""
    import platform
    if platform.system() == 'Windows':
        # Use Windows 'start' command to open new terminal window
        cmd = f'start "Test: {test_name}" cmd /k python "{script_name}"'
        subprocess.Popen(cmd, shell=True)
    else:
        # For Linux/Mac, use xterm or gnome-terminal
        subprocess.Popen(['gnome-terminal', '--', 'python3', script_name])

def run_test(script_name, test_name):
    """Run a test script in background (for Process-based parallel execution)."""
    print(f"[{test_name}] Starting...")
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        if result.returncode == 0:
            print(f"[{test_name}] ✓ Completed successfully")
        else:
            print(f"[{test_name}] ✗ Failed with return code {result.returncode}")
    except Exception as e:
        print(f"[{test_name}] ✗ Error: {e}")

def main():
    print("="*70)
    print("Test Configuration Runner")
    print("="*70)
    
    # Find all backup log files
    backup_files = sorted(list(Path('.').glob('**/experiment_log_backup.txt')))
    
    if not backup_files:
        print("\n✗ No experiment_log_backup.txt files found!")
        return
    
    print(f"\nFound {len(backup_files)} test configurations:")
    for bf in backup_files:
        print(f"  - {bf.parent.name}")
    
    # Parse each backup file
    print("\n" + "="*70)
    print("Parsing configuration files...")
    test_configs = []
    for backup_file in backup_files:
        try:
            params = parse_log_file(backup_file)
            test_configs.append(params)
            print(f"  ✓ Parsed: {params['folder_name']}")
        except Exception as e:
            print(f"  ✗ Error parsing {backup_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not test_configs:
        print("\n✗ No valid test configurations found!")
        return
    
    # Clean up old scripts first (optional, but recommended)
    print("\n" + "="*70)
    print("Cleaning up old test scripts...")
    old_scripts = list(Path('.').glob('run_Test_*.py'))
    if old_scripts:
        for old_script in old_scripts:
            try:
                old_script.unlink()
                print(f"  Deleted: {old_script.name}")
            except:
                pass
    
    # Create individual test scripts
    print("\n" + "="*70)
    print("Creating test scripts...")
    scripts = []
    for params in test_configs:
        try:
            script_name = create_test_script(params)
            scripts.append((script_name, params['folder_name']))
            print(f"  ✓ Created: {script_name}")
        except Exception as e:
            print(f"  ✗ Error creating script for {params['folder_name']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not scripts:
        print("\n✗ No scripts created!")
        return
    
    # Ask user for confirmation
    print("\n" + "="*70)
    print(f"Ready to run {len(scripts)} tests in parallel.")
    print("Each test will run in a separate process.")
    response = input("\nContinue? (y/n): ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    # Ask user how to run
    print("\n" + "="*70)
    print("How would you like to run the tests?")
    print("  1. In separate terminal windows (each test visible)")
    print("  2. In parallel background processes (all at once, not visible)")
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        # Launch each test in a separate terminal window
        print("\n" + "="*70)
        print("Launching tests in separate terminal windows...")
        print("="*70 + "\n")
        
        for script_name, test_name in scripts:
            run_test_in_terminal(script_name, test_name)
            print(f"  Launched terminal window for: {test_name}")
            import time
            time.sleep(0.5)  # Small delay to avoid overwhelming the system
        
        print(f"\n✓ Launched {len(scripts)} terminal windows")
        print("Each test is running in its own window.")
        print("Close this window when ready - tests will continue running.")
        
    else:
        # Run all tests in parallel background processes
        print("\n" + "="*70)
        print("Starting parallel execution in background...")
        print("="*70 + "\n")
        
        processes = []
        for script_name, test_name in scripts:
            p = Process(target=run_test, args=(script_name, test_name))
            p.start()
            processes.append((p, test_name))
            print(f"  Started process for: {test_name}")
        
        # Wait for all processes to complete
        print(f"\nWaiting for {len(processes)} processes to complete...")
        print("(This may take a long time depending on simulation duration)\n")
        
        for p, test_name in processes:
            p.join()
            print(f"  Process finished: {test_name}")
        
        print("\n" + "="*70)
        print("All tests completed!")
        print("="*70)

if __name__ == '__main__':
    main()
