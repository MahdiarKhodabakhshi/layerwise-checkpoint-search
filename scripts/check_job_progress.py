#!/usr/bin/env python3
"""
Check progress of running SLURM jobs and estimate completion time.
"""

import subprocess
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

def get_job_info(job_id: str) -> Dict:
    """Get job information from scontrol."""
    try:
        result = subprocess.run(
            ['scontrol', 'show', 'job', job_id],
            capture_output=True,
            text=True,
            check=True
        )
        info = {}
        for line in result.stdout.split('\n'):
            if '=' in line:
                key, value = line.split('=', 1)
                info[key.strip()] = value.strip()
        return info
    except:
        return {}

def parse_time(time_str: str) -> float:
    """Parse SLURM time format (HH:MM:SS or D-HH:MM:SS) to hours."""
    if not time_str or time_str == 'N/A':
        return 0.0
    
    # Handle D-HH:MM:SS format
    if '-' in time_str:
        days, time_part = time_str.split('-')
        days = int(days)
    else:
        days = 0
        time_part = time_str
    
    parts = time_part.split(':')
    hours = int(parts[0])
    minutes = int(parts[1]) if len(parts) > 1 else 0
    seconds = int(parts[2]) if len(parts) > 2 else 0
    
    total_hours = days * 24 + hours + minutes / 60 + seconds / 3600
    return total_hours

def get_running_jobs() -> List[Dict]:
    """Get list of running jobs for user."""
    try:
        result = subprocess.run(
            ['squeue', '-u', 'mahdiar', '-o', '%.18i %.20j %.2t %.10M'],
            capture_output=True,
            text=True,
            check=True
        )
        jobs = []
        for line in result.stdout.strip().split('\n')[1:]:  # Skip header
            if line.strip():
                parts = line.split()
                if len(parts) >= 4:
                    job_id = parts[0]
                    job_name = parts[1]
                    state = parts[2]
                    time_str = parts[3]
                    if state == 'R':  # Running
                        jobs.append({
                            'job_id': job_id,
                            'name': job_name,
                            'state': state,
                            'runtime': time_str
                        })
        return jobs
    except:
        return []

def main():
    print("=" * 80)
    print("JOB PROGRESS AND TIME ESTIMATE")
    print("=" * 80)
    print(f"\nCurrent time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Get running jobs
    running_jobs = get_running_jobs()
    
    # Group by job name
    job_groups = {}
    for job in running_jobs:
        # Extract base job name (remove array task suffix)
        base_name = job['name']
        if '_' in job['job_id']:
            # Array job
            base_id = job['job_id'].split('_')[0]
        else:
            base_id = job['job_id']
        
        if base_id not in job_groups:
            job_groups[base_id] = {
                'name': base_name,
                'tasks': []
            }
        job_groups[base_id]['tasks'].append(job)
    
    # Analyze each job group
    for base_id, group in job_groups.items():
        print(f"\n{'=' * 80}")
        print(f"Job Group: {group['name']} (ID: {base_id})")
        print(f"{'=' * 80}")
        print(f"Running tasks: {len(group['tasks'])}")
        
        # Get job info for first task
        first_task_id = group['tasks'][0]['job_id']
        job_info = get_job_info(first_task_id)
        
        if job_info:
            time_limit = parse_time(job_info.get('TimeLimit', '0'))
            runtime = parse_time(group['tasks'][0]['runtime'])
            
            print(f"  Time limit: {time_limit:.1f} hours")
            print(f"  Current runtime: {runtime:.1f} hours")
            print(f"  Time remaining: {time_limit - runtime:.1f} hours")
            
            # Estimate completion
            if time_limit > 0:
                estimated_completion = datetime.now() + timedelta(hours=time_limit - runtime)
                print(f"  Estimated completion: {estimated_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Overall estimate
    print(f"\n{'=' * 80}")
    print("OVERALL ESTIMATE")
    print(f"{'=' * 80}")
    
    # Validation job (1858957): 4 tasks, should complete in ~15 hours
    # All-checkpoints job (1858946): 16 tasks, first 10 running, next 6 pending
    # First batch completes in ~15 hours, second batch takes another ~18 hours
    
    validation_completion = datetime.now() + timedelta(hours=15)
    first_batch_completion = datetime.now() + timedelta(hours=15)
    second_batch_completion = first_batch_completion + timedelta(hours=18)
    
    print(f"\nValidation job (1858957):")
    print(f"  - 4 tasks (layers 20-23)")
    print(f"  - Estimated completion: {validation_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nAll-checkpoints job (1858946):")
    print(f"  - 16 tasks total (10 running, 6 pending)")
    print(f"  - First batch (0-9): {first_batch_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  - Second batch (10-15): {second_batch_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\n{'=' * 80}")
    print(f"ALL TASKS SHOULD COMPLETE BY: {second_batch_completion.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"TOTAL TIME REMAINING: ~{(second_batch_completion - datetime.now()).total_seconds() / 3600:.1f} hours")
    print(f"  (approximately {int((second_batch_completion - datetime.now()).total_seconds() / 3600 / 24)} days)")
    print(f"{'=' * 80}")
    
    print("\nNOTE: These estimates assume:")
    print("  - Jobs complete within their 24-hour time limit")
    print("  - No failures requiring retries")
    print("  - Pending tasks start immediately when slots open")
    print("  - Task complexity is consistent across all evaluations")

if __name__ == '__main__':
    main()
