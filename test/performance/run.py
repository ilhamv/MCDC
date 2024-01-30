import os, math
import numpy as np


def run_job(N_node, N_particle, tag):
    """
    N_node: number of nodes
    N_particle: number of particles per node
    tag: tag for the job
    """
    N_rank = N_node * 36  # Quartz
    job_ID = tag + "_%s" % str(int(math.log10(N_particle)))
    output = tag + "_%s_%s" % (str(N_node), str(int(math.log10(N_particle))))

    with open("job.pbs", "w") as f:
        text = (
            "#!/bin/tcsh\n srun -n %i python input.py --mode=numba --N_particle=%i --output=%s"
            % (
                N_rank,
                N_particle,
                output,
            )
        )
        f.write(text)

    os.system(
        "sbatch -N %i -J %s -t 12:00:00 -p pbatch -A orsu -o %s.out job.pbs"
        % (N_node, job_ID, output)
    )
    os.remove("job.pbs")


task = {
    "c5g7_tdx": [1e4, 1e5, 1e6, 1e7],
    "kobayashi_g7": [1e4, 1e5, 1e6, 1e7],
    "c5g7_tdx-branchless_collision": [1e4, 1e5, 1e6, 1e7],
    "kobayashi_g7-implicit_capture": [1e4, 1e5, 1e6, 1e7],
}

for tag in task.keys():
    if not os.path.isdir(tag):
        continue
    os.chdir(tag)

    for N_particle in task[tag]:
        for N_node in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            run_job(N_node, N_particle, tag)

    os.chdir(r"..")
