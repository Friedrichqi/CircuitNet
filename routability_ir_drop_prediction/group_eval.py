import os
import subprocess
for iters in range(1e4, 13e4+1, 1e4):
    print(f"Evaluating model_iters_{iters}.pth")")

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # save results to a file
    with open(f"./work_dir/congestion_gpdl/results_{iters}.txt", "w") as f:
        subprocess.run(f"python test.py --pretrained ./work_dir/congestion_gpdl/model_iters_{iters}.pth", shell=True, stdout=f)