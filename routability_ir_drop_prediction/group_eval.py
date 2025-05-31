import os
import subprocess
for iters in range(10000, 130001, 10000):
    print(f"Evaluating model_iters_{iters}.pth")

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # save results to a file
    with open(f"./work_dir/congestion_gpdl/results_{iters}.txt", "w") as f:
        subprocess.run(f"python test.py --pretrained ./work_dir/congestion_gpdl/model_iters_{iters}.pth", shell=True, stdout=f)