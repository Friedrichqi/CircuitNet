import os
import subprocess
for iters in range(160000, 200001, 10000):
    print(f"Evaluating model_iters_{iters}.pth")

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # save results to a file
    with open(f"./work_dir/congestion_gpdl_sft/results_{iters}.txt", "w") as f:
        subprocess.run(f"python test.py --pretrained ./work_dir/congestion_gpdl_sft/model_iters_{iters}.pth", shell=True, stdout=f)