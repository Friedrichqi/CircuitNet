import os
import subprocess
for iters in range(10000, 400001, 10000):
    print(f"Evaluating model_iters_{iters}.pth")

    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    # save results to a file
    for suffix in ['_deep', '_squeeze', '_doubleUNet']:
        try:
            folder_name = os.path.join("./work_dir", f"congestion_gpdl{suffix}_further2")
            with open(os.path.join(folder_name, f"results_{iters}.txt"), "w") as f:
                subprocess.run(f"python test_newmodels.py --pretrained {os.path.join(folder_name, f"model_iters_{iters}.pth")} --save_path {folder_name} --model_type {"GPDL"+suffix}", shell=True, stdout=f)
        except:
            pass
    