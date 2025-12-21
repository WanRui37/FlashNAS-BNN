import os
import shutil

target_folder_all = (
    "/mnt/wr/Train_Quant_Imagenet100_184_channel/outputs/"  # 指定要检查的文件夹路径，可以根据实际情况修改
)
checkpoint_folder_name = "checkpoint"

output_folder = "/mnt/wr/Train_Quant_Imagenet100_184_channel/outputs/"

# for target_folder in os.listdir(target_folder_all):
#     if os.path.isfile(target_folder_all + target_folder):
#         print(f"{target_folder} is file!")
#         continue
#     if checkpoint_folder_name not in os.listdir(target_folder_all + target_folder):
#         print(f"{target_folder} don't have checkpoint!")
#         # os.rmdir(target_folder_all + target_folder)
#         shutil.rmtree(target_folder_all + target_folder)
#     else:
#         print(f"{target_folder} have checkpoint!")

# exit(0)


for folder_name in os.listdir(output_folder):
    if folder_name.startswith("BNN"):
        log_file_path = os.path.join(output_folder, folder_name, "log.txt")
        if os.path.exists(log_file_path):
            with open(log_file_path, "r") as f:
                content = f.read()
                if "Best Acc:" not in content:
                    print(f"Error: {output_folder + folder_name}")
                    # os.rmdir(os.path.join(output_folder, folder_name))
                    shutil.rmtree(os.path.join(output_folder, folder_name))
                else:
                    print(f"True: {output_folder + folder_name}")
