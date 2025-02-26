import os
from huggingface_hub import hf_hub_download, HfApi
from utils import obs, ufile

if __name__ == "__main__":
    # hf_list_files(repo_id="MLCommons/unsupervised_peoples_speech")
    # obs.is_exist("/QUWAN_DATA/temp_data/installer/clash-for-windows.zip")

    download_index_path = "./downloads/indexs"
    download_file_path = "./downloads/origins"
    # 测试文件写入、追加
    # ufile.write_string_to_file(text_string="hello world", out_file=f"{download_index_path}/test.txt")
    # ufile.add_string_to_file(text_string="hello world2222", out_file=f"{download_index_path}/test.txt")

    # 测试空白文件创建
    ufile.write_string_to_file(text_string="", out_file=f"{download_index_path}/test_empty_file.fail")