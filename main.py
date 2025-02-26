from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import hf_hub_download, HfApi
from utils import obs, ufile

hf_token = os.getenv("HF_TOKEN")
assert not hf_token

# 列举hf文件
def hf_list_files(repo_id, revision="main"):
    api = HfApi()
    # repo_id = "MLCommons/unsupervised_peoples_speech"
    files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="dataset", token=hf_token)
    # print(file_names)
    print(f"hf_list_files > 检测到{repo_id}/tree/{revision}下一共{len(files)}个文件")
    return files

def hf_download_batch(file_list, batch_size=100):
    """
    分批下载文件。
    :param file_list: 文件名列表
    :param batch_size: 每次下载的文件数量
    """
    for i in range(0, len(file_list), batch_size):
        batch = file_list[i:i+batch_size]
        for file in batch:
            try:
                print(f"正在下载: {file}")
                hf_hub_download(repo_id="MLCommons/unsupervised_peoples_speech", 
                                filename=file, 
                                revision="main", 
                                local_files_only=False)
            except Exception as e:
                print(f"下载失败: {file}, 错误: {e}")
        print(f"第 {i//batch_size + 1} 批文件下载完成")


if __name__ == "__main__":
    download_index_path = "./downloads/indexs"
    os.makedirs(download_index_path, exist_ok=True)
    download_file_path = "./downloads/origins"
    os.makedirs(download_file_path, exist_ok=True)

    repo_id = "MLCommons/unsupervised_peoples_speech"
    repo_type = "dataset"
    revision = "main"

    # 所有文件列表写入downloads\indexs\unsupervised_peoples_speech.txt
    # target_files = hf_list_files(repo_id=repo_id)
    # for file in target_files:
    #     ufile.add_string_to_file(text_string=file, out_file=f"{download_index_path}/{repo_id}.txt")

    # audio/000001.tar
    file = "audio/000001.tar"
    hf_hub_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        filename=file, 
        local_dir=download_file_path,
        cache_dir="./cache",
        local_files_only=False,
        force_download=False,
    )

    # hf_download_batch(file_list=target_files)
