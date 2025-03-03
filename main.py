from dotenv import load_dotenv
load_dotenv()

import os
from huggingface_hub import hf_hub_download, HfApi
from utils import obs, ufile, utime
from utils.logger import logger

# HF仓库信息
repo_id = "MLCommons/unsupervised_peoples_speech"
repo_type = "dataset"
revision = "main"
hf_token = os.getenv("HF_TOKEN")
assert hf_token
logger.info(f"[Huggingface Info] repo_id:{repo_id}, repo_type:{repo_type}, revision:{revision}, hf_token:{hf_token}")

def hf_list_files(repo_id, revision="main"):
    """
    通过Huggingface Hub API获取HF仓库下的所有文件

    Args:
        repo_id (str): 仓库ID
        revision (str, optional): 仓库分支. Defaults to "main".

    Returns:
        list: 仓库下所有文件
    """
    api = HfApi()
    # repo_id = "MLCommons/unsupervised_peoples_speech"
    files = api.list_repo_files(repo_id=repo_id, revision=revision, repo_type="dataset", token=hf_token)
    # print(file_names)
    print(f"hf_list_files > 检测到{repo_id}/tree/{revision}下一共{len(files)}个文件")
    logger.info(f"hf_list_files > 检测到{repo_id}/tree/{revision}下一共{len(files)}个文件")
    return files

def hf_batch_download_concurrent(file_list:list, repo_id:str, download_file_path:str, repo_type="dataset", revision="main", batch_size=10, n_jobs=4):
    """
    分批下载文件，以提高速度和效率。
    :param file_list: 文件名列表
    :param batch_size: 每次下载的文件数量
    :param n_jobs: 并发下载的线程数
    """
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = []
        for i in range(0, len(file_list), batch_size):
            batch = file_list[i:i+batch_size]
            future = executor.submit(_hf_download_handler, batch, repo_id, repo_type, revision, download_file_path)
            futures.append(future)
        for future in futures:
            future.result()
            print(f"第 {futures.index(future) + 1} 批文件下载完成")

def _hf_download_handler(file_list, repo_id, repo_type, revision, download_file_path):
    """
    hf_batch_download_concurrent的handler
    :param file_list: 文件名列表
    :param repo_id: 仓库ID
    :param repo_type: 仓库类型
    :param revision: 仓库分支
    :param download_file_path: 文件下载路径
    """
    for file in file_list:
        try:
            print(f"当前正在下载: {file} \n")
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
        except Exception as e:
            print(f"[FAILED] {file} 下载失败, 错误: {e}")
            logger.error(f"{repo_id}.{revision} -> {download_file_path}/{file}下载失败, 错误: {e}")
            # ufile.write_string_to_file(text_string="", out_file=f"{download_index_path}/{file}.fail")
            continue
        else:
            print(f"[SUCCESS] {file} 下载成功 \n")
        finally:
            utime.random_sleep(1, 2)

def read_files_list_from_file(index_file:str)->list:
    """
    Reads a list of file names from a specified index file.

    Args:
        index_file (str): The path to the index file containing file names.

    Returns:
        list: A list of file names read from the index file, with each file name stripped of leading and trailing whitespace.
    """
    ret_list = []
    with open(index_file, "r", encoding="utf8") as f:
        lines = f.readlines()
        ret_list = [line.strip() for line in lines]
    print(f"read_files_list_from_file > 一共读取到{len(ret_list)}个文件")
    logger.info(f"read_files_list_from_file > 一共读取到{len(ret_list)}个文件")
    return ret_list

if __name__ == "__main__":
    # 初始化
    download_index_path = "./downloads/indexs"
    os.makedirs(download_index_path, exist_ok=True)
    download_file_path = "./downloads/origins"
    os.makedirs(download_file_path, exist_ok=True)
    logger.info(f"download_index_path:{download_index_path}, download_file_path:{download_file_path}")

    # 所有文件列表写入downloads\indexs\unsupervised_peoples_speech.txt
    # target_files = hf_list_files(repo_id=repo_id)
    # index_file = os.path.join(download_index_path, repo_id.split(r"/")[-1]+".txt") 
    # for file in target_files:
    #     ufile.add_string_to_file(text_string=file, out_file=index_file)

    # download audio/000001.tar
    # file = "audio/000001.tar"
    # hf_hub_download(
    #     repo_id=repo_id,
    #     repo_type=repo_type,
    #     revision=revision,
    #     filename=file, 
    #     local_dir=download_file_path,
    #     cache_dir="./cache",
    #     local_files_only=False,
    #     force_download=False,
    # )

    # download batch files
    # files = read_files_list_from_file(r"downloads\indexs\unsupervised_peoples_speech_audio_b1.txt")
    # hf_batch_download_concurrent(
    #     file_list=hf_list_files(repo_id=repo_id),
    #     download_file_path=download_file_path,
    #     repo_id=repo_id,
    #     repo_type=repo_type,
    #     revision=revision,
    # )
