# utils.py: utility functions 
import os
import torch
import torch.nn.functional as F

def get_path_to_task_files(data_root, task_path, ok_to_download=True, xt=None, fn_xt_config=None):

    if task_path.endswith("/"):
        # remove ending slash
        task_path = task_path[:-1]

    dataset_name, dataset_version, task_name = task_path.split("/")

    rel_task_path = "{}/{}/{}".format(dataset_name, dataset_version, task_name)
    task_path = "{}/{}".format(data_root, rel_task_path)
    task_path = os.path.abspath(task_path)

    if os.path.exists(task_path):
        print("using LOCAL dataset/task files found at: {}".format(task_path))

    else:
        print("dataset/task files not found on local machine: {}".format(task_path))

        if not ok_to_download:
            raise Exception("download of dataset/task files is disabled")

        # dataset doesn't exist locally; use XT to download it from Azure Storage
        close_xt = False

        if xt is None:
            from xtlib.run import Run
            if fn_xt_config:
                # Run() needs to know where global config is located and we 
                # don't want to reply on the IDE setting this env var
                os.environ["XT_GLOBAL_CONFIG"] = fn_xt_config

            xt = Run()
            close_xt = True

        store_path = "$data/{}".format(rel_task_path)
        local_dest = task_path
        store_name = xt.config.get("store") if xt.config is not None else "<default store>"

        print("DOWNLOADING dataset from Azure Storage: store: {}, path: {}".format(store_name, store_path))

        xt.download_files_from_share(None, store_path + "/*.xy", local_dest, show_feedback=False, snapshot=False)
        xt.download_files_from_share(None, store_path + "/*.indexes_pt", local_dest, show_feedback=False, snapshot=False)
        xt.download_files_from_share(None, store_path + "/*vocab.json", local_dest, show_feedback=False, snapshot=False)

        if close_xt:
            xt.close()

    return task_path

def pashamax(x : torch.Tensor, dim : int, eps : float = 1e-6):
    unnormed_logits = F.relu(x.exp()-1)
    return unnormed_logits / (unnormed_logits.sum(dim=dim, keepdim=True)+eps)