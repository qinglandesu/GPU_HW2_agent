import argparse
from modelscope import HubApi
from modelscope import snapshot_download

def parse_args():
    parser = argparse.ArgumentParser(description='Download model parameters settings')
    parser.add_argument('--model_name', type=str, default='qinglandesu/GPUclass_qwen0', 
                       help='Model name in format: organization/model_name')
    parser.add_argument('--cache_dir', type=str, default='./', 
                       help='Directory to save the model')
    parser.add_argument('--revision', type=str, default='master', 
                       help='Model revision/version')
    return parser.parse_args()

if __name__ == "__main__":
    api = HubApi()
    api.login('ms-a0ddd846-25ce-4e78-bd7a-68d389153978')
    sh_args = parse_args()
    model_dir = snapshot_download(
        sh_args.model_name,
        cache_dir=sh_args.cache_dir,
        revision=sh_args.revision
    )
    print("Model download successful!")
    print(f"Model saved at: {model_dir}")
