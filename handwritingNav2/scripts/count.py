import os

def count_pngs(root_dir: str) -> int:
    """递归统计 root_dir 及其子目录中的 PNG 文件数量（大小写不敏感）"""
    count = 0
    for dirpath, _, filenames in os.walk(root_dir):
        count += sum(
            1 for fname in filenames
            if fname.lower().endswith('.png')
        )
    return count

if __name__ == "__main__":
    folder = "/data/xhj/handwritingNav/data/mp3d_hwnav/train/handwriting_instr"
    print(f"PNG 图片总数：{count_pngs(folder)}")
