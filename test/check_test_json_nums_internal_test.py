import json
import os
import sys

def check_json_count(file_path):
    """
    读取指定的 JSON 文件并返回其中包含的数据条数。
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"❌ 错误：找不到文件 '{file_path}'")
        return

    try:
        # 读取 JSON 文件
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 判断数据类型并输出数量
        if isinstance(data, list):
            count = len(data)
            print(f"✅ 文件 '{os.path.basename(file_path)}' 读取成功！")
            print(f"📊 当前文件共有 【{count}】 条数据。")
        elif isinstance(data, dict):
            # 如果最外层是字典，通常算作 1 个大对象，或者根据具体 key 来算
            print(f"✅ 文件 '{os.path.basename(file_path)}' 读取成功！")
            print("📊 该 JSON 的最外层是一个字典 (Dict)，通常代表 1 条总数据或配置信息。")
            print(f"   包含的顶级键 (Keys) 数量: {len(data.keys())}")
        else:
            print(f"⚠️ 文件格式无法统计长度，类型为: {type(data)}")

    except json.JSONDecodeError as e:
        print(f"❌ 解析错误：'{file_path}' 不是有效的 JSON 文件。\n错误详情: {e}")
    except Exception as e:
        print(f"❌ 发生未知错误：{e}")

if __name__ == "__main__":
    # ================= 使用说明 =================
    # 方式一：直接在代码里修改你要检查的文件路径

    # 自动定位项目根目录 (假设此脚本放在 utils/ 文件夹下)
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(current_script_path))

    # 设定你要检查的目标文件 (例如你要查 public 的测试集)
    target_file = os.path.join(project_root, 'split_data', 'dev_CRMUS_MU_internal_test.json')

    print("-" * 40)
    check_json_count(target_file)
    print("-" * 40)

    # -------------------------------------------
    # 方式二：通过命令行参数传入路径 (可选高级用法)
    # 运行示例: python utils/check_json_count.py data/test_CRMUS_MU_public.json
    if len(sys.argv) > 1:
        custom_file = sys.argv[1]
        print("\n[检测到命令行参数，正在检查指定文件...]")
        check_json_count(custom_file)
        print("-" * 40)