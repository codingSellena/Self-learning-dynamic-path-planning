import PyInstaller.__main__
import os
import sys


def build_exe():
    # 获取项目根目录的绝对路径
    project_root = os.path.dirname(os.path.abspath(__file__))

    # 主程序文件路径
    main_file = os.path.join(project_root, 'main.py')

    # 数据文件和资源 - 确保这些目录确实存在
    data_dirs = ['core', 'handlers', 'ui', 'utils']
    datas = []
    for data_dir in data_dirs:
        src = os.path.join(project_root, data_dir)
        if os.path.exists(src):  # 检查目录是否存在
            datas.append((src, data_dir))

    # PyInstaller参数
    args = [
        main_file,  # 主程序文件
        '--name=GraphVisualization',  # 可执行文件名
        '--onefile',  # 打包成单个文件
        '--windowed',  # 使用GUI模式（不显示控制台窗口）
        # '--icon=path/to/icon.ico',  # 取消注释并指定图标路径
        '--clean',  # 清理临时文件
        '--noconfirm',  # 不确认覆盖
    ]

    # 添加数据文件
    for src, dst in datas:
        args.append(f'--add-data={src}{os.pathsep}{dst}')

    # 添加所有必要的隐式导入
    hidden_imports = [
        'numpy',
        'pandas',
        'matplotlib',
        'networkx',
        'Pillow',
        'pyinstaller',
        'tk',
        'scipy'
    ]
    for imp in hidden_imports:
        args.append(f'--hidden-import={imp}')

    print("Starting build process...")
    print("Command:", ' '.join(args))  # 打印完整命令便于调试
    PyInstaller.__main__.run(args)
    print("Build completed!")


if __name__ == '__main__':
    build_exe()
    input("Press Enter to exit...")
