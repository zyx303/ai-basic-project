"""
PyInstaller打包脚本
使用此脚本来构建可执行文件
"""

import os
import subprocess
import sys

def build_exe():
    """构建可执行文件"""
    try:
        # 确保在正确的目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        os.chdir(script_dir)
        
        # 首先尝试使用spec文件
        if os.path.exists('window.spec'):
            print("使用spec文件构建...")
            cmd = ['pyinstaller', 'window.spec']
        else:
            print("直接构建...")
            cmd = [
                'pyinstaller',
                '--onedir',  # 创建单目录分发
                '--windowed',  # 不显示控制台
                '--name=SentimentAnalyzer',
                '--add-data=model;model',
                '--add-data=utils;utils',
                '--hidden-import=model.SentimentClassifier',
                '--hidden-import=utils.data_loader',
                '--hidden-import=utils.train_utils',
                '--hidden-import=transformers',
                '--hidden-import=sklearn.metrics',
                '--hidden-import=sklearn.preprocessing',
                'window.py'
            ]
        
        # 执行构建命令
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("构建成功！")
            print("可执行文件位于: dist/SentimentAnalyzer/")
        else:
            print("构建失败！")
            print("错误信息:")
            print(result.stderr)
            
    except Exception as e:
        print(f"构建过程出错: {str(e)}")

if __name__ == "__main__":
    build_exe()
