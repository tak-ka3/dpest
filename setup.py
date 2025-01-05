from setuptools import setup, find_packages

setup(
    name="dpest",                  # パッケージ名
    version="0.1.0",                    # バージョン
    description="dpest",  # 説明文
    author="tak-ka3",                 # 作者名
    url="https://github.com/tak-ka3/dpest",  # プロジェクトのURL
    packages=find_packages(),           # パッケージを自動検出
    install_requires=[                  # 依存パッケージ
        # 必要に応じて依存ライブラリを記載
        "numpy",
        "matplotlib",
        "scipy",
        "mmh3"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.11",            # 対応するPythonバージョン
)