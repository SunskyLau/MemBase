"""进入保留官方实现的服务连接适配器。"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from membase.runners.native_transport import main

if __name__ == "__main__":
    main()
