@echo off
chcp 65001 >nul
REM 无人机集群视觉感知系统 - 启动脚本

echo ============================================
echo 🚁 无人机集群视觉感知系统
echo ============================================
echo.

REM 检查虚拟环境
if not exist "venv\Scripts\activate.bat" (
    echo 正在创建虚拟环境...
    python -m venv venv
)

REM 激活虚拟环境
call venv\Scripts\activate.bat

REM 运行主程序
python main.py %*

REM 保持窗口打开（如果出错）
if errorlevel 1 (
    echo.
    echo 程序异常退出，按任意键关闭...
    pause >nul
)
