# 无人机集群视觉感知系统 - PowerShell 启动脚本

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "🚁 无人机集群视觉感知系统" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

# 检查虚拟环境
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "正在创建虚拟环境..." -ForegroundColor Yellow
    python -m venv venv
}

# 激活虚拟环境
& .\venv\Scripts\Activate.ps1

# 运行主程序
python main.py @args

# 保持窗口打开（如果出错）
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "程序异常退出，按任意键关闭..." -ForegroundColor Red
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
}
