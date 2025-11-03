# =========================
# Nightly Pipeline Runner
# =========================
# 순서:
#  1) economic_indicators.py
#  2) news_indicators.py
#  3) preprocess.py
#  4) predict_next_day.py
#
# - 평일만 실행(토/일 스킵), 휴장일 CSV 있으면 추가 스킵
# - .venv 우선(여러 후보 경로 탐색), 없으면 시스템 python 사용
# - 중복 실행 방지용 잠금 파일 + 오래된 락 자동 제거
# - 로그: <repo_root>\logs\nightly_YYYY-MM-DD_HH-mm-ss.log

# ----- 출력 인코딩(한글 깨짐 완화) -----
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$ErrorActionPreference = "Stop"

# ----- 경로 설정 -----
if (-not $PSScriptRoot) { $PSScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
$PIPE_DIR  = $PSScriptRoot
$REPO_ROOT = Split-Path -Parent (Split-Path -Parent $PIPE_DIR)   # ...\Capstone_Design 형태를 가정
Set-Location $PIPE_DIR

# ----- 로그 준비 -----
$LOG_DIR = Join-Path $REPO_ROOT "logs"
New-Item -ItemType Directory -Force -Path $LOG_DIR | Out-Null
$STAMP = Get-Date -Format "yyyy-MM-dd_HH-mm-ss"
$LOG   = Join-Path $LOG_DIR "nightly_$STAMP.log"

function Log([string]$msg) {
  $ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
  "$ts $msg" | Tee-Object -FilePath $LOG -Append
}

Log "=== Nightly pipeline start ==="
Log "PIPE_DIR  = $PIPE_DIR"
Log "REPO_ROOT = $REPO_ROOT"

# ----- Python 해석기 결정 (.venv 여러 후보 + 폴백) -----
# 주로 쓰는 위치들을 전부 후보로 잡아 탐색 (상위 vscode\.venv 포함)
$CANDIDATES = @(
  (Join-Path $REPO_ROOT ".venv\Scripts\python.exe"),
  (Join-Path (Split-Path -Parent $REPO_ROOT) ".venv\Scripts\python.exe"),
  (Join-Path $PIPE_DIR ".venv\Scripts\python.exe"),
  (Join-Path (Split-Path -Parent $PIPE_DIR) ".venv\Scripts\python.exe")
)

$PY = $null
foreach ($p in $CANDIDATES) {
  if (Test-Path $p) { $PY = $p; break }
}

if (-not $PY) {
  $PY = "python"
  Log "[WARN] .venv python not found in candidates. Falling back to system python."
} else {
  Log "[INFO] Using venv: $PY"
}

# ----- 주말/휴장일 스킵 -----
$todayKST = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId((Get-Date), "Korea Standard Time").Date
$dayOfWeek = (Get-Date $todayKST).DayOfWeek  # Sunday .. Saturday
if ($dayOfWeek -eq 'Saturday' -or $dayOfWeek -eq 'Sunday') {
  Log "[SKIP] 주말이므로 실행하지 않습니다. ($($todayKST.ToString('yyyy-MM-dd')))"
  exit 0
}

# 선택: 휴장일 CSV (YYYY-MM-DD 한 줄씩)
$HOLI_CSV = Join-Path $REPO_ROOT "data_analyze\economic_indicator\krx_holidays.csv"
if (Test-Path $HOLI_CSV) {
  try {
    $holi = Get-Content $HOLI_CSV | Where-Object { $_ -match '^\d{4}-\d{2}-\d{2}$' }
    if ($holi -contains ($todayKST.ToString("yyyy-MM-dd"))) {
      Log "[SKIP] 휴장일 목록에 포함되어 스킵합니다. ($($todayKST.ToString('yyyy-MM-dd')))"
      exit 0
    }
  } catch {
    Log "[WARN] 휴장일 CSV 확인 중 예외: $($_.Exception.Message)"
  }
}

# ----- 중복 실행 방지(잠금 파일 + stale 정리) -----
$LOCK = Join-Path $PIPE_DIR ".nightly.lock"
if (Test-Path $LOCK) {
  $age = (Get-Date) - (Get-Item $LOCK).LastWriteTime
  if ($age.TotalHours -gt 6) {
    Log "[WARN] Stale lock detected (~$([int]$age.TotalHours)h). Removing old lock: $LOCK"
    Remove-Item $LOCK -Force
  } else {
    Log "[SKIP] 잠금 파일 감지: 이전 실행이 종료되지 않았거나 중복 실행입니다. ($LOCK)"
    exit 0
  }
}
New-Item -ItemType File -Path $LOCK -Force | Out-Null

# ----- 공용 실행 함수 -----
function Run_Step([string]$title, [string]$cmd, [string[]]$args) {
  Log "▶ $title 시작: $cmd $($args -join ' ')"
  & $cmd @args 2>&1 | Tee-Object -FilePath $LOG -Append
  if ($LASTEXITCODE -ne 0) {
    Log "❌ $title 실패 (exit=$LASTEXITCODE). 파이프라인 중단."
    throw "$title failed"
  }
  Log "✅ $title 완료"
}
# (호환용) 기존 Run-Step 호출이 남아 있어도 동작하도록 별칭
Set-Item -Path Function:\Run-Step -Value ${function:Run_Step} -Force

# ----- 메인 실행 -----
try {
  # 1) indicators
  Run_Step "economic_indicators" $PY @("economic_indicators.py")
  Run_Step "news_indicators"     $PY @("news_indicators.py")

  # 2) 전처리
  Run_Step "preprocess"          $PY @("preprocess.py")

  # 3) 예측
  Run_Step "predict_next_day"    $PY @("predict_next_day.py")

  Log "🎉 Nightly pipeline ALL DONE."
  exit 0
}
catch {
  Log "[FATAL] 파이프라인 실패: $($_.Exception.Message)"
  exit 1
}
finally {
  if (Test-Path $LOCK) { Remove-Item $LOCK -Force }
  Log "=== Nightly pipeline end ==="
}
