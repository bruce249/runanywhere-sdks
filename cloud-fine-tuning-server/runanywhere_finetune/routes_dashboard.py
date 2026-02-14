"""
API Routes — Dashboard and server stats.
"""

from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from runanywhere_finetune.database import (
    JobStatus,
    TrainedAdapter,
    TrainingDataUpload,
    TrainingJob,
    get_session,
)
from runanywhere_finetune.schemas import ServerStatsResponse

logger = logging.getLogger(__name__)
router = APIRouter(tags=["dashboard"])


@router.get("/api/v1/stats", response_model=ServerStatsResponse)
async def get_server_stats(
    db: AsyncSession = Depends(get_session),
):
    """Get server-wide statistics."""

    # Total uploads
    result = await db.execute(select(func.count(TrainingDataUpload.id)))
    total_uploads = result.scalar_one_or_none() or 0

    # Total samples
    result = await db.execute(select(func.sum(TrainingDataUpload.sample_count)))
    total_samples = result.scalar_one_or_none() or 0

    # Jobs
    result = await db.execute(select(func.count(TrainingJob.id)))
    total_jobs = result.scalar_one_or_none() or 0

    result = await db.execute(
        select(func.count(TrainingJob.id)).where(
            TrainingJob.status.in_([JobStatus.PENDING, JobStatus.TRAINING, JobStatus.PREPROCESSING])
        )
    )
    active_jobs = result.scalar_one_or_none() or 0

    result = await db.execute(
        select(func.count(TrainingJob.id)).where(TrainingJob.status == JobStatus.COMPLETED)
    )
    completed_jobs = result.scalar_one_or_none() or 0

    result = await db.execute(
        select(func.count(TrainingJob.id)).where(TrainingJob.status == JobStatus.FAILED)
    )
    failed_jobs = result.scalar_one_or_none() or 0

    # Adapters
    result = await db.execute(select(func.count(TrainedAdapter.id)))
    total_adapters = result.scalar_one_or_none() or 0

    result = await db.execute(select(func.sum(TrainedAdapter.downloaded_count)))
    total_downloads = result.scalar_one_or_none() or 0

    # Unique devices
    result = await db.execute(select(TrainingDataUpload.device_id).distinct())
    devices = [row[0] for row in result.fetchall()]

    return ServerStatsResponse(
        total_uploads=total_uploads,
        total_samples=total_samples,
        total_jobs=total_jobs,
        active_jobs=active_jobs,
        completed_jobs=completed_jobs,
        failed_jobs=failed_jobs,
        total_adapters=total_adapters,
        total_adapter_downloads=total_downloads,
        devices=devices,
    )


# ─── Detail API endpoints for the dashboard ──────────────────────────────────


@router.get("/api/v1/uploads")
async def list_uploads(db: AsyncSession = Depends(get_session)):
    """List all data uploads with details."""
    stmt = select(TrainingDataUpload).order_by(TrainingDataUpload.uploaded_at.desc())
    result = await db.execute(stmt)
    uploads = result.scalars().all()
    return [
        {
            "id": u.id,
            "device_id": u.device_id,
            "model_id": u.model_id,
            "sample_count": u.sample_count,
            "uploaded_at": u.uploaded_at.isoformat() if u.uploaded_at else None,
            "metadata": json.loads(u.metadata_json) if u.metadata_json else {},
        }
        for u in uploads
    ]


@router.get("/api/v1/adapters-detail")
async def list_adapters_detail(db: AsyncSession = Depends(get_session)):
    """List all adapters with full details for the dashboard."""
    stmt = select(TrainedAdapter).order_by(TrainedAdapter.created_at.desc())
    result = await db.execute(stmt)
    adapters = result.scalars().all()
    return [
        {
            "id": a.id,
            "name": a.name,
            "device_id": a.device_id,
            "base_model": a.base_model,
            "format": a.format.value if a.format else "unknown",
            "file_size_bytes": a.file_size_bytes or 0,
            "lora_rank": a.lora_rank,
            "lora_alpha": a.lora_alpha,
            "training_samples": a.training_samples or 0,
            "final_loss": a.final_loss,
            "downloaded_count": a.downloaded_count or 0,
            "created_at": a.created_at.isoformat() if a.created_at else None,
        }
        for a in adapters
    ]


@router.get("/api/v1/jobs-detail")
async def list_jobs_detail(db: AsyncSession = Depends(get_session)):
    """List all training jobs with full details for the dashboard."""
    stmt = select(TrainingJob).order_by(TrainingJob.created_at.desc())
    result = await db.execute(stmt)
    jobs = result.scalars().all()
    return [
        {
            "id": j.id,
            "device_id": j.device_id,
            "model_id": j.model_id,
            "base_model": j.base_model,
            "status": j.status.value if j.status else "unknown",
            "progress": j.progress or 0.0,
            "current_epoch": j.current_epoch or 0,
            "total_epochs": j.total_epochs or 3,
            "current_loss": j.current_loss,
            "best_loss": j.best_loss,
            "error_message": j.error_message,
            "adapter_id": j.adapter_id,
            "started_at": j.started_at.isoformat() if j.started_at else None,
            "completed_at": j.completed_at.isoformat() if j.completed_at else None,
            "created_at": j.created_at.isoformat() if j.created_at else None,
        }
        for j in jobs
    ]


# ─── Dashboard HTML ──────────────────────────────────────────────────────────


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>RunAnywhere Fine-Tuning Server</title>
<style>
  :root {
    --bg: #0f1117; --surface: #161b22; --surface2: #1c2333;
    --border: #30363d; --text: #e6edf3; --text2: #8b949e;
    --accent: #58a6ff; --green: #3fb950; --red: #f85149;
    --orange: #d29922; --purple: #bc8cff;
  }
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif;
    background: var(--bg); color: var(--text); line-height: 1.5;
  }
  .header {
    background: var(--surface); border-bottom: 1px solid var(--border);
    padding: 16px 24px; display: flex; align-items: center; gap: 16px;
  }
  .header h1 { font-size: 20px; font-weight: 600; }
  .header .badge {
    background: var(--green); color: #000; font-size: 11px; font-weight: 600;
    padding: 2px 8px; border-radius: 12px;
  }
  .header .refresh-info { margin-left: auto; color: var(--text2); font-size: 12px; }
  .container { max-width: 1400px; margin: 0 auto; padding: 24px; }

  /* Stats Cards */
  .stats-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 16px; margin-bottom: 32px;
  }
  .stat-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 20px;
  }
  .stat-card .label { font-size: 12px; color: var(--text2); text-transform: uppercase; letter-spacing: 0.5px; }
  .stat-card .value { font-size: 32px; font-weight: 700; margin-top: 4px; }
  .stat-card .value.accent { color: var(--accent); }
  .stat-card .value.green { color: var(--green); }
  .stat-card .value.orange { color: var(--orange); }
  .stat-card .value.red { color: var(--red); }
  .stat-card .value.purple { color: var(--purple); }

  /* Devices */
  .devices-bar {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 16px 20px; margin-bottom: 32px;
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  .devices-bar .label { font-size: 13px; color: var(--text2); font-weight: 600; }
  .device-chip {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 16px; padding: 4px 12px; font-size: 12px; color: var(--accent);
    font-family: monospace;
  }

  /* Sections */
  .section { margin-bottom: 32px; }
  .section-header {
    display: flex; align-items: center; gap: 10px; margin-bottom: 16px;
  }
  .section-header h2 { font-size: 16px; font-weight: 600; }
  .section-header .count {
    background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 2px 10px; font-size: 12px; color: var(--text2);
  }

  /* Buttons */
  .btn {
    background: var(--accent); color: #000; border: none; border-radius: 6px;
    padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer;
    transition: opacity 0.2s;
  }
  .btn:hover { opacity: 0.85; }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; }
  .btn-sm { padding: 3px 10px; font-size: 11px; }
  .btn-red { background: var(--red); }
  .btn-green { background: var(--green); }

  /* Tables */
  .table-wrap {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; overflow: hidden;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  thead { background: var(--surface2); }
  th {
    text-align: left; padding: 10px 16px; font-weight: 600;
    color: var(--text2); font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.5px; border-bottom: 1px solid var(--border);
  }
  td { padding: 10px 16px; border-bottom: 1px solid var(--border); }
  tr:last-child td { border-bottom: none; }
  tr:hover { background: rgba(88, 166, 255, 0.04); }

  .mono { font-family: monospace; font-size: 12px; color: var(--accent); }
  .id-cell { max-width: 120px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }

  /* Status badges */
  .status {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-size: 11px; font-weight: 600; text-transform: uppercase;
  }
  .status.pending { background: rgba(210, 153, 34, 0.15); color: var(--orange); }
  .status.preprocessing { background: rgba(210, 153, 34, 0.15); color: var(--orange); }
  .status.training { background: rgba(88, 166, 255, 0.15); color: var(--accent); }
  .status.converting { background: rgba(188, 140, 255, 0.15); color: var(--purple); }
  .status.completed { background: rgba(63, 185, 80, 0.15); color: var(--green); }
  .status.failed { background: rgba(248, 81, 73, 0.15); color: var(--red); }
  .status.cancelled { background: rgba(139, 148, 158, 0.15); color: var(--text2); }

  /* Progress bar */
  .progress-bar {
    width: 100px; height: 6px; background: var(--surface2);
    border-radius: 3px; overflow: hidden; display: inline-block; vertical-align: middle;
  }
  .progress-bar .fill { height: 100%; background: var(--accent); border-radius: 3px; transition: width 0.3s; }
  .progress-text { font-size: 12px; color: var(--text2); margin-left: 6px; }

  /* File size */
  .size { color: var(--text2); }

  /* Empty state */
  .empty {
    text-align: center; padding: 40px; color: var(--text2); font-size: 14px;
  }
  .empty .icon { font-size: 32px; margin-bottom: 8px; }

  /* Live indicator */
  .live-dot {
    width: 8px; height: 8px; border-radius: 50%; background: var(--green);
    display: inline-block; animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Loss value */
  .loss { font-family: monospace; font-size: 12px; }
</style>
</head>
<body>

<div class="header">
  <h1>RunAnywhere Fine-Tuning Server</h1>
  <span class="badge">RUNNING</span>
  <div class="refresh-info">
    <span class="live-dot"></span>&nbsp;
    Auto-refresh every 5s &mdash; Last: <span id="lastRefresh">--</span>
  </div>
</div>

<div class="container">
  <!-- Stats -->
  <div class="stats-grid" id="statsGrid">
    <div class="stat-card"><div class="label">Uploads</div><div class="value accent" id="s-uploads">-</div></div>
    <div class="stat-card"><div class="label">Total Samples</div><div class="value accent" id="s-samples">-</div></div>
    <div class="stat-card"><div class="label">Active Jobs</div><div class="value orange" id="s-active">-</div></div>
    <div class="stat-card"><div class="label">Completed Jobs</div><div class="value green" id="s-completed">-</div></div>
    <div class="stat-card"><div class="label">Failed Jobs</div><div class="value red" id="s-failed">-</div></div>
    <div class="stat-card"><div class="label">Adapters</div><div class="value purple" id="s-adapters">-</div></div>
    <div class="stat-card"><div class="label">Downloads</div><div class="value accent" id="s-downloads">-</div></div>
    <div class="stat-card"><div class="label">Devices</div><div class="value green" id="s-devices">-</div></div>
  </div>

  <!-- Connected Devices -->
  <div class="devices-bar" id="devicesBar" style="display:none;">
    <span class="label">Connected Devices:</span>
    <span id="deviceChips"></span>
  </div>

  <!-- Training Jobs -->
  <div class="section">
    <div class="section-header">
      <h2>Training Jobs</h2>
      <span class="count" id="jobCount">0</span>
      <button class="btn btn-sm" id="retryAllBtn" style="display:none;" onclick="retryAllFailed()">Retry All Failed</button>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Job ID</th><th>Device</th><th>Base Model</th><th>Status</th>
            <th>Progress</th><th>Epoch</th><th>Loss</th><th>Best Loss</th>
            <th>Adapter</th><th>Created</th><th>Actions</th>
          </tr>
        </thead>
        <tbody id="jobsBody">
          <tr><td colspan="11" class="empty"><div class="icon">&#128640;</div>No training jobs yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Data Uploads -->
  <div class="section">
    <div class="section-header">
      <h2>Data Uploads</h2>
      <span class="count" id="uploadCount">0</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Upload ID</th><th>Device</th><th>Model</th>
            <th>Samples</th><th>Uploaded At</th>
          </tr>
        </thead>
        <tbody id="uploadsBody">
          <tr><td colspan="5" class="empty"><div class="icon">&#128229;</div>No uploads yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>

  <!-- Adapters -->
  <div class="section">
    <div class="section-header">
      <h2>Trained Adapters</h2>
      <span class="count" id="adapterCount">0</span>
    </div>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Adapter ID</th><th>Name</th><th>Device</th><th>Base Model</th>
            <th>Format</th><th>Size</th><th>Samples</th><th>Final Loss</th>
            <th>Downloads</th><th>Created</th>
          </tr>
        </thead>
        <tbody id="adaptersBody">
          <tr><td colspan="10" class="empty"><div class="icon">&#129520;</div>No adapters yet</td></tr>
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
const API = window.location.origin;

function fmtTime(iso) {
  if (!iso) return '—';
  const d = new Date(iso);
  return d.toLocaleString(undefined, { month:'short', day:'numeric', hour:'2-digit', minute:'2-digit', second:'2-digit' });
}

function fmtSize(b) {
  if (!b) return '—';
  if (b < 1024) return b + ' B';
  if (b < 1048576) return (b/1024).toFixed(1) + ' KB';
  return (b/1048576).toFixed(1) + ' MB';
}

function shortId(id) {
  return id ? id.substring(0, 8) : '—';
}

function statusBadge(s) {
  return '<span class="status ' + s + '">' + s + '</span>';
}

function progressBar(p) {
  const pct = Math.round((p || 0) * 100);
  return '<div class="progress-bar"><div class="fill" style="width:' + pct + '%"></div></div>'
       + '<span class="progress-text">' + pct + '%</span>';
}

function lossVal(v) {
  return v != null ? '<span class="loss">' + v.toFixed(4) + '</span>' : '—';
}

async function fetchJSON(path) {
  try {
    const r = await fetch(API + path);
    return await r.json();
  } catch(e) { console.error('Fetch error:', path, e); return null; }
}

async function refresh() {
  const [stats, jobs, uploads, adapters] = await Promise.all([
    fetchJSON('/api/v1/stats'),
    fetchJSON('/api/v1/jobs-detail'),
    fetchJSON('/api/v1/uploads'),
    fetchJSON('/api/v1/adapters-detail'),
  ]);

  // Stats
  if (stats) {
    document.getElementById('s-uploads').textContent = stats.total_uploads;
    document.getElementById('s-samples').textContent = stats.total_samples;
    document.getElementById('s-active').textContent = stats.active_jobs;
    document.getElementById('s-completed').textContent = stats.completed_jobs;
    document.getElementById('s-failed').textContent = stats.failed_jobs;
    document.getElementById('s-adapters').textContent = stats.total_adapters;
    document.getElementById('s-downloads').textContent = stats.total_adapter_downloads;
    document.getElementById('s-devices').textContent = (stats.devices||[]).length;

    const bar = document.getElementById('devicesBar');
    const chips = document.getElementById('deviceChips');
    if (stats.devices && stats.devices.length > 0) {
      bar.style.display = 'flex';
      chips.innerHTML = stats.devices.map(d => '<span class="device-chip">' + d + '</span>').join('');
    } else {
      bar.style.display = 'none';
    }
  }

  // Jobs
  if (jobs) {
    document.getElementById('jobCount').textContent = jobs.length;
    const hasFailed = jobs.some(j => j.status === 'failed');
    document.getElementById('retryAllBtn').style.display = hasFailed ? 'inline-block' : 'none';
    const tbody = document.getElementById('jobsBody');
    if (jobs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="11" class="empty"><div class="icon">&#128640;</div>No training jobs yet</td></tr>';
    } else {
      tbody.innerHTML = jobs.map(j => {
        const retryBtn = j.status === 'failed'
          ? '<button class="btn btn-sm" onclick="retryJob(\\''+j.id+'\\')">Retry</button>'
          : (j.status === 'pending' || j.status === 'training' || j.status === 'preprocessing'
            ? '<span class="status ' + j.status + '">...</span>' : '—');
        return '<tr>'
        + '<td class="id-cell mono" title="' + j.id + '">' + shortId(j.id) + '</td>'
        + '<td class="mono">' + shortId(j.device_id) + '</td>'
        + '<td>' + (j.base_model || j.model_id || '—') + '</td>'
        + '<td>' + statusBadge(j.status) + '</td>'
        + '<td>' + progressBar(j.progress) + '</td>'
        + '<td>' + j.current_epoch + '/' + j.total_epochs + '</td>'
        + '<td>' + lossVal(j.current_loss) + '</td>'
        + '<td>' + lossVal(j.best_loss) + '</td>'
        + '<td class="id-cell mono">' + shortId(j.adapter_id) + '</td>'
        + '<td>' + fmtTime(j.created_at) + '</td>'
        + '<td>' + retryBtn + '</td>'
        + '</tr>';
      }).join('');
    }
  }

  // Uploads
  if (uploads) {
    document.getElementById('uploadCount').textContent = uploads.length;
    const tbody = document.getElementById('uploadsBody');
    if (uploads.length === 0) {
      tbody.innerHTML = '<tr><td colspan="5" class="empty"><div class="icon">&#128229;</div>No uploads yet</td></tr>';
    } else {
      tbody.innerHTML = uploads.map(u => '<tr>'
        + '<td class="id-cell mono" title="' + u.id + '">' + shortId(u.id) + '</td>'
        + '<td class="mono">' + shortId(u.device_id) + '</td>'
        + '<td>' + (u.model_id || '—') + '</td>'
        + '<td><strong>' + u.sample_count + '</strong></td>'
        + '<td>' + fmtTime(u.uploaded_at) + '</td>'
        + '</tr>').join('');
    }
  }

  // Adapters
  if (adapters) {
    document.getElementById('adapterCount').textContent = adapters.length;
    const tbody = document.getElementById('adaptersBody');
    if (adapters.length === 0) {
      tbody.innerHTML = '<tr><td colspan="10" class="empty"><div class="icon">&#129520;</div>No adapters yet</td></tr>';
    } else {
      tbody.innerHTML = adapters.map(a => '<tr>'
        + '<td class="id-cell mono" title="' + a.id + '">' + shortId(a.id) + '</td>'
        + '<td>' + (a.name || '—') + '</td>'
        + '<td class="mono">' + shortId(a.device_id) + '</td>'
        + '<td>' + (a.base_model || '—') + '</td>'
        + '<td>' + (a.format || '—').toUpperCase() + '</td>'
        + '<td class="size">' + fmtSize(a.file_size_bytes) + '</td>'
        + '<td>' + a.training_samples + '</td>'
        + '<td>' + lossVal(a.final_loss) + '</td>'
        + '<td>' + (a.downloaded_count || 0) + '</td>'
        + '<td>' + fmtTime(a.created_at) + '</td>'
        + '</tr>').join('');
    }
  }

  document.getElementById('lastRefresh').textContent = new Date().toLocaleTimeString();
}

refresh();
setInterval(refresh, 5000);

async function retryJob(jobId) {
  try {
    const r = await fetch(API + '/api/v1/training/retry/' + jobId, { method: 'POST' });
    if (r.ok) { refresh(); } else { alert('Retry failed: ' + (await r.text())); }
  } catch(e) { alert('Retry error: ' + e); }
}

async function retryAllFailed() {
  const btn = document.getElementById('retryAllBtn');
  btn.disabled = true; btn.textContent = 'Retrying...';
  try {
    const r = await fetch(API + '/api/v1/training/retry-all', { method: 'POST' });
    const data = await r.json();
    btn.textContent = 'Retried ' + data.count;
    setTimeout(() => { btn.textContent = 'Retry All Failed'; btn.disabled = false; }, 3000);
    refresh();
  } catch(e) { alert('Retry error: ' + e); btn.disabled = false; btn.textContent = 'Retry All Failed'; }
}
</script>
</body>
</html>"""


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Live dashboard showing uploads, training jobs, and adapters."""
    return DASHBOARD_HTML


@router.get("/health")
async def health():
    """Health check."""
    return {"status": "ok", "service": "runanywhere-finetune-server"}
