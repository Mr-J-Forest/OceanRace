// @ts-nocheck
import { ref, computed, watch, nextTick, onUnmounted } from 'vue'
import { api } from '../api/client'
import {
  renderAnomalyMonitorChart,
  renderAnomalyOceanMaps,
  renderAnomalyRiskMapChart,
  renderAnomalyTimelineChart,
  renderAnomalyWindowChart,
  resizeAnomalyInspectCharts
} from './anomalyInspectCharts'

const anomalyLabelsPath = ref('outputs/anomaly_detection/labels_competition.json')
const anomalyEventsPath = ref('outputs/anomaly_detection/events_competition.json')
const anomalyManifestPath = ref('data/processed/splits/anomaly_detection_competition.json')
const anomalyProcessedPath = ref('data/processed/anomaly_detection/path.txt')
const anomalySplit = ref('test')
const anomalyLoading = ref(false)
const anomalyError = ref('')
const anomalyData = ref(null)
const anomalyView = ref('monitor')
const anomalyRiskScore = ref(0)
const anomalyRiskLevel = ref('低')
const anomalyRiskClass = ref('text-emerald-300 border-emerald-400/40 bg-emerald-400/10')
const anomalySnapshot = ref(null)
const anomalySnapshotLoading = ref(false)
const anomalySnapshotError = ref('')
const anomalyRecentWindow = ref([])
const anomalyRecentWeek = ref([])
const anomalySelectedSnapshotIndex = ref(-1)
const anomalyLatestTimeText = ref('-')
const anomalySelectedTimeText = ref('-')
const anomalySelectedIndexText = ref('-')
const anomalyTracebackRows = ref([])
const anomalyTracebackWindowHours = ref(72)
const anomalyTracebackStartHour = ref(0)
const anomalyTracebackRangeText = ref('-')
const anomalyDataTimeText = ref('-')
const anomalyIssueTimeText = ref('-')
const anomalyCouplings = ref([])
const anomalyCases = ref([])
const anomalyBrief = ref('')
const anomalyWarning = ref({
  level: '蓝',
  targets: '值班中心/港口/海上作业单位',
  actions: [] as string[]
})
const anomalyWarningLogs = ref<
  { id: string; time?: string; issuedAt?: string; dataTime?: string; action: string; note: string }[]
>([])
const anomalyMonitor = ref({
  windNow: 0,
  waveNow: 0,
  windBand: '5.0-10.5',
  waveBand: '0.8-2.2',
  status: '平稳'
})

/** 与旧版 `anomalyProduct` 形状兼容，供管理驾驶舱等模块读取。 */
const anomalyProduct = computed(() => {
  const rs = Number(anomalyRiskScore.value || 0)
  const levelOf = (score: number) => {
    if (score >= 80) return { name: '极高', cls: 'text-rose-400' }
    if (score >= 60) return { name: '高', cls: 'text-orange-400' }
    if (score >= 35) return { name: '中', cls: 'text-amber-400' }
    return { name: '低', cls: 'text-sky-400' }
  }
  const zonesRaw = [
    { name: '渤海湾', base: 28 },
    { name: '黄海北部', base: 36 },
    { name: '黄海中部', base: 42 },
    { name: '黄海南部', base: 47 }
  ].map((z, idx) => {
    const score = Math.min(99, Math.max(8, z.base + Math.round(rs * 0.55) + idx * 4 - 8))
    const lv = levelOf(score)
    return { name: z.name, score, level: lv.name, levelClass: lv.cls }
  })
  return {
    riskScore: rs,
    riskLevel: anomalyRiskLevel.value,
    warning: {
      level: anomalyWarning.value.level,
      levelClass: '',
      targets: anomalyWarning.value.targets,
      actions: anomalyWarning.value.actions
    },
    monitor: {
      windNow: String(anomalyMonitor.value.windNow),
      waveNow: String(anomalyMonitor.value.waveNow),
      windBand: anomalyMonitor.value.windBand,
      waveBand: anomalyMonitor.value.waveBand,
      statusText: anomalyMonitor.value.status,
      statusClass: anomalyRiskClass.value,
      baselineMode: '季节+海域+气候分型',
      baselines: []
    },
    detect: { accuracy: 0, autoEvents: 0, retroHits: 0, archiveCount: 0, details: [] },
    typhoon: { zones: zonesRaw, couplings: [], historyCases: [] }
  }
})

let anomalySnapshotRequestSeq = 0
let anomalySnapshotPrefetchToken = 0
const anomalySnapshotLocalCache = new Map()
const anomalySnapshotInflight = new Map()

const ANOMALY_LOOKBACK_HOURS = 168
const ANOMALY_MONITOR_WINDOW_HOURS = 24

const anomalyTracebackMaxStartHour = computed(() => {
  const maxStart = ANOMALY_LOOKBACK_HOURS - Number(anomalyTracebackWindowHours.value || 72)
  return Math.max(0, maxStart)
})

const anomalyTracebackStepHours = computed(() => {
  const week = Array.isArray(anomalyRecentWeek.value) ? anomalyRecentWeek.value : []
  if (week.length < 2) return 1
  const deltas = []
  for (let i = 1; i < week.length; i++) {
    const dt = Number(week[i].timestamp) - Number(week[i - 1].timestamp)
    if (Number.isFinite(dt) && dt > 0) {
      deltas.push(Math.round(dt / 3600))
    }
  }
  if (!deltas.length) return 1
  deltas.sort((a, b) => a - b)
  const mid = Math.floor(deltas.length / 2)
  const median = deltas[mid]
  return Math.max(1, Number.isFinite(median) ? median : 1)
})

const _formatEpochText = (ts) => {
  const t = Number(ts)
  if (!Number.isFinite(t) || t <= 0) return '-'
  return new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19)
}

const _buildRecentWindow = (payload) => {
  const source = Array.isArray(payload?.recent_window) ? payload.recent_window : []
  if (!source.length) {
    return []
  }
  return source
    .map((row) => {
      const hits = Array.isArray(row?.event_hits) ? row.event_hits : []
      const windMean = Number(row?.wind_mean)
      const waveMean = Number(row?.wave_mean)
      const windP95 = Number(row?.wind_p95)
      const waveP95 = Number(row?.wave_p95)
      return {
        index: Number.isFinite(Number(row?.index)) ? Number(row.index) : -1,
        timestamp: Number.isFinite(Number(row?.timestamp)) ? Number(row.timestamp) : -1,
        label: Number.isFinite(Number(row?.label)) ? Number(row.label) : 0,
        matched: !!row?.matched,
        eventHits: hits,
        windMean: Number.isFinite(windMean) ? windMean : null,
        waveMean: Number.isFinite(waveMean) ? waveMean : null,
        windP95: Number.isFinite(windP95) ? windP95 : null,
        waveP95: Number.isFinite(waveP95) ? waveP95 : null,
        time: _formatEpochText(row?.timestamp)
      }
    })
    .filter((row) => row.index >= 0)
}

const recomputeAnomalyTracebackRows = () => {
  const week = Array.isArray(anomalyRecentWeek.value) ? anomalyRecentWeek.value : []
  if (!week.length) {
    anomalyTracebackRows.value = []
    anomalyTracebackRangeText.value = '-'
    return
  }

  const latestTs = Number(week[week.length - 1]?.timestamp || -1)
  const earliestTs = Number(week[0]?.timestamp || -1)
  if (!Number.isFinite(latestTs) || !Number.isFinite(earliestTs) || latestTs <= 0 || earliestTs <= 0) {
    anomalyTracebackRows.value = []
    anomalyTracebackRangeText.value = '-'
    return
  }

  const windowSec = Math.max(1, Number(anomalyTracebackWindowHours.value || 72)) * 3600
  const stepHours = Math.max(1, Number(anomalyTracebackStepHours.value || 1))
  const snappedStartHour = Math.round(Number(anomalyTracebackStartHour.value || 0) / stepHours) * stepHours
  if (snappedStartHour !== Number(anomalyTracebackStartHour.value || 0)) {
    anomalyTracebackStartHour.value = snappedStartHour
  }
  const startOffsetSec = Math.max(0, snappedStartHour) * 3600
  const selectedStart = earliestTs + startOffsetSec
  const selectedEnd = Math.min(latestTs, selectedStart + windowSec)

  anomalyTracebackRangeText.value = `${_formatEpochText(selectedStart)} ~ ${_formatEpochText(selectedEnd)}`

  const validWind = week.map((r) => Number(r.windMean)).filter((v) => Number.isFinite(v))
  const validWave = week.map((r) => Number(r.waveMean)).filter((v) => Number.isFinite(v))
  const mean = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0)
  const std = (arr, m) => {
    if (!arr.length) return 1
    const v = arr.reduce((s, x) => s + (x - m) * (x - m), 0) / arr.length
    return Math.sqrt(Math.max(v, 1e-6))
  }
  const windMeanBase = mean(validWind)
  const waveMeanBase = mean(validWave)
  const windStdBase = std(validWind, windMeanBase)
  const waveStdBase = std(validWave, waveMeanBase)

  const windowRows = week
    .map((row, i) => ({ row, i }))
    .filter(({ row }) => Number(row.timestamp) >= selectedStart && Number(row.timestamp) <= selectedEnd)

  const realScores = windowRows.map(({ row }) => {
    const windVal = Number(row.windMean)
    const waveVal = Number(row.waveMean)
    const windZ = Number.isFinite(windVal) ? Math.max(0, (windVal - windMeanBase) / windStdBase) : 0
    const waveZ = Number.isFinite(waveVal) ? Math.max(0, (waveVal - waveMeanBase) / waveStdBase) : 0
    return 0.55 * windZ + 0.45 * waveZ
  })

  const scoreMin = realScores.length ? Math.min(...realScores) : 0
  const scoreMax = realScores.length ? Math.max(...realScores) : 0
  const scoreSpan = Math.max(1e-6, scoreMax - scoreMin)

  anomalyTracebackRows.value = windowRows.map(({ row }, i) => {
    const hits = Array.isArray(row.eventHits) ? row.eventHits : []
    const score = Number.isFinite(realScores[i]) ? realScores[i] : 0
    const scoreNorm = (score - scoreMin) / scoreSpan
    const amplitude = 1.0 + scoreNorm * 2.0
    const duration = Math.max(1, Math.round(1 + scoreNorm * 8))
    return {
      index: row.index,
      timestamp: row.timestamp,
      time: row.time,
      amplitude,
      duration,
      labelSignal: Number(row.label) === 1 ? 1 : 0,
      windMean: row.windMean,
      waveMean: row.waveMean,
      scope: amplitude >= 2.35 ? '广域' : amplitude >= 1.7 ? '区域' : '局地',
      eventHits: hits.length ? hits.join(', ') : '-',
      matched: !!row.matched,
      realScore: score
    }
  })
}

const _requestAnomalySnapshot = async (snapshotIndex, timeoutMs = 30000) => {
  const idx = Number(snapshotIndex)
  if (!Number.isFinite(idx) || idx < 0) return null
  if (anomalySnapshotInflight.has(idx)) {
    return anomalySnapshotInflight.get(idx)
  }
  const req = api
    .post(
      '/anomaly/inspect',
      {
        labels_json: anomalyLabelsPath.value,
        events_json: anomalyEventsPath.value,
        manifest_path: anomalyManifestPath.value,
        processed_dir: anomalyProcessedPath.value,
        split: anomalySplit.value,
        recent_window_hours: 24,
        snapshot_only: true,
        max_points: 1,
        include_snapshot: true,
        snapshot_index: idx
      },
      { timeout: timeoutMs }
    )
    .then((res) => res.data?.snapshot || null)
    .finally(() => {
      anomalySnapshotInflight.delete(idx)
    })

  anomalySnapshotInflight.set(idx, req)
  return req
}

const selectAnomalyWindowPoint = (sampleIndex, withSnapshot = true) => {
  anomalySelectedSnapshotIndex.value = Number.isFinite(Number(sampleIndex)) ? Number(sampleIndex) : -1
  const selected = anomalyRecentWindow.value.find((row) => row.index === anomalySelectedSnapshotIndex.value)
  anomalySelectedTimeText.value = selected ? selected.time : '-'
  anomalySelectedIndexText.value = selected ? String(selected.index) : '-'
  void renderAnomalyWindowChart(
    document.getElementById('anomaly-window-chart'),
    anomalyRecentWindow.value,
    anomalySelectedSnapshotIndex.value,
    (ix) => selectAnomalyWindowPoint(ix, true)
  )
  if (withSnapshot && anomalySelectedSnapshotIndex.value >= 0) {
    void fetchAnomalySnapshot(anomalySelectedSnapshotIndex.value)
  }
}

const fetchAnomalySnapshot = async (snapshotIndex) => {
  if (!Number.isFinite(Number(snapshotIndex)) || Number(snapshotIndex) < 0) return
  const idx = Number(snapshotIndex)

  const cached = anomalySnapshotLocalCache.get(idx)
  if (cached) {
    anomalySnapshot.value = cached
    anomalySnapshotError.value = ''
    anomalySnapshotLoading.value = false
    await nextTick()
    if (anomalyView.value === 'monitor') {
      renderAnomalyOceanMaps(anomalySnapshot.value)
    }
    return
  }

  const reqSeq = ++anomalySnapshotRequestSeq
  anomalySnapshotLoading.value = true
  anomalySnapshotError.value = ''
  const prevSnapshot = anomalySnapshot.value

  try {
    let snap = await _requestAnomalySnapshot(idx, 30000)
    if (!snap) {
      snap = await _requestAnomalySnapshot(idx, 45000)
    }
    if (reqSeq !== anomalySnapshotRequestSeq) return
    anomalySnapshot.value = snap
    if (snap) {
      anomalySnapshotLocalCache.set(idx, snap)
    } else {
      anomalySnapshotError.value = '该时间点未返回精细图数据'
    }
  } catch (err) {
    if (reqSeq !== anomalySnapshotRequestSeq) return
    anomalySnapshot.value = prevSnapshot || null
    const msg =
      err?.code === 'ECONNABORTED'
        ? '精细图加载超时（30s），请重试或切换其他时间点'
        : `精细图加载失败: ${err.response?.data?.detail || err.message}`
    anomalySnapshotError.value = msg
  } finally {
    if (reqSeq !== anomalySnapshotRequestSeq) return
    anomalySnapshotLoading.value = false
    await nextTick()
    if (anomalyView.value === 'monitor') {
      renderAnomalyOceanMaps(anomalySnapshot.value)
    }
  }
}

const prefetchAnomalySnapshots = async (indices) => {
  if (!Array.isArray(indices) || !indices.length) return
  const token = ++anomalySnapshotPrefetchToken
  for (const idxRaw of indices) {
    if (token !== anomalySnapshotPrefetchToken) return
    const idx = Number(idxRaw)
    if (!Number.isFinite(idx) || idx < 0) continue
    if (anomalySnapshotLocalCache.has(idx)) continue
    try {
      const snap = await _requestAnomalySnapshot(idx, 25000)
      if (snap) {
        anomalySnapshotLocalCache.set(idx, snap)
      }
    } catch {
      /* silent */
    }
  }
}

const loadAnomalyOverview = async () => {
  anomalyLoading.value = true
  anomalyError.value = ''
  anomalySnapshot.value = null
  anomalySnapshotError.value = ''
  anomalySnapshotLocalCache.clear()
  anomalySnapshotPrefetchToken += 1
  anomalyRecentWindow.value = []
  anomalyRecentWeek.value = []
  anomalyTracebackRows.value = []
  anomalyTracebackStartHour.value = 0
  anomalyTracebackRangeText.value = '-'
  anomalySelectedSnapshotIndex.value = -1
  anomalySelectedTimeText.value = '-'
  anomalySelectedIndexText.value = '-'
  try {
    const res = await api.post(
      '/anomaly/inspect',
      {
        labels_json: anomalyLabelsPath.value,
        events_json: anomalyEventsPath.value,
        manifest_path: anomalyManifestPath.value,
        processed_dir: anomalyProcessedPath.value,
        split: anomalySplit.value,
        recent_window_hours: ANOMALY_LOOKBACK_HOURS,
        include_snapshot: false
      },
      { timeout: 120000 }
    )
    anomalyData.value = res.data

    anomalyRecentWeek.value = _buildRecentWindow(res.data)
    anomalyRecentWindow.value = anomalyRecentWeek.value.filter(
      (row) => Number(row.timestamp) >= (Number(res.data?.latest_timestamp || 0) - ANOMALY_MONITOR_WINDOW_HOURS * 3600)
    )
    anomalyTracebackStartHour.value = anomalyTracebackMaxStartHour.value
    recomputeAnomalyTracebackRows()

    anomalyLatestTimeText.value = _formatEpochText(res.data?.latest_timestamp)
    if (anomalyRecentWindow.value.length) {
      const latest = anomalyRecentWindow.value[anomalyRecentWindow.value.length - 1]
      if (anomalyLatestTimeText.value === '-') {
        anomalyLatestTimeText.value = latest.time
      }
      selectAnomalyWindowPoint(latest.index, false)
    }

    const score = Math.round(
      Number(res.data?.matched_positive_ratio || 0) * 70 + Number(res.data?.positive_ratio || 0) * 30
    )
    anomalyRiskScore.value = score
    if (score >= 75) {
      anomalyRiskLevel.value = '高'
      anomalyRiskClass.value = 'text-rose-300 border-rose-400/40 bg-rose-400/10'
    } else if (score >= 45) {
      anomalyRiskLevel.value = '中'
      anomalyRiskClass.value = 'text-amber-300 border-amber-400/40 bg-amber-400/10'
    } else {
      anomalyRiskLevel.value = '低'
      anomalyRiskClass.value = 'text-emerald-300 border-emerald-400/40 bg-emerald-400/10'
    }

    const warningLevel = score >= 85 ? '红' : score >= 65 ? '橙' : score >= 45 ? '黄' : '蓝'
    anomalyWarning.value = {
      level: warningLevel,
      targets: warningLevel === '红' ? '应急指挥中心/港口/全部海上作业单位' : '值班中心/港口/重点航线单位',
      actions: [
        '持续滚动监测风速、波高和异常事件命中情况。',
        '对高风险区域限制作业窗口，发布避险建议。',
        '关联台风路径变化并更新分区风险等级。'
      ]
    }

    const points = Array.isArray(res.data?.points) ? res.data.points : []
    const byEvent = new Map()
    points.forEach((row) => {
      const hits = Array.isArray(row.event_hits) ? row.event_hits : []
      hits.forEach((name) => {
        byEvent.set(name, (byEvent.get(name) || 0) + 1)
      })
    })
    anomalyCouplings.value = [...byEvent.entries()].slice(0, 6).map(([name, c], idx) => ({
      name,
      score: Math.min(99, 52 + c * 8 + idx * 2),
      speed: (16 + c * 2 + idx).toFixed(1),
      intensity: (34 + c * 3 + idx * 2).toFixed(1)
    }))
    if (!anomalyCouplings.value.length) {
      anomalyCouplings.value = [{ name: '无强台风命中事件', score: 36, speed: '12.0', intensity: '28.0' }]
    }

    anomalyCases.value = anomalyCouplings.value.slice(0, 4).map((c, idx) => ({
      id: `CASE-${String(idx + 1).padStart(3, '0')} ${c.name}`,
      similarity: `${Math.min(98, 72 + idx * 7)}%`,
      window: `${8 + idx * 4}h`
    }))

    const windNow = 6.5 + Number(res.data?.positive_ratio || 0) * 28 + Number(res.data?.matched_positive_ratio || 0) * 5
    const waveNow = 1.0 + Number(res.data?.positive_ratio || 0) * 4 + Number(res.data?.matched_positive_ratio || 0) * 1.8
    anomalyMonitor.value = {
      windNow: Number(windNow.toFixed(1)),
      waveNow: Number(waveNow.toFixed(2)),
      windBand: '5.0-10.5',
      waveBand: '0.8-2.2',
      status: score >= 65 ? '异常增强中' : score >= 45 ? '轻度异常' : '整体平稳'
    }

    const nowText = new Date().toISOString().replace('T', ' ').slice(0, 19)
    const dataTimeText = _formatEpochText(res.data?.latest_timestamp)
    anomalyIssueTimeText.value = nowText
    anomalyDataTimeText.value = dataTimeText

    anomalyWarningLogs.value = [
      {
        id: `${Date.now()}-issue`,
        time: nowText,
        issuedAt: nowText,
        dataTime: dataTimeText,
        action: `${warningLevel}色预警发布`,
        note: `风险分=${score}，异常点=${res.data?.num_positive || 0}，命中事件=${res.data?.matched_event_count || 0}`
      },
      {
        id: `${Date.now()}-risk`,
        time: nowText,
        issuedAt: nowText,
        dataTime: dataTimeText,
        action: '风险区域更新',
        note: '按黄渤海分区完成低/中/高/极高风险重算'
      },
      ...anomalyWarningLogs.value
    ].slice(0, 12)

    anomalyBrief.value = [
      '【风-浪异常灾害预警简报】',
      `数据时效时间: ${dataTimeText} UTC`,
      `发布时间: ${nowText} UTC`,
      `分片: ${res.data?.split || '-'} | 风险等级: ${anomalyRiskLevel.value} | 预警: ${warningLevel}色`,
      `异常样本: ${res.data?.num_positive || 0}/${res.data?.num_samples || 0}`,
      `事件命中: ${res.data?.matched_event_count || 0} | 命中异常点: ${res.data?.matched_positive || 0}`,
      `监测估计: 风速 ${anomalyMonitor.value.windNow} m/s，波高 ${anomalyMonitor.value.waveNow} m`,
      '建议: 加强重点海域巡检，按风险等级动态调整作业窗口。'
    ].join('\n')

    await nextTick()
    renderAnomalyMonitorChart(document.getElementById('anomaly-monitor-chart'), anomalyMonitor.value)
    await renderAnomalyWindowChart(
      document.getElementById('anomaly-window-chart'),
      anomalyRecentWindow.value,
      anomalySelectedSnapshotIndex.value,
      (ix) => selectAnomalyWindowPoint(ix, true)
    )
    renderAnomalyTimelineChart(anomalyTracebackRows.value)
    renderAnomalyRiskMapChart(anomalyData.value, anomalyRiskScore.value, anomalySnapshot.value)

    if (anomalySelectedSnapshotIndex.value >= 0) {
      void fetchAnomalySnapshot(anomalySelectedSnapshotIndex.value)
      const pending = anomalyRecentWindow.value
        .map((row) => Number(row.index))
        .filter((ix) => Number.isFinite(ix) && ix >= 0 && ix !== Number(anomalySelectedSnapshotIndex.value))
      void prefetchAnomalySnapshots(pending)
    }
  } catch (err) {
    anomalyError.value = `异常模块加载失败: ${err.response?.data?.detail || err.message}`
    anomalyData.value = null
  } finally {
    anomalyLoading.value = false
  }
}

const copyAnomalyBrief = async () => {
  try {
    await navigator.clipboard.writeText(anomalyBrief.value || '')
  } catch (err) {
    anomalyError.value = `复制失败: ${err?.message || 'unknown'}`
  }
}

export function resizeAnomalyCharts() {
  resizeAnomalyInspectCharts()
}

export { anomalyData, anomalyProduct }

export function useAnomaly() {
  const stop1 = watch(anomalyView, async (view) => {
    if (!anomalyData.value) return
    await nextTick()
    if (view === 'monitor') {
      renderAnomalyMonitorChart(document.getElementById('anomaly-monitor-chart'), anomalyMonitor.value)
      await renderAnomalyWindowChart(
        document.getElementById('anomaly-window-chart'),
        anomalyRecentWindow.value,
        anomalySelectedSnapshotIndex.value,
        (ix) => selectAnomalyWindowPoint(ix, true)
      )
      renderAnomalyOceanMaps(anomalySnapshot.value)
    }
    if (view === 'detect') renderAnomalyTimelineChart(anomalyTracebackRows.value)
    if (view === 'typhoon') renderAnomalyRiskMapChart(anomalyData.value, anomalyRiskScore.value, anomalySnapshot.value)
  })

  const stop2 = watch(anomalyTracebackStartHour, async () => {
    recomputeAnomalyTracebackRows()
    if (anomalyView.value === 'detect') {
      await nextTick()
      renderAnomalyTimelineChart(anomalyTracebackRows.value)
    }
  })

  const stop3 = watch(anomalySnapshot, async () => {
    if (anomalyView.value === 'typhoon') {
      await nextTick()
      renderAnomalyRiskMapChart(anomalyData.value, anomalyRiskScore.value, anomalySnapshot.value)
    }
  })

  onUnmounted(() => {
    stop1()
    stop2()
    stop3()
  })

  return {
    anomalyLabelsPath,
    anomalyEventsPath,
    anomalyManifestPath,
    anomalyProcessedPath,
    anomalySplit,
    anomalyLoading,
    anomalyError,
    anomalyData,
    anomalyView,
    anomalyRiskScore,
    anomalyRiskLevel,
    anomalyRiskClass,
    anomalySnapshot,
    anomalySnapshotLoading,
    anomalySnapshotError,
    anomalyRecentWindow,
    anomalyRecentWeek,
    anomalySelectedSnapshotIndex,
    anomalyLatestTimeText,
    anomalySelectedTimeText,
    anomalySelectedIndexText,
    anomalyTracebackRows,
    anomalyTracebackWindowHours,
    anomalyTracebackStartHour,
    anomalyTracebackRangeText,
    anomalyTracebackMaxStartHour,
    anomalyTracebackStepHours,
    anomalyDataTimeText,
    anomalyIssueTimeText,
    anomalyCouplings,
    anomalyCases,
    anomalyBrief,
    anomalyWarning,
    anomalyWarningLogs,
    anomalyMonitor,
    loadAnomalyOverview,
    copyAnomalyBrief,
    resizeAnomalyCharts,
    anomalyProduct
  }
}
