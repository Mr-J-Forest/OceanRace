// @ts-nocheck
import { ref, computed, watch, nextTick } from 'vue'
import { api } from '../api/client'
import Plotly from 'plotly.js-dist-min'
import { getChartLayoutBase } from './plotly/chartLayout'

let eddyProgressTimer: ReturnType<typeof setInterval> | null = null

const eddyModelPath = ref('models/eddy_model.pt')
const eddyDataPath = ref('data/processed/eddy_detection/path.txt')
const eddyDatasetInfo = ref('')
const eddyDates = ref([])
const eddyDayIndex = ref(0)
const eddySelectedDate = ref('')
const eddyLoading = ref(false)
const eddyPredicting = ref(false)
export const eddyResult = ref<any>(null)
const eddyProgress = ref(0)
const normalizeDateLabel = (label: unknown) => String(label || '').slice(0, 10)
const eddyMinDate = computed(() => {
  if (!eddyDates.value.length) return ''
  return normalizeDateLabel(eddyDates.value[0])
})

const eddyMaxDate = computed(() => {
  if (!eddyDates.value.length) return ''
  return normalizeDateLabel(eddyDates.value[eddyDates.value.length - 1])
})

const eddyDateHint = computed(() => {
  if (!eddySelectedDate.value || !eddyDates.value.length) return ''
  const idx = eddyDates.value.findIndex((d) => normalizeDateLabel(d) === eddySelectedDate.value)
  if (idx >= 0) return ''
  return `该日期不在数据集中，将自动使用最近可选日期（${normalizeDateLabel(eddyDates.value[eddyDayIndex.value] || '')}）。`
})
const eddySelectedCenter = ref(null)

const eddySelectedRadius = computed(() => {
  if (!eddySelectedCenter.value || !eddySelectedCenter.value.area) return '0.0'
  const latRad = Math.abs(eddySelectedCenter.value.lat) * Math.PI / 180
  const cellArea = 13.875 * 13.875 * Math.cos(latRad) // km^2
  const physArea = eddySelectedCenter.value.area * cellArea
  return Math.sqrt(physArea / Math.PI).toFixed(1)
})

const eddySelectedVelocity = computed(() => {
  if (!eddySelectedCenter.value || !eddySelectedCenter.value.area) return '0.000'
  const radius = parseFloat(eddySelectedRadius.value)
  if (radius <= 0) return '0.000'
  // v = w * R -> w = v / R. Assuming mean velocity v ~ 0.3 m/s = 25.92 km/d
  return (25.92 / radius).toFixed(3)
})

const eddyWorkbenchTab = ref<'forecast' | 'track'>('forecast')
const eddyTrackData = ref(null)
const eddyTrackLoading = ref(false)

const startEddyProgress = () => {
  if (eddyProgressTimer) clearInterval(eddyProgressTimer)
  eddyProgress.value = 8
  eddyProgressTimer = setInterval(() => {
    if (eddyProgress.value >= 90) return
    const step = Math.max(1, Math.ceil((90 - eddyProgress.value) / 8))
    eddyProgress.value = Math.min(90, eddyProgress.value + step)
  }, 260)
}

const completeEddyProgress = () => {
  if (eddyProgressTimer) {
    clearInterval(eddyProgressTimer)
    eddyProgressTimer = null
  }
  eddyProgress.value = 100
}

const resetEddyProgress = () => {
  if (eddyProgressTimer) {
    clearInterval(eddyProgressTimer)
    eddyProgressTimer = null
  }
  eddyProgress.value = 0
}
watch(eddyDayIndex, (idx) => {
  if (idx < 0 || idx >= eddyDates.value.length) return
  eddySelectedDate.value = normalizeDateLabel(eddyDates.value[idx])
  
  // Date changed: clear track data and selection
  eddySelectedCenter.value = null
  eddyTrackData.value = null
})

watch(eddySelectedDate, (selected) => {
  if (!selected || !eddyDates.value.length) return
  const exact = eddyDates.value.findIndex((d) => normalizeDateLabel(d) === selected)
  if (exact >= 0) {
    eddyDayIndex.value = exact
    return
  }
})
const loadEddyDefaults = async () => {
  try {
    const res = await api.get('/eddy/default-paths')
    if (res.data?.model_path) eddyModelPath.value = res.data.model_path
    if (res.data?.data_path) eddyDataPath.value = res.data.data_path
  } catch (err) {
    try {
      const fallback = await api.get('/eddy/default-data-path')
      if (fallback.data?.path) eddyDataPath.value = fallback.data.path
    } catch (fallbackErr) {
      // no-op
    }
  }
}

const loadEddyDataInfo = async () => {
  eddyLoading.value = true
  try {
    const res = await api.post('/eddy/dataset-info', {
      data_path: eddyDataPath.value
    })
    eddyDates.value = Array.isArray(res.data.dates) ? res.data.dates : []
    eddyDayIndex.value = 0
    eddySelectedDate.value = eddyDates.value.length ? normalizeDateLabel(eddyDates.value[0]) : ''
    eddyDatasetInfo.value = res.data.info || `可选天数: ${eddyDates.value.length}`
  } catch (err: unknown) {
    const e = err as { response?: { data?: { detail?: string } }; message?: string }
    eddyDatasetInfo.value = `[ERR] ${e.response?.data?.detail || e.message}`
    eddyDates.value = []
    eddySelectedDate.value = ''
  } finally {
    eddyLoading.value = false
  }
}

const shiftEddyDate = (offset: number) => {
  if (!eddyDates.value.length) return
  const target = Math.max(0, Math.min(eddyDates.value.length - 1, eddyDayIndex.value + offset))
  eddyDayIndex.value = target
}

const runEddyPrediction = async () => {
  if (eddyDates.value.length === 0) return
  if (!eddySelectedDate.value) return
  const exactIdx = eddyDates.value.findIndex((d) => normalizeDateLabel(d) === eddySelectedDate.value)
  if (exactIdx >= 0) {
    eddyDayIndex.value = exactIdx
  }
  eddyPredicting.value = true
  startEddyProgress()
  eddyResult.value = null
  try {
    const res = await api.post('/eddy/predict-day', {
      model_path: eddyModelPath.value,
      data_path: eddyDataPath.value,
      day_index: eddyDayIndex.value,
      input_steps: 1,
      base_channels: 32,
      min_region_pixels: 16
    })
    completeEddyProgress()
    eddyResult.value = res.data
    await nextTick()
    renderEddyPlot(res.data)
  } catch (err: unknown) {
    resetEddyProgress()
    const e = err as { response?: { data?: { detail?: string } }; message?: string }
    alert(`涡旋预测失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    eddyPredicting.value = false
    if (eddyProgress.value === 100) {
      setTimeout(() => {
        if (!eddyPredicting.value) resetEddyProgress()
      }, 600)
    }
  }
}

const downloadEddyInfo = () => {
  if (!eddyResult.value) {
    alert('请先完成当天预测，再下载信息。')
    return
  }

  const payload = eddyResult.value
  const centers = Array.isArray(payload.centers) ? payload.centers : []
  const mask = Array.isArray(payload.pred_mask) ? payload.pred_mask : []
  const nrows = mask.length
  const ncols = nrows > 0 && Array.isArray(mask[0]) ? mask[0].length : 0

  const exportData = {
    exported_at: new Date().toISOString(),
    day_index: payload.day_index,
    day_label: payload.day_label,
    model_path: eddyModelPath.value,
    data_path: eddyDataPath.value,
    cyclonic_count: Number(payload.cyclonic_count || 0),
    anticyclonic_count: Number(payload.anticyclonic_count || 0),
    total_eddy_count: Number(payload.cyclonic_count || 0) + Number(payload.anticyclonic_count || 0),
    center_count: centers.length,
    grid_shape: [nrows, ncols],
    centers,
    pred_mask: mask
  }

  const day = String(payload.day_label || `idx_${payload.day_index ?? 'unknown'}`).replace(/[^0-9A-Za-z_-]/g, '_')
  const blob = new Blob([JSON.stringify(exportData, null, 2)], { type: 'application/json;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = `eddy_info_${day}.json`
  document.body.appendChild(a)
  a.click()
  a.remove()
  URL.revokeObjectURL(url)
}

const loadEddyTrack = async () => {
  if (!eddySelectedCenter.value || eddyDates.value.length === 0) return
  eddyTrackLoading.value = true
  try {
    const res = await api.post('/eddy/track', {
      data_path: eddyDataPath.value,
      model_path: eddyModelPath.value,
      start_day_index: eddyDayIndex.value,
      r: eddySelectedCenter.value.r,
      c: eddySelectedCenter.value.c,
      class_id: eddySelectedCenter.value.class_id
    })
    eddyTrackData.value = res.data
  } catch (err: unknown) {
    const e = err as { response?: { data?: { detail?: string } }; message?: string }
    alert(`追踪失败: ${e.response?.data?.detail || e.message}`)
  } finally {
    eddyTrackLoading.value = false
    if (eddyTrackData.value) {
      await nextTick()
      renderEddyTrackChart()
    }
  }
}

const renderEddyPlot = (payload) => {
  const container = document.getElementById('eddy-chart')
  if (!container || !payload) return

  const adt = payload.adt || []
  const mask = payload.pred_mask || []
  const centers = Array.isArray(payload.centers) ? payload.centers : []

  const traces = [
    {
      z: adt,
      type: 'heatmap',
      colorscale: [
        [0.0, '#012b6a'],   // 深蓝
        [0.15, '#4d8dcd'],
        [0.3, '#7bb7d9'],
        [0.45, '#c6dbef'],
        [0.5, '#f9eeee'],   // 中间白色（关键！）
        [0.55, '#fff5f5'],
        [0.7, '#f29371'],
        [0.85, '#ee4633'],
        [1.0, '#c9050caf']    // 深红
      ],
      opacity: 0.79,
      zsmooth: 'best',
      colorbar: {
        title: { text: 'ADT' },
        tickfont: { color: '#94a3b8', size: 10 }
      },
      hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4' }
    }
  ]

  traces.push({
    z: (mask || []).map((row) => (row || []).map((v) => (v === 1 ? 1 : 0))),
    type: 'contour',
    contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
    line: { color: '#003f7a', width: 1.4 },
    showscale: false,
    hoverinfo: 'skip'
  })

  const cyclonicColor = '#003f7a'
  const anticyclonicColor = '#a30303'

  traces.push({
    z: (mask || []).map((row) => (row || []).map((v) => (v === 2 ? 1 : 0))),
    type: 'contour',
    contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
    line: { color: '#a30303', width: 1.4 },
    showscale: false,
    hoverinfo: 'skip'
  })

  if (centers.length > 0) {
    const cyclonicCenters = centers.filter((c) => Number(c?.class_id) === 1)
    const anticyclonicCenters = centers.filter((c) => Number(c?.class_id) === 2)
    const otherCenters = centers.filter((c) => !Number.isFinite(Number(c?.class_id)) || Number(c?.class_id) <= 0)

    if (cyclonicCenters.length > 0) {
      traces.push({
        x: cyclonicCenters.map((c) => Number(c?.c)),
        y: cyclonicCenters.map((c) => Number(c?.r)),
        type: 'scatter',
        mode: 'markers',
        marker: { color: cyclonicColor, size: 4, symbol: 'x' },
        name: '气旋式',
        hovertext: cyclonicCenters.map((c) => `气旋式涡旋<br>Lat: ${Number(c.lat).toFixed(2)}<br>Lon: ${Number(c.lon).toFixed(2)}<br>像素面积: ${c.area}`),
        hoverinfo: 'text',
        customdata: cyclonicCenters // for click event
      })
    }

    if (anticyclonicCenters.length > 0) {
      traces.push({
        x: anticyclonicCenters.map((c) => Number(c?.c)),
        y: anticyclonicCenters.map((c) => Number(c?.r)),
        type: 'scatter',
        mode: 'markers',
        marker: { color: anticyclonicColor, size: 4, symbol: 'x' },
        name: '反气旋式',
        hovertext: anticyclonicCenters.map((c) => `反气旋式涡旋<br>Lat: ${Number(c.lat).toFixed(2)}<br>Lon: ${Number(c.lon).toFixed(2)}<br>像素面积: ${c.area}`),
        hoverinfo: 'text',
        customdata: anticyclonicCenters
      })
    }

    if (otherCenters.length > 0) {
      traces.push({
        x: otherCenters.map((c) => Number(c?.c)),
        y: otherCenters.map((c) => Number(c?.r)),
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#f8fafc', size: 7, symbol: 'diamond-open', line: { width: 1.2, color: '#0f172a' } },
        name: '未知类型',
        hovertext: otherCenters.map((c) => `未知涡旋<br>Lat: ${Number(c.lat).toFixed(2)}<br>Lon: ${Number(c.lon).toFixed(2)}<br>像素面积: ${c.area}`),
        hoverinfo: 'text',
        customdata: otherCenters
      })
    }
  }

  const layout = {
    ...getChartLayoutBase(''),
    margin: { l: 42, r: 24, t: 16, b: 36 },
    xaxis: {
      title: { text: 'longitude index', font: { color: '#64748b', size: 11 } },
      showgrid: false,
      zeroline: false,
      constrain: 'domain',
      tickfont: { color: '#64748b', size: 10 }
    },
    yaxis: {
      title: { text: 'latitude index', font: { color: '#64748b', size: 11 } },
      autorange: 'reversed',
      showgrid: false,
      zeroline: false,
      scaleanchor: 'x',
      scaleratio: 1,
      constrain: 'domain',
      tickfont: { color: '#64748b', size: 10 }
    },
    showlegend: false,
    hovermode: 'closest',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(2, 6, 23, 0.55)'
  }

  Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false })

  container.on('plotly_click', (data) => {
    if (data.points && data.points.length > 0) {
      const pt = data.points[0]
      if (pt.customdata) {
        eddySelectedCenter.value = pt.customdata
      }
    }
  })
}

const renderEddyTrackChart = () => {
  const container = document.getElementById('eddy-track-chart')
  if (!container || !eddyTrackData.value || !eddyTrackData.value.nodes) return

  const nodes = eddyTrackData.value.nodes
  
  const traces = [
    {
      x: nodes.map(n => n.day_index),
      y: nodes.map(n => n.intensity),
      type: 'scatter',
      mode: 'lines+markers',
      line: { color: '#0ea5e9', width: 2 },
      marker: { color: '#38bdf8', size: 6 },
      name: '核心温盐异常',
      hovertemplate: 'Day %{x}<br>强度: %{y:.2f}<extra></extra>'
    }
  ]

  const layout = {
    ...getChartLayoutBase(''),
    margin: { l: 40, r: 20, t: 20, b: 30 },
    xaxis: {
      title: { text: '时间 (Day)', font: { color: '#64748b', size: 10 } },
      tickfont: { color: '#64748b', size: 10 },
      showgrid: true,
      gridcolor: 'rgba(51,65,85,0.4)',
      zeroline: false
    },
    yaxis: {
      title: { text: '核心异常强度', font: { color: '#64748b', size: 10 } },
      tickfont: { color: '#64748b', size: 10 },
      showgrid: true,
      gridcolor: 'rgba(51,65,85,0.4)',
      zeroline: false
    },
    showlegend: false,
    hovermode: 'x unified',
    paper_bgcolor: 'rgba(0,0,0,0)',
    plot_bgcolor: 'rgba(0,0,0,0)'
  }

  Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false })
}

export function resizeEddyChart() {
  if (eddyResult.value) {
    Plotly.Plots.resize('eddy-chart')
  }
  if (eddyTrackData.value) {
    Plotly.Plots.resize('eddy-track-chart')
  }
}

export function disposeEddyTimers() {
  if (eddyProgressTimer) {
    clearInterval(eddyProgressTimer)
    eddyProgressTimer = null
  }
}

export function useEddy() {
  return {
    eddyModelPath,
    eddyDataPath,
    eddyDatasetInfo,
    eddyDates,
    eddyDayIndex,
    eddySelectedDate,
    eddyLoading,
    eddyPredicting,
    eddyResult,
    eddyProgress,
    eddyMinDate,
    eddyMaxDate,
    eddyDateHint,
    eddySelectedCenter,
    eddySelectedRadius,
    eddySelectedVelocity,
    eddyWorkbenchTab,
    eddyTrackData,
    eddyTrackLoading,
    loadEddyTrack,
    loadEddyDefaults,
    loadEddyDataInfo,
    shiftEddyDate,
    runEddyPrediction,
    downloadEddyInfo,
    renderEddyPlot,
    resizeEddyChart,
    disposeEddyTimers
  }
}
