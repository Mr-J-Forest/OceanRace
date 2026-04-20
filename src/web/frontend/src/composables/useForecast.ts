// @ts-nocheck
import { ref, nextTick, watch } from 'vue'
import { api } from '../api/client'
import Plotly from 'plotly.js-dist-min'
import { getChartLayoutBase } from './plotly/chartLayout'

const modelPath = ref('models/forecast_model.pt')
const dataPath = ref('data/processed/element_forecasting/path.txt')
const datasetInfo = ref('')
const maxIndex = ref(0)
const startIdx = ref(0)
const loadingInfo = ref(false)
const predicting = ref(false)
const hasResult = ref(false)
const sessionId = ref('')
const geoLon = ref([])
const geoLat = ref([])

const totalSteps = ref(0)
const currentStep = ref(0)

const isPlaying = ref(false)
let playTimer = null
let playToken = 0
let currentStepDataCache = null
let stepFetchTimer = null
let spatialStreamlineTraceIndices = []
let stepRequestSeq = 0
let lastAppliedStepSeq = 0
let lastLoadedStepIdx = -1
const colorRangeCache = new Map()
const showQuiver = ref(true)
const playbackSpeed = ref(1.0)
const curveTitle = ref('区域变化趋势')
/** 单步对应预报小时数（与后端 PredictRequest.step_hours 一致） */
const stepHours = ref(1)
/** 空间展示子页：预报场 | 实况·偏差 | 异常预警 | 关联分析 */
const spatialWorkbenchTab = ref<'forecast' | 'compare' | 'warning' | 'correlation'>('forecast')
const compareVarIndex = ref(0)
/** 底部时序折线图显隐；关闭时主图区占满高度，避免地图被压扁 */
const showCurvePanel = ref(true)

// --- 异常预警与关联分析状态 ---
const warningsData = ref(null)
const loadingWarning = ref(false)
const warningThresholds = ref({ SST: 1.5, SSS: 0.5, SSUV: 0.3 })
const currentWarningVar = ref('')

const correlationData = ref(null)
const loadingCorr = ref(false)
const corrVar1 = ref('SST')
const corrVar2 = ref('SSS')

const loadDefaultDataPath = async () => {
  try {
    const res = await api.get('/default-data-path')
    if (res.data.path) dataPath.value = res.data.path
  } catch (err) {
    // no-op
  }
}

const loadDataInfo = async () => {
  loadingInfo.value = true
  try {
    const res = await api.post('/dataset-info', { data_path: dataPath.value })
    maxIndex.value = res.data.max_index
    datasetInfo.value = res.data.info
  } catch (err) {
    datasetInfo.value = `[ERR] 通信阻断: ${err.response?.data?.detail || err.message}`
  } finally {
    loadingInfo.value = false
  }
}

const runPrediction = async () => {
  predicting.value = true
  hasResult.value = false
  stopPlay()
  colorRangeCache.clear()
  lastLoadedStepIdx = -1
  stepRequestSeq = 0
  lastAppliedStepSeq = 0

  try {
    const res = await api.post('/predict', {
      model_path: modelPath.value,
      data_path: dataPath.value,
      start_idx: startIdx.value
    })

    sessionId.value = res.data.session_id
    totalSteps.value = res.data.steps
    geoLon.value = res.data.lon || []
    geoLat.value = res.data.lat || []
    currentStep.value = 0
    hasResult.value = true
    if (res.data.step_hours != null && Number.isFinite(Number(res.data.step_hours))) {
      stepHours.value = Math.max(0.25, Number(res.data.step_hours))
    }

    await nextTick()
    const tasks = [loadStepData()]
    if (showCurvePanel.value) tasks.push(loadCurveData())
    await Promise.all(tasks)
    await nextTick()
    resizeForecastCharts(hasResult.value)
  } catch (err) {
    alert(`核心推理引擎故障: ${err.response?.data?.detail || err.message}`)
  } finally {
    predicting.value = false
  }
}

const loadStepData = async (stepIdx = currentStep.value) => {
  if (!sessionId.value) return
  const targetStep = Number.isFinite(stepIdx) ? Number(stepIdx) : Number(currentStep.value)
  const reqSeq = ++stepRequestSeq
  try {
    const res = await api.get(`/predict/${sessionId.value}/step/${targetStep}`, { params: { compare: 1 } })
    // Ignore stale responses during rapid slider drags.
    if (reqSeq < stepRequestSeq || reqSeq < lastAppliedStepSeq) return
    lastAppliedStepSeq = reqSeq
    lastLoadedStepIdx = targetStep
    currentStepDataCache = res.data
    if (spatialWorkbenchTab.value === 'compare') {
      renderCompareSpatial(currentStepDataCache)
    } else {
      // Default path keeps forecast view responsive during playback.
      renderSpatialPlot(currentStepDataCache)
    }
    if (showCurvePanel.value) updateVerticalLineOnCurve()
  } catch (err) {
    console.error('数据游标读取失败', err)
  }
}

const renderWarningHeatmap = (varName) => {
  const container = document.getElementById('warning-spatial-chart')
  if (!container || !warningsData.value?.masks) return

  const maskGrid = warningsData.value.masks[varName]
  if (!maskGrid) return

  const style = getVariableRenderStyle(varName)
  const renderRows = maskGrid.length
  const renderCols = renderRows > 0 ? maskGrid[0].length : 0

  let xCoords, yCoords
  if (geoLon.value.length > 1 && geoLat.value.length > 1) {
    const lons = geoLon.value
    const lats = geoLat.value
    xCoords = Array.from({ length: renderCols }, (_, i) => lons[0] + (i / Math.max(1, renderCols - 1)) * (lons[lons.length - 1] - lons[0]))
    yCoords = Array.from({ length: renderRows }, (_, i) => lats[0] + (i / Math.max(1, renderRows - 1)) * (lats[lats.length - 1] - lats[0]))
  }

  const trace = {
    z: maskGrid,
    ...(xCoords && yCoords ? { x: xCoords, y: yCoords } : {}),
    type: 'heatmap',
    zsmooth: false,
    colorscale: [
      [0.0, 'rgba(0,0,0,0)'],
      [0.33, 'rgba(0,0,0,0)'],
      [0.34, '#fde047'], // Level 1: 轻度 (Yellow)
      [0.66, '#fde047'],
      [0.67, '#f97316'], // Level 2: 中度 (Orange)
      [0.99, '#f97316'],
      [1.0, '#ef4444']   // Level 3: 重度 (Red)
    ],
    zmin: 0,
    zmax: 3,
    hoverongaps: false,
    showscale: true,
    colorbar: {
      title: { text: '预警级别', font: { size: 9, color: '#94a3b8' } },
      tickvals: [1, 2, 3],
      ticktext: ['轻度', '中度', '重度'],
      len: 0.8,
      thickness: 10,
      x: 1.02,
      xanchor: 'left',
      y: 0.5,
      yanchor: 'middle',
      tickfont: { color: '#94a3b8', size: 8, family: 'JetBrains Mono, monospace' },
      outlinewidth: 0,
      bgcolor: 'rgba(3,7,18,0.55)',
      bordercolor: 'rgba(6,182,212,0.25)',
      borderwidth: 1
    },
    hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4', font: { color: '#06b6d4', family: 'JetBrains Mono, monospace' } }
  }

  const layout = {
    ...getChartLayoutBase(''),
    margin: { l: 36, r: 80, t: 32, b: 36 },
    title: {
      text: `${style.displayName} 异常预警区`,
      font: { color: '#e2e8f0', size: 12, family: 'Orbitron, sans-serif' },
      x: 0.5,
      xanchor: 'center',
      y: 0.98,
      yanchor: 'top'
    },
    xaxis: { constrain: 'domain', showticklabels: true, tickfont: { color: '#64748b', size: 8 }, gridcolor: 'rgba(30, 41, 59, 0.45)', zeroline: false },
    yaxis: { autorange: 'reversed', scaleanchor: 'x', scaleratio: renderCols > 0 && renderRows > 0 ? renderCols / renderRows : 1, showticklabels: true, tickfont: { color: '#64748b', size: 8 }, gridcolor: 'rgba(30, 41, 59, 0.45)', zeroline: false },
    uirevision: 'forecast-warning-spatial-v1'
  }

  Plotly.react(container, [trace], layout, { responsive: true, displayModeBar: false })
}

const renderCorrelationHeatmap = () => {
  const container = document.getElementById('correlation-chart')
  if (!container || !correlationData.value?.correlation_grid) return

  const grid = correlationData.value.correlation_grid
  const renderRows = grid.length
  const renderCols = renderRows > 0 ? grid[0].length : 0

  let xCoords, yCoords
  if (geoLon.value.length > 1 && geoLat.value.length > 1) {
    const lons = geoLon.value
    const lats = geoLat.value
    xCoords = Array.from({ length: renderCols }, (_, i) => lons[0] + (i / Math.max(1, renderCols - 1)) * (lons[lons.length - 1] - lons[0]))
    yCoords = Array.from({ length: renderRows }, (_, i) => lats[0] + (i / Math.max(1, renderRows - 1)) * (lats[lats.length - 1] - lats[0]))
  }

  const trace = {
    z: grid,
    ...(xCoords && yCoords ? { x: xCoords, y: yCoords } : {}),
    type: 'heatmap',
    zsmooth: 'best',
    colorscale: 'RdBu_r',
    zmin: -1,
    zmax: 1,
    hoverongaps: false,
    showscale: true,
    colorbar: {
      title: { text: 'Pearson<br>Corr', font: { size: 9, color: '#94a3b8' } },
      len: 0.8,
      thickness: 10,
      x: 1.02,
      xanchor: 'left',
      y: 0.5,
      yanchor: 'middle',
      tickfont: { color: '#94a3b8', size: 8, family: 'JetBrains Mono, monospace' },
      outlinewidth: 0,
      bgcolor: 'rgba(3,7,18,0.55)',
      bordercolor: 'rgba(6,182,212,0.25)',
      borderwidth: 1
    },
    hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4', font: { color: '#06b6d4', family: 'JetBrains Mono, monospace' } }
  }

  const layout = {
    ...getChartLayoutBase(''),
    margin: { l: 36, r: 80, t: 32, b: 36 },
    title: {
      text: `${correlationData.value.var1} & ${correlationData.value.var2} 时空关联分析`,
      font: { color: '#e2e8f0', size: 12, family: 'Orbitron, sans-serif' },
      x: 0.5,
      xanchor: 'center',
      y: 0.98,
      yanchor: 'top'
    },
    xaxis: { constrain: 'domain', showticklabels: true, tickfont: { color: '#64748b', size: 8 }, gridcolor: 'rgba(30, 41, 59, 0.45)', zeroline: false },
    yaxis: { autorange: 'reversed', scaleanchor: 'x', scaleratio: renderCols > 0 && renderRows > 0 ? renderCols / renderRows : 1, showticklabels: true, tickfont: { color: '#64748b', size: 8 }, gridcolor: 'rgba(30, 41, 59, 0.45)', zeroline: false },
    uirevision: 'forecast-correlation-spatial-v1'
  }

  Plotly.react(container, [trace], layout, { responsive: true, displayModeBar: false })
}

const scheduleStepDataLoad = (stepIdx = currentStep.value, immediate = false) => {
  if (stepFetchTimer) {
    clearTimeout(stepFetchTimer)
    stepFetchTimer = null
  }
  if (!sessionId.value) return

  const targetStep = Number.isFinite(stepIdx) ? Number(stepIdx) : Number(currentStep.value)
  if (!immediate && targetStep === lastLoadedStepIdx) return

  if (immediate) {
    loadStepData(targetStep)
    return
  }

  // Small debounce keeps slider drag smooth and avoids request storms.
  stepFetchTimer = setTimeout(() => {
    loadStepData(targetStep)
    stepFetchTimer = null
  }, 70)
}

const onQuiverToggle = () => {
  const container = document.getElementById('spatial-chart')
  if (!container) return
  if (!spatialStreamlineTraceIndices.length) return
  Plotly.restyle(container, { visible: showQuiver.value }, spatialStreamlineTraceIndices)
}

const buildStreamlineTrace = (stepData, subplotIndex, renderRows, renderCols, xCoords, yCoords) => {
  const ssuItem = (stepData?.data || []).find((it) => String(it?.var || '').toUpperCase() === 'SSU')
  const ssvItem = (stepData?.data || []).find((it) => String(it?.var || '').toUpperCase() === 'SSV')
  const uField = ssuItem?.data
  const vField = ssvItem?.data

  if (!Array.isArray(uField) || !Array.isArray(vField) || uField.length === 0 || vField.length === 0) {
    return null
  }

  const rows = Math.min(uField.length, vField.length)
  const cols = Math.min(uField[0].length, vField[0].length)
  const stride = Math.max(3, Math.floor(Math.min(rows, cols) / 18))
  const sx = cols > 1 && renderCols > 1 ? (renderCols - 1) / (cols - 1) : 1
  const sy = rows > 1 && renderRows > 1 ? (renderRows - 1) / (rows - 1) : 1

  const mapX = (cIdx) => {
    if (!xCoords || xCoords.length < 2 || renderCols < 2) return cIdx
    return xCoords[0] + (cIdx / (renderCols - 1)) * (xCoords[xCoords.length - 1] - xCoords[0])
  }
  const mapY = (rIdx) => {
    if (!yCoords || yCoords.length < 2 || renderRows < 2) return rIdx
    return yCoords[0] + (rIdx / (renderRows - 1)) * (yCoords[yCoords.length - 1] - yCoords[0])
  }

  const streamX = []
  const streamY = []

  for (let r = 0; r < rows; r += stride) {
    for (let c = 0; c < cols; c += stride) {
      let cxRaw = c
      let cyRaw = r
      const maxSteps = 8
      let lastDx = 0
      let lastDy = 0

      for (let k = 0; k < maxSteps; k++) {
        const rr = Math.max(0, Math.min(rows - 1, Math.round(cyRaw)))
        const cc = Math.max(0, Math.min(cols - 1, Math.round(cxRaw)))

        const u = Number(uField[rr]?.[cc])
        const vv = Number(vField[rr]?.[cc])
        if (!Number.isFinite(u) || !Number.isFinite(vv)) break

        const mag = Math.hypot(u, vv)
        if (mag < 1e-6) break

        const stepScale = 0.45
        const dx = (u / mag) * stepScale
        const dy = (vv / mag) * stepScale

        const x0 = mapX(cxRaw * sx)
        const y0 = mapY(cyRaw * sy)
        const x1 = mapX((cxRaw + dx) * sx)
        const y1 = mapY((cyRaw + dy) * sy)
        streamX.push(x0, x1)
        streamY.push(y0, y1)

        cxRaw += dx
        cyRaw += dy
        lastDx = dx
        lastDy = dy
      }

      if (lastDx !== 0 || lastDy !== 0) {
        const arrowLen = 2.0
        const wingLen = arrowLen * 0.5
        const angle = Math.atan2(lastDy, lastDx)
        const angle1 = angle - Math.PI / 6
        const angle2 = angle + Math.PI / 6

        const endX = mapX(cxRaw * sx)
        const endY = mapY(cyRaw * sy)
        const xw1 = mapX((cxRaw - wingLen * Math.cos(angle1)) * sx)
        const yw1 = mapY((cyRaw - wingLen * Math.sin(angle1)) * sy)
        const xw2 = mapX((cxRaw - wingLen * Math.cos(angle2)) * sx)
        const yw2 = mapY((cyRaw - wingLen * Math.sin(angle2)) * sy)

        streamX.push(endX, xw1, null)
        streamY.push(endY, yw1, null)
        streamX.push(endX, xw2, null)
        streamY.push(endY, yw2, null)
      } else {
        streamX.push(null)
        streamY.push(null)
      }
    }
  }

  if (!streamX.length) return null

  return {
    x: streamX,
    y: streamY,
    mode: 'lines',
    type: 'scatter',
    line: { color: 'rgba(255, 255, 255, 0.55)', width: 1.5, shape: 'spline', smoothing: 1.3 },
    xaxis: `x${subplotIndex + 1 > 1 ? subplotIndex + 1 : ''}`,
    yaxis: `y${subplotIndex + 1 > 1 ? subplotIndex + 1 : ''}`,
    hoverinfo: 'skip',
    showlegend: false,
    visible: showQuiver.value
  }
}

const loadCurveData = async (r = null, c = null) => {
  if (!sessionId.value) return
  try {
    const params = {}
    if (r !== null && c !== null) {
      params.r = r
      params.c = c
    }
    const res = await api.get(`/predict/${sessionId.value}/curve`, { params })
    if (r !== null && c !== null) {
      curveTitle.value = `点 (${r}, ${c}) 时序预测`
    } else {
      curveTitle.value = '区域平均 — 时序预测'
    }
    renderCurvePlot(res.data.data)
  } catch (err) {
    console.error('加载曲线失败', err)
  }
}

const loadWarnings = async () => {
  if (!sessionId.value) return
  loadingWarning.value = true
  try {
    const res = await api.post(`/predict/${sessionId.value}/warnings`, {
      thresholds: warningThresholds.value
    })
    warningsData.value = res.data
  } catch (err) {
    console.error('加载异常预警失败', err)
  } finally {
    loadingWarning.value = false
  }
  
  await nextTick()
  // 渲染第一个变量的警告地图作为默认
  if (warningsData.value?.masks) {
    const vars = Object.keys(warningsData.value.masks)
    if (vars.length > 0) {
      currentWarningVar.value = vars[0]
      renderWarningHeatmap(vars[0])
    }
  }
}

const loadCorrelation = async () => {
  if (!sessionId.value) return
  loadingCorr.value = true
  try {
    const res = await api.get(`/predict/${sessionId.value}/correlation`, {
      params: { var1: corrVar1.value, var2: corrVar2.value }
    })
    correlationData.value = res.data
  } catch (err) {
    console.error('加载关联分析失败', err)
  } finally {
    loadingCorr.value = false
  }
  
  await nextTick()
  if (correlationData.value) {
    renderCorrelationHeatmap()
    // 强制触发一次 resize 确保自适应容器
    const el = document.getElementById('correlation-chart')
    if (el) Plotly.Plots.resize(el)
  }
}

const onStepSliderInput = () => {
  stopPlay()
  scheduleStepDataLoad(currentStep.value, false)
}

const jumpForecastStep = (delta: number) => {
  stopPlay()
  const next = Math.max(0, Math.min(totalSteps.value - 1, currentStep.value + Number(delta)))
  currentStep.value = next
  scheduleStepDataLoad(next, true)
}

const applyCompareVarChange = async () => {
  if (currentStepDataCache) renderCompareSpatial(currentStepDataCache)
}

const togglePlay = () => {
  if (isPlaying.value) {
    stopPlay()
    return
  }

  isPlaying.value = true
  playToken += 1
  const localToken = playToken
  if (currentStep.value >= totalSteps.value - 1) currentStep.value = 0
  scheduleStepDataLoad(currentStep.value, true)
  const playLoop = async () => {
    if (!isPlaying.value || localToken !== playToken) return
    if (currentStep.value >= totalSteps.value - 1) {
      stopPlay()
      return
    }
    currentStep.value += 1
    await loadStepData(currentStep.value)
    if (!isPlaying.value || localToken !== playToken) return
    playTimer = setTimeout(playLoop, Math.max(100, 1200 / playbackSpeed.value))
  }
  playTimer = setTimeout(playLoop, Math.max(100, 1200 / playbackSpeed.value))
}

const onSpeedChange = () => {
  if (isPlaying.value) {
    stopPlay()
    togglePlay()
  }
}

const stopPlay = () => {
  isPlaying.value = false
  playToken += 1
  if (playTimer) clearTimeout(playTimer)
  playTimer = null
}

function getVariableRenderStyle(varNameRaw: string) {
  const key = String(varNameRaw || '').toUpperCase().trim()

  if (key === 'ADT') {
    return {
      displayName: 'ADT',
      title: '海面高度异常',
      unit: 'm',
      colorscale: 'RdBu',
      diverging: true,
      quantileLow: 0.02,
      quantileHigh: 0.98,
      upsampleScale: 4,
      smoothPasses: 1,
      smoothMode: 'normal',
      validWeightThreshold: 0.55,
      isMask: false
    }
  }

  if (key === 'TRUE_MASK' || key === 'PRED_MASK') {
    return {
      displayName: key,
      title: key === 'TRUE_MASK' ? '真实掩膜' : '预测掩膜',
      unit: 'class',
      colorscale: [
        [0.0, '#0f172a'],
        [0.33, '#0f172a'],
        [0.34, '#ef4444'],
        [0.66, '#ef4444'],
        [0.67, '#3b82f6'],
        [1.0, '#3b82f6']
      ],
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      isMask: true,
      maskMax: 2
    }
  }

  if (key === 'CYCLONIC') {
    return {
      displayName: 'CYCLONIC',
      title: '气旋数量',
      unit: 'count',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      validWeightThreshold: 1,
      isMask: false,
      lineColor: '#ef4444',
      markerColor: '#f87171'
    }
  }

  if (key === 'ANTICYCLONIC') {
    return {
      displayName: 'ANTICYCLONIC',
      title: '反气旋数量',
      unit: 'count',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      validWeightThreshold: 1,
      isMask: false,
      lineColor: '#3b82f6',
      markerColor: '#60a5fa'
    }
  }

  if (key === 'TOTAL') {
    return {
      displayName: 'TOTAL',
      title: '总涡旋数量',
      unit: 'count',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      validWeightThreshold: 1,
      isMask: false,
      lineColor: '#06b6d4',
      markerColor: '#22d3ee'
    }
  }

  if (key === 'TRUE_CYCLONIC') {
    return {
      displayName: 'TRUE_CYCLONIC',
      title: '真实气旋数量',
      unit: 'count',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      validWeightThreshold: 1,
      isMask: false,
      lineColor: '#fb7185',
      markerColor: '#fda4af'
    }
  }

  if (key === 'TRUE_ANTICYCLONIC') {
    return {
      displayName: 'TRUE_ANTICYCLONIC',
      title: '真实反气旋数量',
      unit: 'count',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      validWeightThreshold: 1,
      isMask: false,
      lineColor: '#93c5fd',
      markerColor: '#bfdbfe'
    }
  }

  if (key === 'TRUE_TOTAL') {
    return {
      displayName: 'TRUE_TOTAL',
      title: '真实总涡旋数量',
      unit: 'count',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      smoothMode: 'normal',
      validWeightThreshold: 1,
      isMask: false,
      lineColor: '#facc15',
      markerColor: '#fde68a'
    }
  }

  if (key.includes('SSUV')) {
    return {
      displayName: 'SSUV',
      title: '合成流速',
      unit: 'm/s',
      colorscale: 'Cividis',
      diverging: false,
      quantileLow: 0.02,
      quantileHigh: 0.98,
      upsampleScale: 4,
      smoothPasses: 1,
      smoothMode: 'normal',
      validWeightThreshold: 0.55,
      isMask: false
    }
  }

  if (key.includes('SST')) {
    return {
      displayName: 'SST',
      title: '海表温度',
      unit: 'degC',
      colorscale: 'Turbo',
      diverging: false,
      quantileLow: 0.01,
      quantileHigh: 0.99,
      upsampleScale: 8,
      smoothPasses: 2,
      smoothMode: 'strong',
      validWeightThreshold: 0.3,
      isMask: false
    }
  }

  if (key.includes('SSS')) {
    return {
      displayName: 'SSS',
      title: '海水盐度',
      unit: 'psu',
      colorscale: 'Viridis',
      diverging: false,
      quantileLow: 0.01,
      quantileHigh: 0.99,
      upsampleScale: 6,
      smoothPasses: 1,
      smoothMode: 'normal',
      validWeightThreshold: 0.45,
      isMask: false
    }
  }

  if (key.includes('SSU')) {
    return {
      displayName: 'SSU',
      title: '东向流速',
      unit: 'm/s',
      colorscale: 'RdBu',
      diverging: true,
      quantileLow: 0.02,
      quantileHigh: 0.98,
      upsampleScale: 4,
      smoothPasses: 1,
      smoothMode: 'normal',
      validWeightThreshold: 0.55,
      isMask: false
    }
  }

  if (key.includes('SSV')) {
    return {
      displayName: 'SSV',
      title: '北向流速',
      unit: 'm/s',
      colorscale: 'RdBu',
      diverging: true,
      quantileLow: 0.02,
      quantileHigh: 0.98,
      upsampleScale: 4,
      smoothPasses: 1,
      smoothMode: 'normal',
      validWeightThreshold: 0.55,
      isMask: false
    }
  }

  if (key.endsWith('_VALID')) {
    return {
      displayName: key,
      title: '有效掩膜',
      unit: '0/1',
      colorscale: [
        [0, 'rgba(0,0,0,0)'],
        [1, '#06b6d4']
      ],
      diverging: false,
      quantileLow: 0,
      quantileHigh: 1,
      upsampleScale: 1,
      smoothPasses: 0,
      isMask: true
    }
  }

  return {
    displayName: key || 'VAR',
    title: '变量场',
    unit: '',
    colorscale: 'Turbo',
    diverging: false,
    quantileLow: 0.02,
    quantileHigh: 0.98,
    upsampleScale: 5,
    smoothPasses: 1,
    smoothMode: 'normal',
    validWeightThreshold: 0.55,
    isMask: false
  }
}

const mergeVectorFieldsSpatial = (items) => {
  if (!Array.isArray(items)) return []

  const ssu = items.find((it) => String(it?.var || '').toUpperCase() === 'SSU')
  const ssv = items.find((it) => String(it?.var || '').toUpperCase() === 'SSV')
  const mergedBase = items.filter((it) => {
    const k = String(it?.var || '').toUpperCase()
    return k !== 'SSU' && k !== 'SSV'
  })

  if (!ssu || !ssv || !Array.isArray(ssu.data) || !Array.isArray(ssv.data)) {
    return items
  }

  const rows = Math.min(ssu.data.length, ssv.data.length)
  const cols = rows > 0 ? Math.min(ssu.data[0]?.length || 0, ssv.data[0]?.length || 0) : 0
  const speed = Array.from({ length: rows }, (_, r) =>
    Array.from({ length: cols }, (_, c) => {
      const u = ssu.data[r]?.[c]
      const v = ssv.data[r]?.[c]
      if (!Number.isFinite(u) || !Number.isFinite(v)) return null
      return Math.sqrt(u * u + v * v)
    })
  )

  return [...mergedBase, { var: 'ssuv', data: speed }]
}

const mergeVectorFieldsCurve = (items) => {
  if (!Array.isArray(items)) return []

  const ssu = items.find((it) => String(it?.var || '').toUpperCase() === 'SSU')
  const ssv = items.find((it) => String(it?.var || '').toUpperCase() === 'SSV')
  const mergedBase = items.filter((it) => {
    const k = String(it?.var || '').toUpperCase()
    return k !== 'SSU' && k !== 'SSV'
  })

  if (!ssu || !ssv || !Array.isArray(ssu.means) || !Array.isArray(ssv.means)) {
    return items
  }

  const n = Math.min(ssu.means.length, ssv.means.length)
  const speedMeans = Array.from({ length: n }, (_, i) => {
    const u = ssu.means[i]
    const v = ssv.means[i]
    if (!Number.isFinite(u) || !Number.isFinite(v)) return null
    return Math.sqrt(u * u + v * v)
  })

  return [...mergedBase, { var: 'ssuv', means: speedMeans }]
}

const fillInternalMissingPoints = (grid, passes = 2, minNeighbors = 5) => {
  if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) return grid

  let source = grid.map((row) => row.slice())
  const rows = source.length
  const cols = source[0].length
  const offsets = [-1, 0, 1]

  for (let p = 0; p < passes; p++) {
    const target = source.map((row) => row.slice())

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (Number.isFinite(source[r][c])) continue

        let sum = 0
        let weightSum = 0
        let validCount = 0

        for (const dr of offsets) {
          for (const dc of offsets) {
            if (dr === 0 && dc === 0) continue

            const rr = r + dr
            const cc = c + dc
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue

            const v = source[rr][cc]
            if (!Number.isFinite(v)) continue

            const dist2 = dr * dr + dc * dc
            const w = dist2 === 1 ? 1 : 0.707
            sum += v * w
            weightSum += w
            validCount += 1
          }
        }

        if (validCount >= minNeighbors && weightSum > 0) {
          target[r][c] = sum / weightSum
        }
      }
    }
    source = target
  }

  return source
}

const upsampleGridBilinear = (grid, scale = 5, minValidWeight = 0.55) => {
  if (!Array.isArray(grid) || grid.length < 2 || !Array.isArray(grid[0]) || grid[0].length < 2 || scale <= 1) {
    return grid
  }

  const rows = grid.length
  const cols = grid[0].length
  const outRows = (rows - 1) * scale + 1
  const outCols = (cols - 1) * scale + 1
  const output = Array.from({ length: outRows }, () => Array(outCols).fill(null))

  for (let r = 0; r < outRows; r++) {
    const srcR = r / scale
    const r0 = Math.floor(srcR)
    const r1 = Math.min(r0 + 1, rows - 1)
    const fr = srcR - r0

    for (let c = 0; c < outCols; c++) {
      const srcC = c / scale
      const c0 = Math.floor(srcC)
      const c1 = Math.min(c0 + 1, cols - 1)
      const fc = srcC - c0

      const q00 = grid[r0][c0]
      const q01 = grid[r0][c1]
      const q10 = grid[r1][c0]
      const q11 = grid[r1][c1]

      const w00 = (1 - fr) * (1 - fc)
      const w01 = (1 - fr) * fc
      const w10 = fr * (1 - fc)
      const w11 = fr * fc

      let weightedSum = 0
      let weightTotal = 0
      let validWeight = 0

      if (Number.isFinite(q00)) { weightedSum += q00 * w00; weightTotal += w00; validWeight += w00; }
      if (Number.isFinite(q01)) { weightedSum += q01 * w01; weightTotal += w01; validWeight += w01; }
      if (Number.isFinite(q10)) { weightedSum += q10 * w10; weightTotal += w10; validWeight += w10; }
      if (Number.isFinite(q11)) { weightedSum += q11 * w11; weightTotal += w11; validWeight += w11; }

      output[r][c] = validWeight >= minValidWeight && weightTotal > 0 ? weightedSum / weightTotal : null
    }
  }

  return output
}

const smoothGridMasked = (grid, passes = 1, mode = 'normal') => {
  if (!Array.isArray(grid) || !grid.length || !Array.isArray(grid[0])) return grid

  let source = grid
  const rows = grid.length
  const cols = grid[0].length
  const kernel = mode === 'strong'
    ? [
        [1, 4, 6, 4, 1],
        [4, 16, 24, 16, 4],
        [6, 24, 36, 24, 6],
        [4, 16, 24, 16, 4],
        [1, 4, 6, 4, 1]
      ]
    : [
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
      ]
  const radius = Math.floor(kernel.length / 2)

  for (let p = 0; p < passes; p++) {
    const target = Array.from({ length: rows }, () => Array(cols).fill(null))

    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        if (!Number.isFinite(source[r][c])) {
          target[r][c] = null
          continue
        }

        let sum = 0
        let wsum = 0

        for (let kr = -radius; kr <= radius; kr++) {
          for (let kc = -radius; kc <= radius; kc++) {
            const rr = r + kr
            const cc = c + kc
            if (rr < 0 || rr >= rows || cc < 0 || cc >= cols) continue
            const v = source[rr][cc]
            if (!Number.isFinite(v)) continue

            const w = kernel[kr + radius][kc + radius]
            sum += v * w
            wsum += w
          }
        }

        target[r][c] = wsum > 0 ? sum / wsum : source[r][c]
      }
    }
    source = target
  }

  return source
}

const getAutoColorRange = (grid, style) => {
  if (style?.isMask) {
    const maskMax = Number.isFinite(style?.maskMax) ? style.maskMax : 1
    return { zauto: false, zmin: 0, zmax: maskMax }
  }

  const flat = grid.flat().filter((n) => Number.isFinite(n))
  if (!flat.length) return { zauto: true }

  const sorted = [...flat].sort((a, b) => a - b)
  const pick = (q) => sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)))]

  let low = pick(Number.isFinite(style?.quantileLow) ? style.quantileLow : 0.02)
  let high = pick(Number.isFinite(style?.quantileHigh) ? style.quantileHigh : 0.98)
  if (!Number.isFinite(low) || !Number.isFinite(high) || low === high) {
    low = sorted[0]
    high = sorted[sorted.length - 1]
  }

  const currentRange = style?.diverging
    ? (() => { const absMax = Math.max(Math.abs(low), Math.abs(high)); return { zauto: false, zmin: -absMax, zmax: absMax, zmid: 0 } })()
    : { zauto: false, zmin: low, zmax: high }

  const cacheKey = style?.displayName || 'VAR'
  const cached = colorRangeCache.get(cacheKey)
  if (!cached) {
    colorRangeCache.set(cacheKey, currentRange)
    return currentRange
  }
  return cached
}

const renderSpatialPlot = (stepData) => {
  const container = document.getElementById('spatial-chart')
  if (!container) return

  const plotItems = mergeVectorFieldsSpatial(stepData?.data || [])
  const plots = []
  spatialStreamlineTraceIndices = []
  const annotations = []

  const N = Math.max(1, plotItems.length)

  plotItems.forEach((v, index) => {
    const style = getVariableRenderStyle(v.var)

    const holeFilledGrid = style.isMask ? v.data : fillInternalMissingPoints(v.data, 2, 5)
    const upsampledGrid = upsampleGridBilinear(holeFilledGrid, style.upsampleScale, style.validWeightThreshold)
    const smoothedGrid = style.smoothPasses > 0 ? smoothGridMasked(upsampledGrid, style.smoothPasses, style.smoothMode) : upsampledGrid
    const colorRange = getAutoColorRange(smoothedGrid, style)

    const blockWidth = 1.0 / N
    const xStart = index * blockWidth
    const xEnd = xStart + blockWidth * 0.80
    const cbX = xEnd + 0.015
    const titleX = (xStart + xEnd) / 2

    const rawRows = Array.isArray(v?.data) ? v.data.length : 0
    const rawCols = rawRows > 0 && Array.isArray(v?.data?.[0]) ? v.data[0].length : 0
    const renderRows = Array.isArray(smoothedGrid) ? smoothedGrid.length : 0
    const renderCols = renderRows > 0 && Array.isArray(smoothedGrid[0]) ? smoothedGrid[0].length : 0

    // Provide explicit x/y coordinate arrays if possible
    let xCoords, yCoords
    if (geoLon.value.length > 1 && geoLat.value.length > 1) {
      const lons = geoLon.value
      const lats = geoLat.value
      xCoords = Array.from({ length: renderCols }, (_, i) => lons[0] + (i / Math.max(1, renderCols - 1)) * (lons[lons.length - 1] - lons[0]))
      yCoords = Array.from({ length: renderRows }, (_, i) => lats[0] + (i / Math.max(1, renderRows - 1)) * (lats[lats.length - 1] - lats[0]))
    }

    plots.push({
      z: smoothedGrid,
      ...(xCoords && yCoords ? { x: xCoords, y: yCoords } : {}),
      type: 'heatmap',
      zsmooth: style.isMask ? false : 'best',
      colorscale: style.colorscale,
      ...colorRange,
      hoverongaps: false,
      showscale: true,
      colorbar: {
        len: 0.55,
        thickness: 8,
        x: cbX,
        y: 0.5,
        tickfont: { color: '#94a3b8', size: 10, family: 'JetBrains Mono, monospace' },
        outlinewidth: 0,
        bgcolor: 'rgba(3,7,18,0.6)',
        bordercolor: 'rgba(6,182,212,0.3)',
        borderwidth: 1,
        title: { text: style.unit ? `${style.displayName}<br>${style.unit}` : style.displayName, font: { color: '#06b6d4', size: 10 } }
      },
      xaxis: `x${index + 1 > 1 ? index + 1 : ''}`,
      yaxis: `y${index + 1 > 1 ? index + 1 : ''}`,
      meta: {
        rawRows,
        rawCols,
        renderRows,
        renderCols,
        varName: String(v?.var || ''),
      },
      name: v.var,
      hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4', font: { color: '#06b6d4', family: 'JetBrains Mono, monospace' } }
    })

    // Keep streamline as an independent trace and only toggle visibility,
    // avoiding expensive full heatmap re-render on switch.
    if (String(v.var).toUpperCase() === 'SSUV') {
      const trace = buildStreamlineTrace(stepData, index, renderRows, renderCols, xCoords, yCoords)
      if (trace) {
        const traceIndex = plots.length
        plots.push(trace)
        spatialStreamlineTraceIndices.push(traceIndex)
      }
    }

    annotations.push({
      text: `[ ${style.displayName} ] ${style.title}`,
      x: titleX,
      y: 1.02,
      xref: 'paper', yref: 'paper',
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: { color: '#06b6d4', size: 12, family: 'Orbitron, sans-serif' }
    })
  })

  // Set explicit manual domains to dynamically place all charts in a single row
  const layout = {
    ...getChartLayoutBase(''),
    margin: { l: 20, r: 20, t: 50, b: 20 },
    // Keep user zoom/pan state stable across Plotly.react updates.
    uirevision: 'forecast-spatial-v1',
    annotations
  }

  for (let i = 0; i < plotItems.length; i++) {
    const ax = i === 0 ? '' : (i + 1)
    const blockWidth = 1.0 / N
    const xStart = i * blockWidth
    const xEnd = xStart + blockWidth * 0.80

    layout[`xaxis${ax}`] = { domain: [xStart, xEnd], anchor: `y${ax}`, showticklabels: true, tickfont: { color: '#475569', size: 9 }, gridcolor: 'rgba(30, 41, 59, 0.5)', zeroline: false }
    layout[`yaxis${ax}`] = { scaleanchor: `x${ax}`, scaleratio: 1, domain: [0.05, 1.0], anchor: `x${ax}`, autorange: 'reversed', showticklabels: true, tickfont: { color: '#475569', size: 9 }, gridcolor: 'rgba(30, 41, 59, 0.5)', zeroline: false }
  }

  Plotly.react(container, plots, layout, { responsive: true, displayModeBar: false })

  // Add click event listener to update curve chart
  container.on('plotly_click', (data) => {
    if (data.points && data.points.length > 0) {
      const pt = data.points[0]
      const meta = pt?.fullData?.meta || {}
      const rr = Number(meta?.rawRows || 0)
      const rc = Number(meta?.rawCols || 0)
      const pr = Number(meta?.renderRows || 0)
      const pc = Number(meta?.renderCols || 0)

      let c = 0
      let r = 0

      if (geoLon.value.length > 1 && geoLat.value.length > 1) {
        const lons = geoLon.value
        const lats = geoLat.value
        const xRatio = (Number(pt.x) - lons[0]) / (lons[lons.length - 1] - lons[0] || 1)
        const yRatio = (Number(pt.y) - lats[0]) / (lats[lats.length - 1] - lats[0] || 1)
        c = Math.round(xRatio * (rc - 1))
        r = Math.round(yRatio * (rr - 1))
      } else {
        c = Math.round(Number(pt.x))
        r = Math.round(Number(pt.y))
        // Click coordinates are on rendered/upsampled grid; map them back to raw indices for backend.
        if (rr > 1 && rc > 1 && pr > 1 && pc > 1) {
          c = Math.round((Number(pt.x) / (pc - 1)) * (rc - 1))
          r = Math.round((Number(pt.y) / (pr - 1)) * (rr - 1))
        }
      }

      c = Math.max(0, Math.min(rc - 1, c))
      r = Math.max(0, Math.min(rr - 1, r))
      loadCurveData(r, c)
    }
  })
}

const renderCurvePlot = (curveData) => {
  const container = document.getElementById('curve-chart')
  if (!container) return

  const plotItems = mergeVectorFieldsCurve(curveData)
  const plots = []
  const annotations = []
  const N = Math.max(1, plotItems.length)

  plotItems.forEach((v, index) => {
    const hours = Array.from({ length: v.means.length }, (_, i) => (i + 1) * stepHours.value)
    const style = getVariableRenderStyle(v.var)
    const titleX = (index * (1.0 / N)) + (1.0 / N) * 0.5

    plots.push({
      x: hours, y: v.means, type: 'scatter', mode: 'lines',
      line: { color: style.displayName === 'SSUV' ? '#8b5cf6' : '#06b6d4', width: 2, shape: 'spline', smoothing: 1.3 },
      name: v.var,
      xaxis: `x${index + 1 > 1 ? index + 1 : ''}`, yaxis: `y${index + 1 > 1 ? index + 1 : ''}`,
      showlegend: false, hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4' }
    })

    annotations.push({
      text: `[ ${style.displayName} ] ${style.title}`,
      x: titleX,
      y: 1.06,
      xref: 'paper', yref: 'paper',
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: { color: '#06b6d4', size: 12, family: 'Orbitron, sans-serif' }
    })
  })

  const layout = {
    ...getChartLayoutBase(''),
    annotations,
    margin: { l: 40, r: 20, t: 50, b: 30 },
    // Keep user zoom/pan state stable across Plotly.react updates.
    uirevision: 'forecast-curve-v1',
    grid: { rows: 1, columns: N, pattern: 'independent', xgap: 0.08 }
  }

  for (let i = 0; i < plotItems.length; i++) {
    const ax = i === 0 ? '' : (i + 1)
    
    // Calculate custom y-axis range to reduce visual volatility
    const validMeans = plotItems[i].means.filter(v => v !== null && v !== undefined && !isNaN(v))
    let yMin = 0
    let yMax = 1
    if (validMeans.length > 0) {
      const minVal = Math.min(...validMeans)
      const maxVal = Math.max(...validMeans)
      const valRange = maxVal - minVal
      // Expand the range by 150% (75% on each side) to flatten out the curve
      const padding = valRange > 0 ? valRange * 0.75 : Math.abs(minVal) * 0.1 || 0.1
      yMin = minVal - padding
      yMax = maxVal + padding
    }

    layout[`xaxis${ax}`] = { gridcolor: 'rgba(30, 41, 59, 0.5)', tickfont: { color: '#64748b' } }
    layout[`yaxis${ax}`] = { gridcolor: 'rgba(30, 41, 59, 0.5)', tickfont: { color: '#64748b' }, range: [yMin, yMax] }
  }

  Plotly.react(container, plots, layout, { responsive: true, displayModeBar: false })
}

const jsonGridToNum = (g) => {
  if (!Array.isArray(g)) return []
  return g.map((row) =>
    (Array.isArray(row) ? row : []).map((v) =>
      v === null || v === undefined || Number.isNaN(Number(v)) ? NaN : Number(v)
    )
  )
}

const _renderSingleCompareHeatmap = (elId, block, rowVar) => {
  const container = document.getElementById(elId)
  if (!container) return

  const holeFilled = fillInternalMissingPoints(block.z, 2, 5)
  const upsampled = upsampleGridBilinear(holeFilled, block.scale.upsampleScale, block.scale.validWeightThreshold)
  const smoothed =
    block.scale.smoothPasses > 0 ? smoothGridMasked(upsampled, block.scale.smoothPasses, block.scale.smoothMode) : upsampled
  const colorRange = getAutoColorRange(smoothed, block.scale)
    const rawRows = block.z.length
    const rawCols = rawRows > 0 && Array.isArray(block.z[0]) ? block.z[0].length : 0
    const renderRows = smoothed.length
    const renderCols = renderRows > 0 && Array.isArray(smoothed[0]) ? smoothed[0].length : 0

  let xCoords, yCoords
  if (geoLon.value.length > 1 && geoLat.value.length > 1) {
    const lons = geoLon.value
    const lats = geoLat.value
    xCoords = Array.from({ length: renderCols }, (_, i) => lons[0] + (i / Math.max(1, renderCols - 1)) * (lons[lons.length - 1] - lons[0]))
    yCoords = Array.from({ length: renderRows }, (_, i) => lats[0] + (i / Math.max(1, renderRows - 1)) * (lats[lats.length - 1] - lats[0]))
  }

  const trace = {
    z: smoothed,
    ...(xCoords && yCoords ? { x: xCoords, y: yCoords } : {}),
      type: 'heatmap',
    zsmooth: 'best',
    colorscale: block.scale.colorscale,
    ...colorRange,
    hoverongaps: false,
    showscale: true,
    colorbar: {
      title: { text: block.scale.unit ? `${block.scale.displayName}<br>${block.scale.unit}` : block.key, font: { size: 9, color: '#94a3b8' } },
      len: 0.82,
      thickness: 10,
      x: 1.02,
      xanchor: 'left',
      y: 0.5,
      yanchor: 'middle',
      tickfont: { color: '#94a3b8', size: 8, family: 'JetBrains Mono, monospace' },
      outlinewidth: 0,
      bgcolor: 'rgba(3,7,18,0.55)',
      bordercolor: 'rgba(6,182,212,0.25)',
      borderwidth: 1
    },
    meta: { rawRows, rawCols, renderRows, renderCols, varName: String(rowVar || '') },
    hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4', font: { color: '#06b6d4', family: 'JetBrains Mono, monospace' } }
  }

  const layout = {
    ...getChartLayoutBase(''),
    margin: { l: 36, r: 100, t: 32, b: 36 },
    title: {
      text: block.key,
      font: { color: '#e2e8f0', size: 12, family: 'Orbitron, sans-serif' },
      x: 0.5,
      xanchor: 'center',
      y: 0.98,
      yanchor: 'top'
    },
    xaxis: {
      constrain: 'domain',
      showticklabels: true,
      tickfont: { color: '#64748b', size: 8 },
      gridcolor: 'rgba(30, 41, 59, 0.45)',
      zeroline: false
    },
    yaxis: {
      autorange: 'reversed',
      scaleanchor: 'x',
      scaleratio: renderCols > 0 && renderRows > 0 ? renderCols / renderRows : 1,
      showticklabels: true,
      tickfont: { color: '#64748b', size: 8 },
      gridcolor: 'rgba(30, 41, 59, 0.45)',
      zeroline: false
    },
    uirevision: `forecast-compare-slot-${elId}`
  }

  Plotly.react(container, [trace], layout, { responsive: true, displayModeBar: false })
}

const renderCompareSpatial = (stepData) => {
  if (!stepData?.data?.length) return

  const vi = Math.max(0, Math.min(compareVarIndex.value, stepData.data.length - 1))
  const row = stepData.data[vi]
  if (!row || !('true' in row)) return

  const style = getVariableRenderStyle(row.var)
  const triple = [
    { key: '预测', z: jsonGridToNum(row.data), scale: style },
    { key: '实况', z: jsonGridToNum(row.true), scale: style },
    {
      key: '预报−实况',
      z: jsonGridToNum(row.diff),
      scale: { ...style, displayName: `${style.displayName}_DIFF`, diverging: true, colorscale: 'RdBu_r' }
    }
  ]

  const ids = ['compare-spatial-0', 'compare-spatial-1', 'compare-spatial-2']
  triple.forEach((block, i) => {
    _renderSingleCompareHeatmap(ids[i], block, row.var)
  })
}

const updateVerticalLineOnCurve = () => {
  if (!showCurvePanel.value) return
  const container = document.getElementById('curve-chart')
  if (!container || !hasResult.value) return

  const currentHour = (currentStep.value + 1) * stepHours.value
  const shapes = []

  for (let i = 1; i <= 4; i++) {
    shapes.push({
      type: 'line',
      x0: currentHour, x1: currentHour, y0: 0, y1: 1,
      yref: 'paper', xref: `x${i === 1 ? '' : i}`,
      line: { color: 'rgba(245, 158, 11, 0.8)', width: 2, dash: 'dot' }
    })
  }

  Plotly.relayout(container, { shapes })
}
function disposeForecastTimers() {
  stopPlay()
  if (stepFetchTimer) {
    clearTimeout(stepFetchTimer)
    stepFetchTimer = null
  }
}

watch(showCurvePanel, async (open) => {
  await nextTick()
  if (open && hasResult.value && sessionId.value) {
    try {
      await loadCurveData()
      updateVerticalLineOnCurve()
    } catch {
      // ignore
    }
  }
  resizeForecastCharts(hasResult.value)
})

watch(spatialWorkbenchTab, async (tab) => {
  if (!hasResult.value) return
  await nextTick()
  if (tab === 'forecast') {
    if (currentStepDataCache) renderSpatialPlot(currentStepDataCache)
  } else if (tab === 'compare') {
    if (currentStepDataCache) renderCompareSpatial(currentStepDataCache)
  } else if (tab === 'warning') {
    if (!warningsData.value) loadWarnings()
    else {
      const vars = Object.keys(warningsData.value.masks || {})
      if (vars.length > 0) renderWarningHeatmap(vars[0])
    }
  } else if (tab === 'correlation') {
    if (!correlationData.value) loadCorrelation()
    else renderCorrelationHeatmap()
  }
  resizeForecastCharts(hasResult.value)
})

export function useForecast() {
  return {
    modelPath,
    dataPath,
    datasetInfo,
    maxIndex,
    startIdx,
    loadingInfo,
    predicting,
    hasResult,
    sessionId,
    totalSteps,
    currentStep,
    isPlaying,
    showQuiver,
    playbackSpeed,
    curveTitle,
    stepHours,
    spatialWorkbenchTab,
    compareVarIndex,
    showCurvePanel,
    warningsData,
    loadingWarning,
    warningThresholds,
    currentWarningVar,
    correlationData,
    loadingCorr,
    corrVar1,
    corrVar2,
    loadDefaultDataPath,
    loadDataInfo,
    runPrediction,
    loadStepData,
    jumpForecastStep,
    applyCompareVarChange,
    scheduleStepDataLoad,
    onQuiverToggle,
    onStepSliderInput,
    togglePlay,
    onSpeedChange,
    stopPlay,
    loadCurveData,
    loadWarnings,
    loadCorrelation,
    renderSpatialPlot,
    renderCurvePlot,
    renderWarningHeatmap,
    renderCorrelationHeatmap,
    updateVerticalLineOnCurve,
    disposeForecastTimers,
    resizeForecastCharts
  }
}

export function resizeForecastCharts(hasResult: boolean) {
  if (!hasResult) return
  const ids = [
    'spatial-chart',
    'curve-chart',
    'compare-spatial-0',
    'compare-spatial-1',
    'compare-spatial-2',
    'warning-spatial-chart',
    'correlation-chart'
  ]
  for (const id of ids) {
    const el = document.getElementById(id)
    if (el) Plotly.Plots.resize(el)
  }
}
