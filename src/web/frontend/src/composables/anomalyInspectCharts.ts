// @ts-nocheck
/** Plotly 渲染：与 app1.vue 异常模块一致，供 useAnomaly 调用。 */
import Plotly from 'plotly.js-dist-min'
import { getChartLayoutBase } from './plotly/chartLayout'

export function plotlyResize(el: HTMLElement | null) {
  if (!el) return
  try {
    Plotly.Plots.resize(el)
  } catch {
    /* noop */
  }
}

export function renderAnomalyMonitorChart(container: HTMLElement | null, monitor: { windNow: number; waveNow: number; windBand: string; waveBand: string }) {
  if (!container) return
  const windBandHigh = Number(String(monitor.windBand).split('-')[1] || 10.5)
  const waveBandHigh = Number(String(monitor.waveBand).split('-')[1] || 2.2)
  const traces = [
    {
      x: ['风速', '波高'],
      y: [monitor.windNow, monitor.waveNow],
      type: 'bar',
      name: '实况',
      marker: { color: '#22d3ee' }
    },
    {
      x: ['风速', '波高'],
      y: [windBandHigh, waveBandHigh],
      type: 'bar',
      name: '基准上限',
      marker: { color: 'rgba(148,163,184,0.45)' }
    }
  ]
  const layout = {
    ...getChartLayoutBase(''),
    title: undefined,
    barmode: 'group',
    margin: { l: 40, r: 12, t: 8, b: 28 },
    xaxis: { tickfont: { color: '#94a3b8', size: 10 } },
    yaxis: { gridcolor: 'rgba(30,41,59,0.5)', tickfont: { color: '#94a3b8', size: 10 } },
    legend: {
      orientation: 'h',
      x: 1,
      xanchor: 'right',
      y: 1.02,
      yanchor: 'bottom',
      bgcolor: 'rgba(2,6,23,0.35)',
      bordercolor: 'rgba(30,41,59,0.6)',
      borderwidth: 1,
      font: { color: '#94a3b8', size: 10 }
    }
  }
  Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false })
}

export async function renderAnomalyWindowChart(
  container: HTMLElement | null,
  points: Array<{
    index: number
    time: string
    label: number
    matched: boolean
    eventHits: string[]
  }>,
  selectedIndex: number,
  onSelect: (sampleIndex: number) => void
) {
  if (!container) return
  if (!points.length) {
    Plotly.purge(container)
    return
  }
  const x = points.map((p) => p.time.slice(11, 19))
  const y = points.map((p) => p.label)
  const markerColor = points.map((p) => {
    if (p.index === selectedIndex) return '#22d3ee'
    return p.matched ? '#34d399' : '#94a3b8'
  })
  const markerSize = points.map((p) => (p.index === selectedIndex ? 11 : 8))
  await Plotly.react(
    container,
    [
      {
        x,
        y,
        type: 'scatter',
        mode: 'lines+markers',
        name: '24h窗口样本',
        line: { color: 'rgba(148,163,184,0.65)', width: 1.5, shape: 'spline' },
        marker: { color: markerColor, size: markerSize },
        customdata: points.map((p) => [p.index, p.time, (p.eventHits || []).join(', ')]),
        hovertemplate: 'index=%{customdata[0]}<br>time=%{customdata[1]}<br>label=%{y}<br>events=%{customdata[2]}<extra></extra>'
      }
    ],
    {
      ...getChartLayoutBase(''),
      title: undefined,
      margin: { l: 36, r: 12, t: 8, b: 22 },
      xaxis: { tickfont: { color: '#94a3b8', size: 9 }, showgrid: false },
      yaxis: { title: 'label', range: [-0.15, 1.15], dtick: 1, gridcolor: 'rgba(30,41,59,0.5)', tickfont: { color: '#94a3b8' } },
      showlegend: false
    },
    { responsive: true, displayModeBar: false }
  )
  if (typeof (container as any).removeAllListeners === 'function') {
    ;(container as any).removeAllListeners('plotly_click')
  }
  if (typeof (container as any).on === 'function') {
    ;(container as any).on('plotly_click', (ev: any) => {
      const pointNumber = ev?.points?.[0]?.pointNumber
      if (!Number.isFinite(pointNumber)) return
      const row = points[Number(pointNumber)]
      if (!row) return
      onSelect(row.index)
    })
  }
}

function maskByValid(grid: number[][], valid: number[][]) {
  if (!Array.isArray(grid) || !Array.isArray(valid)) return grid || []
  return grid.map((row, r) =>
    (row || []).map((v, c) => {
      const m = valid?.[r]?.[c]
      return Number.isFinite(m) && m >= 0.5 ? v : null
    })
  )
}

function buildBoundaryFromValid(valid: number[][]) {
  if (!Array.isArray(valid) || !valid.length || !Array.isArray(valid[0])) return []
  const rows = valid.length
  const cols = valid[0].length
  const out = Array.from({ length: rows }, () => Array(cols).fill(0))
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const cur = Number(valid?.[r]?.[c]) >= 0.5
      if (!cur) continue
      const up = r > 0 ? Number(valid?.[r - 1]?.[c]) >= 0.5 : false
      const down = r < rows - 1 ? Number(valid?.[r + 1]?.[c]) >= 0.5 : false
      const left = c > 0 ? Number(valid?.[r]?.[c - 1]) >= 0.5 : false
      const right = c < cols - 1 ? Number(valid?.[r]?.[c + 1]) >= 0.5 : false
      if (!(up && down && left && right)) out[r][c] = 1
    }
  }
  return out
}

function quantileRange(grid: (number | null)[][], qLow = 0.02, qHigh = 0.98) {
  const flat = (grid || []).flat().filter((v) => Number.isFinite(v)) as number[]
  if (!flat.length) return { zmin: 0, zmax: 1 }
  const sorted = [...flat].sort((a, b) => a - b)
  const pick = (q: number) => sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)))]
  let zmin = pick(qLow)
  let zmax = pick(qHigh)
  if (!Number.isFinite(zmin) || !Number.isFinite(zmax) || zmin === zmax) {
    zmin = sorted[0]
    zmax = sorted[sorted.length - 1]
  }
  return { zmin, zmax }
}

export function renderAnomalyOceanMaps(snap: Record<string, unknown> | null) {
  const windEl = document.getElementById('anomaly-wind-map')
  const waveEl = document.getElementById('anomaly-wave-map')
  if (!windEl || !waveEl) return
  if (!snap) {
    Plotly.purge(windEl)
    Plotly.purge(waveEl)
    return
  }
  const windMasked = maskByValid(snap.wind_speed as number[][], snap.wind_valid as number[][])
  const waveMasked = maskByValid(snap.wave_swh as number[][], snap.wave_valid as number[][])
  const windBoundary = buildBoundaryFromValid(snap.wind_valid as number[][])
  const waveBoundary = buildBoundaryFromValid(snap.wave_valid as number[][])
  const windRange = quantileRange(windMasked, 0.02, 0.98)
  const waveRange = quantileRange(waveMasked, 0.02, 0.98)

  Plotly.react(
    windEl,
    [
      {
        z: windMasked,
        type: 'heatmap',
        zsmooth: 'best',
        colorscale: [
          [0.0, '#0b2f6b'],
          [0.45, '#7aaad6'],
          [0.5, '#f5f7fb'],
          [0.75, '#f09063'],
          [1.0, '#c93b2c']
        ],
        zmin: windRange.zmin,
        zmax: windRange.zmax,
        colorbar: { title: { text: 'wind m/s' }, tickfont: { color: '#94a3b8', size: 10 } },
        hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4' }
      },
      {
        z: windBoundary,
        type: 'contour',
        contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
        line: { color: '#0a1228', width: 2.2 },
        showscale: false,
        hoverinfo: 'skip'
      },
      {
        z: windBoundary,
        type: 'contour',
        contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
        line: { color: 'rgba(148,163,184,0.6)', width: 0.9 },
        showscale: false,
        hoverinfo: 'skip'
      }
    ],
    {
      ...getChartLayoutBase('风速空间分布'),
      margin: { l: 40, r: 20, t: 36, b: 30 },
      xaxis: { showgrid: false, zeroline: false },
      yaxis: { showgrid: false, zeroline: false, autorange: 'reversed' },
      showlegend: false
    },
    { responsive: true, displayModeBar: false }
  )

  Plotly.react(
    waveEl,
    [
      {
        z: waveMasked,
        type: 'heatmap',
        zsmooth: 'best',
        colorscale: [
          [0.0, '#3b0f70'],
          [0.25, '#3f4da1'],
          [0.5, '#2fbf9b'],
          [0.75, '#9fd95e'],
          [1.0, '#f4e74f']
        ],
        zmin: waveRange.zmin,
        zmax: waveRange.zmax,
        colorbar: { title: { text: 'swh m' }, tickfont: { color: '#94a3b8', size: 10 } },
        hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4' }
      },
      {
        z: waveBoundary,
        type: 'contour',
        contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
        line: { color: '#0a1228', width: 2.2 },
        showscale: false,
        hoverinfo: 'skip'
      },
      {
        z: waveBoundary,
        type: 'contour',
        contours: { start: 0.5, end: 0.5, size: 1, coloring: 'none' },
        line: { color: 'rgba(148,163,184,0.6)', width: 0.9 },
        showscale: false,
        hoverinfo: 'skip'
      }
    ],
    {
      ...getChartLayoutBase('波高空间分布'),
      margin: { l: 40, r: 20, t: 36, b: 30 },
      xaxis: { showgrid: false, zeroline: false },
      yaxis: { showgrid: false, zeroline: false, autorange: 'reversed' },
      showlegend: false
    },
    { responsive: true, displayModeBar: false }
  )
}

export function renderAnomalyTimelineChart(
  rows: Array<{
    time: string
    amplitude: number
    duration: number
    labelSignal: number
    matched: boolean
  }>
) {
  const container = document.getElementById('anomaly-timeline-chart')
  if (!container) return
  if (!rows.length) {
    Plotly.react(
      container,
      [],
      {
        ...getChartLayoutBase('异常事件幅度与持续时长'),
        margin: { l: 40, r: 20, t: 42, b: 40 },
        annotations: [
          {
            text: '当前窗口暂无数据，请拖动回溯窗口',
            x: 0.5,
            y: 0.5,
            xref: 'paper',
            yref: 'paper',
            showarrow: false,
            font: { color: '#94a3b8', size: 12, family: 'JetBrains Mono, monospace' }
          }
        ],
        xaxis: { visible: false },
        yaxis: { visible: false }
      },
      { responsive: true, displayModeBar: false }
    )
    return
  }
  const x = rows.map((r) => r.time)
  const amplitude = rows.map((r) => Number(r.amplitude.toFixed(2)))
  const duration = rows.map((r) => r.duration)
  const labelRaw = rows.map((r) => (Number(r.labelSignal) === 1 ? 1 : 0))
  const markerColor = rows.map((r) => (r.matched ? '#34d399' : '#f59e0b'))
  const ampMin = Math.min(...amplitude)
  const ampMax = Math.max(...amplitude)
  const ampPad = Math.max(0.12, (ampMax - ampMin) * 0.25)
  const labelBand = Math.max(0.5, ampMax - ampMin + ampPad * 0.5)
  const labelOnAmp = labelRaw.map((v) => ampMin - ampPad * 0.25 + v * labelBand)
  const durMax = Math.max(3, ...duration)
  const traces = [
    {
      x,
      y: amplitude,
      type: 'scatter',
      mode: 'lines+markers',
      name: '异常幅度(x)',
      line: { color: '#22d3ee', width: 2, shape: 'spline', smoothing: 0.45 },
      marker: { color: markerColor, size: 6 }
    },
    {
      x,
      y: duration,
      type: 'bar',
      name: '持续时长(h)',
      marker: { color: 'rgba(139,92,246,0.45)' },
      yaxis: 'y2'
    },
    {
      x,
      y: labelOnAmp,
      type: 'scatter',
      mode: 'lines+markers',
      name: 'label(0/1)',
      line: { color: '#e2e8f0', width: 1.4, dash: 'dot' },
      marker: { color: '#e2e8f0', size: 5 }
    }
  ]
  const layout = {
    ...getChartLayoutBase('异常事件幅度与持续时长'),
    margin: { l: 50, r: 52, t: 42, b: 55 },
    xaxis: { tickangle: -35, tickfont: { color: '#64748b', size: 9 } },
    yaxis: {
      title: 'real amplitude(x)',
      range: [ampMin - ampPad * 0.45, ampMax + labelBand + ampPad * 0.35],
      gridcolor: 'rgba(30,41,59,0.5)',
      tickfont: { color: '#94a3b8' }
    },
    yaxis2: {
      title: 'duration(h)',
      range: [0, durMax + 1],
      overlaying: 'y',
      side: 'right',
      tickfont: { color: '#a78bfa' }
    },
    legend: { orientation: 'h', x: 0, y: 1.12, font: { color: '#94a3b8', size: 10 } }
  }
  Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false })
}

export function renderAnomalyRiskMapChart(
  anomalyData: { latest_timestamp?: number } | null,
  riskScore: number,
  snap: Record<string, unknown> | null
) {
  const container = document.getElementById('anomaly-riskmap-chart')
  if (!container || !anomalyData) return

  const safeNumber = (v: unknown) => {
    const n = Number(v)
    return Number.isFinite(n) ? n : NaN
  }
  const percentile = (arr: number[], q: number) => {
    if (!arr.length) return 0
    const sorted = [...arr].sort((a, b) => a - b)
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)))
    return sorted[idx]
  }
  const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

  let z = [
    [Math.max(10, riskScore - 28), Math.max(15, riskScore - 16)],
    [Math.max(20, riskScore - 8), Math.min(99, riskScore + 6)]
  ]
  let confidence = [
    [55, 60],
    [60, 68]
  ]
  let validCounts = [
    [0, 0],
    [0, 0]
  ]

  if (snap && Array.isArray(snap.wind_speed) && Array.isArray(snap.wave_swh)) {
    const wind = snap.wind_speed as number[][]
    const wave = snap.wave_swh as number[][]
    const windValid = Array.isArray(snap.wind_valid) ? (snap.wind_valid as number[][]) : []
    const waveValid = Array.isArray(snap.wave_valid) ? (snap.wave_valid as number[][]) : []
    const rows = wind.length
    const cols = rows > 0 && Array.isArray(wind[0]) ? wind[0].length : 0
    if (rows > 1 && cols > 1) {
      const windVals: number[] = []
      const waveVals: number[] = []
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const wv = safeNumber(wind?.[r]?.[c])
          const hv = safeNumber(wave?.[r]?.[c])
          const wOk = Number(windValid?.[r]?.[c]) >= 0.5 || !Array.isArray(windValid) || !windValid.length
          const hOk = Number(waveValid?.[r]?.[c]) >= 0.5 || !Array.isArray(waveValid) || !waveValid.length
          if (Number.isFinite(wv) && wOk) windVals.push(wv)
          if (Number.isFinite(hv) && hOk) waveVals.push(hv)
        }
      }
      const windP05 = percentile(windVals, 0.05)
      const windP50 = percentile(windVals, 0.5)
      const windP95 = percentile(windVals, 0.95)
      const waveP05 = percentile(waveVals, 0.05)
      const waveP50 = percentile(waveVals, 0.5)
      const waveP95 = percentile(waveVals, 0.95)
      const windSpan = Math.max(1e-6, Math.max(windP95 - windP50, windP50 - windP05))
      const waveSpan = Math.max(1e-6, Math.max(waveP95 - waveP50, waveP50 - waveP05))
      const rCut = Math.floor(rows / 2)
      const cCut = Math.floor(cols / 2)
      const regions = [
        { r0: 0, r1: rCut, c0: 0, c1: cCut },
        { r0: 0, r1: rCut, c0: cCut, c1: cols },
        { r0: rCut, r1: rows, c0: 0, c1: cCut },
        { r0: rCut, r1: rows, c0: cCut, c1: cols }
      ]
      const regionRawScores: number[] = []
      const regionConfidence: number[] = []
      const regionValidCounts: number[] = []
      for (const reg of regions) {
        let riskSum = 0
        let validN = 0
        let eligibleN = 0
        let windValidN = 0
        let waveValidN = 0
        for (let r = reg.r0; r < reg.r1; r++) {
          for (let c = reg.c0; c < reg.c1; c++) {
            const wv = safeNumber(wind?.[r]?.[c])
            const hv = safeNumber(wave?.[r]?.[c])
            const wMask = Number(windValid?.[r]?.[c])
            const hMask = Number(waveValid?.[r]?.[c])
            const wOk = wMask >= 0.15 || !Array.isArray(windValid) || !windValid.length
            const hOk = hMask >= 0.15 || !Array.isArray(waveValid) || !waveValid.length
            const windUsable = Number.isFinite(wv) && wOk
            const waveUsable = Number.isFinite(hv) && hOk
            if (windUsable || waveUsable) eligibleN += 1
            if (!windUsable && !waveUsable) continue
            if (windUsable) windValidN += 1
            if (waveUsable) waveValidN += 1
            const windNorm = windUsable ? clamp(Math.abs(wv - windP50) / windSpan, 0, 1) : null
            const waveNorm = waveUsable ? clamp(Math.abs(hv - waveP50) / waveSpan, 0, 1) : null
            const cellRisk =
              windNorm !== null && waveNorm !== null
                ? 0.62 * windNorm + 0.38 * waveNorm
                : windNorm !== null
                  ? windNorm
                  : waveNorm
            riskSum += cellRisk as number
            validN += 1
          }
        }
        const rawScore = validN > 0 ? riskSum / validN : 0
        const coverage = eligibleN > 0 ? validN / eligibleN : 0
        const sampleStrength = clamp(Math.log10(validN + 1) / 2.4, 0, 1)
        const channelBalance =
          Math.max(windValidN, waveValidN) > 0 ? Math.min(windValidN, waveValidN) / Math.max(windValidN, waveValidN) : 0
        const conf = 100 * (0.55 * sampleStrength + 0.1 * channelBalance + 0.35 * coverage)
        regionRawScores.push(rawScore)
        regionConfidence.push(conf)
        regionValidCounts.push(validN)
      }
      const rawMin = Math.min(...regionRawScores)
      const rawMax = Math.max(...regionRawScores)
      const rawSpan = Math.max(1e-6, rawMax - rawMin)
      const anchor = Number.isFinite(Number(riskScore)) ? Number(riskScore) : 50
      const regionScores = regionRawScores.map((s) => {
        const norm = (s - rawMin) / rawSpan
        const regional = 25 + 70 * norm
        return clamp(anchor * 0.55 + regional * 0.45, 0, 100)
      })
      const adjustedConfidence = regionConfidence.map((v) => clamp(v, 0, 100))
      z = [
        [regionScores[0], regionScores[1]],
        [regionScores[2], regionScores[3]]
      ]
      confidence = [
        [adjustedConfidence[0], adjustedConfidence[1]],
        [adjustedConfidence[2], adjustedConfidence[3]]
      ]
      validCounts = [
        [regionValidCounts[0], regionValidCounts[1]],
        [regionValidCounts[2], regionValidCounts[3]]
      ]
    }
  }

  const text = z.map((row, r) => row.map((v, c) => `${Number(v).toFixed(0)}分<br>可信${Number(confidence[r][c]).toFixed(0)}% | N=${validCounts[r][c]}`))

  const latestTs = Number(anomalyData?.latest_timestamp || -1)
  const dataLagMin = latestTs > 0 ? Math.max(0, Math.round((Date.now() / 1000 - latestTs) / 60)) : -1
  const lagText =
    dataLagMin >= 0
      ? dataLagMin >= 1440
        ? `数据时效滞后 ${Math.floor(dataLagMin / 1440)} 天 ${Math.floor((dataLagMin % 1440) / 60)} 小时`
        : dataLagMin >= 60
          ? `数据时效滞后 ${Math.floor(dataLagMin / 60)} 小时 ${dataLagMin % 60} 分钟`
          : `数据时效滞后 ${dataLagMin} 分钟`
      : '数据时效未知'

  const trace = {
    z,
    x: ['黄海北部', '黄海中部'],
    y: ['渤海湾', '黄海南部'],
    type: 'heatmap',
    colorscale: [
      [0.0, '#1d4ed8'],
      [0.35, '#facc15'],
      [0.65, '#fb923c'],
      [1.0, '#f43f5e']
    ],
    zmin: 0,
    zmax: 100,
    colorbar: { title: '风险分', tickfont: { color: '#94a3b8', size: 10 } },
    text,
    texttemplate: '%{text}',
    textfont: { color: '#e2e8f0', size: 12 },
    hovertemplate: '区域: %{y} / %{x}<br>%{text}<extra></extra>'
  }
  const layout = {
    ...getChartLayoutBase('海洋灾害风险区域分布'),
    margin: { l: 60, r: 40, t: 42, b: 45 },
    xaxis: { tickfont: { color: '#94a3b8' } },
    yaxis: { tickfont: { color: '#94a3b8' } },
    annotations: [
      {
        xref: 'paper',
        yref: 'paper',
        x: 1,
        y: 1.13,
        showarrow: false,
        align: 'right',
        text: lagText,
        font: { color: '#94a3b8', size: 10, family: 'JetBrains Mono, monospace' }
      }
    ]
  }
  Plotly.react(container, [trace], layout, { responsive: true, displayModeBar: false })
}

export function resizeAnomalyInspectCharts() {
  ;[
    'anomaly-monitor-chart',
    'anomaly-window-chart',
    'anomaly-timeline-chart',
    'anomaly-riskmap-chart',
    'anomaly-wind-map',
    'anomaly-wave-map'
  ].forEach((id) => {
    const el = document.getElementById(id)
    plotlyResize(el)
  })
}
