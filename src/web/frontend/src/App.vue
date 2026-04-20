<template>
  <div
    class="h-screen w-screen flex flex-col bg-tech-bg text-slate-300 font-sans overflow-hidden"
    :class="{ 'demo-recording-mode': demoRunning && demoRecordingMode }"
  >
    <!-- Header -->
    <header class="h-16 border-b border-tech-border/50 bg-tech-panel/80 backdrop-blur-md flex items-center justify-between px-6 shrink-0 z-10 relative">
      <div class="absolute bottom-0 left-0 w-full h-[1px] bg-gradient-to-r from-transparent via-tech-cyan to-transparent opacity-50"></div>

      <div class="flex items-center gap-3">
        <div class="relative flex items-center justify-center w-8 h-8">
          <span class="absolute inset-0 rounded-full border border-tech-cyan/50 animate-[spin_4s_linear_infinite]"></span>
          <span class="pulse-dot"></span>
        </div>
        <h1 class="font-display text-xl font-bold tracking-wider text-white drop-shadow-[0_0_8px_rgba(6,182,212,0.5)]">
          OCEAN<span class="text-tech-cyan">RACE</span>
        </h1>
        <span class="ml-2 px-2 py-0.5 rounded text-[10px] font-mono bg-tech-cyan/10 text-tech-cyan border border-tech-cyan/30">v2.0</span>
      </div>

      <nav class="flex items-center gap-1 bg-slate-900/50 p-1 rounded-lg border border-slate-800" aria-label="模块切换">
        <button
          v-for="m in modules"
          :key="m.key"
          class="relative px-4 py-1.5 text-sm font-medium rounded-md transition-all duration-300 overflow-hidden group"
          :class="activeModule === m.key ? 'text-tech-cyan bg-tech-cyan/10 shadow-[0_0_10px_rgba(6,182,212,0.2)]' : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'"
          @click="switchModule(m.key)"
        >
          <span class="relative z-10 flex items-center gap-2">
            <component :is="m.icon" class="w-4 h-4" />
            {{ m.label }}
          </span>
          <div v-if="activeModule === m.key" class="absolute bottom-0 left-0 w-full h-[2px] bg-tech-cyan shadow-[0_0_8px_rgba(6,182,212,0.8)]"></div>
        </button>
      </nav>

      <div class="flex items-center gap-4 font-mono text-xs">
        <div class="flex items-center gap-2">
          <span
            v-if="demoRunning"
            class="px-2 py-0.5 rounded border border-amber-400/40 bg-amber-400/10 text-amber-300"
          >
            DEMO_RUNNING
          </span>
          <button
            class="px-3 py-1 rounded border text-[11px] transition-colors"
            :class="demoRunning ? 'border-rose-400/50 bg-rose-400/10 text-rose-300 hover:bg-rose-400/20' : 'border-tech-cyan/40 bg-tech-cyan/10 text-tech-cyan hover:bg-tech-cyan/20'"
            @click="toggleDemoMode"
          >
            {{ demoRunning ? '停止演示' : '演示模式' }}
          </button>
          <span v-if="demoRunning && demoStatus" class="max-w-[220px] truncate text-slate-400">
            {{ demoStatus }}
          </span>
        </div>
        <label class="flex items-center gap-1.5 text-slate-400">
          <span>节奏</span>
          <select v-model="demoPace" class="h-6 rounded border border-slate-700 bg-slate-900/80 px-1 text-[10px] text-slate-300">
            <option value="fast">快</option>
            <option value="normal">中</option>
            <option value="slow">慢</option>
          </select>
        </label>
        <label class="flex items-center gap-1.5 text-slate-400">
          <input type="checkbox" v-model="demoLoop" class="accent-tech-cyan" />
          <span>循环</span>
        </label>
        <label class="flex items-center gap-1.5 text-slate-400">
          <input type="checkbox" v-model="demoRecordingMode" class="accent-tech-cyan" />
          <span>录屏友好</span>
        </label>
        <div class="flex items-center gap-2 text-tech-cyan">
          <Activity class="w-4 h-4 animate-pulse" />
          <span>SYS_ONLINE</span>
        </div>
        <div class="px-3 py-1 rounded bg-slate-900/80 border border-slate-700 text-slate-300">
          {{ currentTime }} UTC
        </div>
      </div>
    </header>

    <!-- Main Content -->
    <div class="flex-1 overflow-hidden p-6 relative">
      <ForecastPanel v-if="activeModule === 'forecast'" />
      <EddyPanel v-else-if="activeModule === 'eddy'" />
      <AnomalyPanel v-else-if="activeModule === 'anomaly'" />

      <section v-else class="h-full flex items-center justify-center">
        <div class="glass-panel p-10 max-w-lg text-center relative overflow-hidden group">
          <div class="absolute inset-0 bg-gradient-to-br from-tech-cyan/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <component :is="modules.find((m) => m.key === activeModule)?.icon" class="w-16 h-16 mx-auto text-slate-600 mb-6" />
          <h2 class="font-display text-2xl text-white mb-4 tracking-wider">{{ getModuleLabel(activeModule) }}</h2>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, watch, markRaw, nextTick } from 'vue'
import {
  Activity,
  Waves,
  Compass,
  ShieldAlert
} from 'lucide-vue-next'
import { useForecast, resizeForecastCharts } from './composables/useForecast'
import { useEddy, resizeEddyChart, disposeEddyTimers } from './composables/useEddy'
import { useAnomaly, resizeAnomalyCharts } from './composables/useAnomaly'
import ForecastPanel from './components/panels/ForecastPanel.vue'
import EddyPanel from './components/panels/EddyPanel.vue'
import AnomalyPanel from './components/panels/AnomalyPanel.vue'

const modules = [
  { key: 'eddy', label: '涡旋检测', icon: markRaw(Compass) },
  { key: 'forecast', label: '要素预测', icon: markRaw(Waves) },
  { key: 'anomaly', label: '异常检测', icon: markRaw(ShieldAlert) }
]

const activeModule = ref('forecast')
const currentTime = ref(new Date().toISOString().substring(11, 19))
const demoRunning = ref(false)
const demoStatus = ref('')
const demoPace = ref('normal')
const demoLoop = ref(false)
const demoRecordingMode = ref(true)
let demoToken = 0

const forecast = useForecast()
const eddy = useEddy()
const anomaly = useAnomaly()

watch(activeModule, (moduleKey) => {
  if (moduleKey !== 'forecast') {
    forecast.stopPlay()
  }
})

let clockInterval
const handleResize = () => {
  if (activeModule.value === 'forecast') {
    resizeForecastCharts(forecast.hasResult.value)
  }
  if (activeModule.value === 'eddy') {
    resizeEddyChart()
  }
  if (activeModule.value === 'anomaly') {
    resizeAnomalyCharts()
  }
}

onMounted(() => {
  clockInterval = setInterval(() => {
    currentTime.value = new Date().toISOString().substring(11, 19)
  }, 1000)

  forecast.loadDefaultDataPath()
  eddy.loadEddyDefaults()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  clearInterval(clockInterval)
  forecast.disposeForecastTimers()
  disposeEddyTimers()
  window.removeEventListener('resize', handleResize)
})

const switchModule = (moduleKey) => {
  activeModule.value = moduleKey
}

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms))
const waitDemo = async (token, ms) => {
  ensureDemoActive(token)
  await sleep(ms)
  ensureDemoActive(token)
}

const demoPaceDurationMap = {
  fast: {
    settle: 350,
    eddyAfterPredict: 450,
    eddyTrackStay: 700,
    forecastAfterPredict: 500,
    forecastPlay: 4200,
    tabStayShort: 850,
    tabStayMedium: 1100,
    anomalyViewStay: 1000,
    outroStay: 1200
  },
  normal: {
    settle: 500,
    eddyAfterPredict: 600,
    eddyTrackStay: 900,
    forecastAfterPredict: 700,
    forecastPlay: 6500,
    tabStayShort: 1200,
    tabStayMedium: 1500,
    anomalyViewStay: 1300,
    outroStay: 1800
  },
  slow: {
    settle: 800,
    eddyAfterPredict: 1000,
    eddyTrackStay: 1400,
    forecastAfterPredict: 1200,
    forecastPlay: 9000,
    tabStayShort: 1800,
    tabStayMedium: 2200,
    anomalyViewStay: 2200,
    outroStay: 2600
  }
}

const getDemoDurations = () => demoPaceDurationMap[demoPace.value] || demoPaceDurationMap.normal

const waitFor = async (predicate, timeoutMs = 120000, pollMs = 150) => {
  const start = Date.now()
  while (!predicate()) {
    if (Date.now() - start > timeoutMs) {
      throw new Error('等待超时')
    }
    await sleep(pollMs)
  }
}

const ensureDemoActive = (token) => {
  if (!demoRunning.value || token !== demoToken) {
    throw new Error('演示已停止')
  }
}

const runDemoMode = async (token) => {
  const d = getDemoDurations()
  // Part 1: Eddy
  ensureDemoActive(token)
  demoStatus.value = '切换到涡旋检测'
  switchModule('eddy')
  await nextTick()
  await waitDemo(token, d.settle)

  ensureDemoActive(token)
  demoStatus.value = '加载涡旋数据'
  await eddy.loadEddyDataInfo()
  await waitFor(() => !eddy.eddyLoading.value, 30000)

  ensureDemoActive(token)
  if (eddy.eddyDates.value.length) {
    const idx2000 = eddy.eddyDates.value.findIndex((dateLabel) => String(dateLabel || '').startsWith('2000-'))
    if (idx2000 >= 0) {
      eddy.eddyDayIndex.value = idx2000
      eddy.eddySelectedDate.value = String(eddy.eddyDates.value[idx2000]).slice(0, 10)
    } else if (eddy.eddyDates.value.length > 2) {
      eddy.eddyDayIndex.value = Math.min(1, eddy.eddyDates.value.length - 1)
    }
    demoStatus.value = '执行涡旋预测'
    await eddy.runEddyPrediction()
    await waitFor(() => !eddy.eddyPredicting.value, 120000)
    await waitDemo(token, d.eddyAfterPredict)
  }

  ensureDemoActive(token)
  const centers = Array.isArray(eddy.eddyResult.value?.centers) ? eddy.eddyResult.value.centers : []
  if (centers.length) {
    demoStatus.value = '生成涡旋演变轨迹'
    eddy.eddySelectedCenter.value = centers[0]
    eddy.eddyWorkbenchTab.value = 'track'
    await nextTick()
    await eddy.loadEddyTrack()
    await waitFor(() => !eddy.eddyTrackLoading.value, 90000)
    await waitDemo(token, d.eddyTrackStay)
  }

  // Part 2: Forecast
  ensureDemoActive(token)
  demoStatus.value = '切换到要素预测'
  switchModule('forecast')
  await nextTick()
  await waitDemo(token, d.settle)

  ensureDemoActive(token)
  demoStatus.value = '校验要素数据'
  await forecast.loadDataInfo()
  await waitFor(() => !forecast.loadingInfo.value, 30000)

  ensureDemoActive(token)
  demoStatus.value = '执行72小时预测'
  await forecast.runPrediction()
  await waitFor(() => !forecast.predicting.value, 180000)
  await waitDemo(token, d.forecastAfterPredict)

  ensureDemoActive(token)
  demoStatus.value = '播放预报色斑'
  forecast.playbackSpeed.value = 1
  if (!forecast.isPlaying.value) {
    forecast.togglePlay()
  }
  await waitDemo(token, d.forecastPlay)
  forecast.stopPlay()
  await waitDemo(token, d.settle)

  ensureDemoActive(token)
  demoStatus.value = '切换实况偏差'
  forecast.spatialWorkbenchTab.value = 'compare'
  await nextTick()
  await waitDemo(token, d.tabStayShort)
  forecast.compareVarIndex.value = 2
  forecast.applyCompareVarChange()
  await waitDemo(token, d.tabStayShort)

  ensureDemoActive(token)
  demoStatus.value = '运行异常预警诊断'
  forecast.spatialWorkbenchTab.value = 'warning'
  await nextTick()
  await forecast.loadWarnings()
  await waitFor(() => !forecast.loadingWarning.value, 120000)
  await waitDemo(token, d.tabStayMedium)

  ensureDemoActive(token)
  demoStatus.value = '执行关联分析'
  forecast.spatialWorkbenchTab.value = 'correlation'
  forecast.corrVar1.value = 'SST'
  forecast.corrVar2.value = 'SSU'
  await nextTick()
  await forecast.loadCorrelation()
  await waitFor(() => !forecast.loadingCorr.value, 120000)
  await waitDemo(token, d.tabStayMedium)

  // Part 3: Anomaly
  ensureDemoActive(token)
  demoStatus.value = '切换到异常检测'
  switchModule('anomaly')
  await nextTick()
  await waitDemo(token, d.settle)

  ensureDemoActive(token)
  demoStatus.value = '加载标签与命中统计'
  anomaly.anomalySplit.value = 'test'
  await anomaly.loadAnomalyOverview()
  await waitFor(() => !anomaly.anomalyLoading.value, 180000)
  await waitDemo(token, d.settle)

  ensureDemoActive(token)
  demoStatus.value = '展示实况监测'
  anomaly.anomalyView.value = 'monitor'
  await nextTick()
  await waitDemo(token, d.anomalyViewStay)

  ensureDemoActive(token)
  demoStatus.value = '展示智能识别回溯'
  anomaly.anomalyView.value = 'detect'
  await nextTick()
  if (anomaly.anomalyTracebackMaxStartHour.value > 0) {
    anomaly.anomalyTracebackStartHour.value = Math.max(
      0,
      anomaly.anomalyTracebackMaxStartHour.value - Math.max(24, anomaly.anomalyTracebackStepHours.value * 8)
    )
  }
  await waitDemo(token, d.tabStayMedium)

  ensureDemoActive(token)
  demoStatus.value = '展示台风关联评估'
  anomaly.anomalyView.value = 'typhoon'
  await nextTick()
  await waitDemo(token, d.tabStayMedium)

  ensureDemoActive(token)
  demoStatus.value = '展示分级预警发布'
  anomaly.anomalyView.value = 'warning'
  await nextTick()
  await waitDemo(token, d.outroStay)
}

const stopDemoMode = () => {
  demoRunning.value = false
  demoToken += 1
  demoStatus.value = ''
  forecast.stopPlay()
}

const toggleDemoMode = async () => {
  if (demoRunning.value) {
    stopDemoMode()
    return
  }
  demoRunning.value = true
  demoToken += 1
  const token = demoToken
  try {
    do {
      await runDemoMode(token)
      if (!demoLoop.value) break
      await waitDemo(token, 900)
    } while (demoRunning.value && token === demoToken)
  } catch (err) {
    console.warn('[demo-mode] interrupted', err)
  } finally {
    if (token === demoToken) {
      demoRunning.value = false
      demoStatus.value = ''
      forecast.stopPlay()
    }
  }
}

const getModuleLabel = (moduleKey) => {
  const found = modules.find((m) => m.key === moduleKey)
  return found ? found.label : '模块'
}
</script>

<style scoped>
.custom-scrollbar::-webkit-scrollbar {
  width: 6px;
  height: 6px;
}
.custom-scrollbar::-webkit-scrollbar-track {
  background: rgba(15, 23, 42, 0.5);
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb {
  background: rgba(6, 182, 212, 0.3);
  border-radius: 4px;
}
.custom-scrollbar::-webkit-scrollbar-thumb:hover {
  background: rgba(6, 182, 212, 0.6);
}

.demo-recording-mode,
.demo-recording-mode * {
  cursor: none !important;
}
</style>
