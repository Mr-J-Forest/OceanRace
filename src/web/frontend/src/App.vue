<template>
  <div class="h-screen w-screen flex flex-col bg-tech-bg text-slate-300 font-sans overflow-hidden">
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
      <ManagerPanel v-else-if="activeModule === 'manager'" />
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
import { ref, onMounted, onUnmounted, watch, markRaw } from 'vue'
import {
  Activity,
  Waves,
  Compass,
  ShieldAlert,
  LayoutDashboard
} from 'lucide-vue-next'
import { useForecast, resizeForecastCharts } from './composables/useForecast'
import { useEddy, resizeEddyChart, disposeEddyTimers } from './composables/useEddy'
import { resizeAnomalyCharts } from './composables/useAnomaly'
import ForecastPanel from './components/panels/ForecastPanel.vue'
import EddyPanel from './components/panels/EddyPanel.vue'
import ManagerPanel from './components/panels/ManagerPanel.vue'
import AnomalyPanel from './components/panels/AnomalyPanel.vue'

const modules = [
  { key: 'manager', label: '管理驾驶舱', icon: markRaw(LayoutDashboard) },
  { key: 'eddy', label: '涡旋检测', icon: markRaw(Compass) },
  { key: 'forecast', label: '要素预测', icon: markRaw(Waves) },
  { key: 'anomaly', label: '异常检测', icon: markRaw(ShieldAlert) }
]

const activeModule = ref('forecast')
const currentTime = ref(new Date().toISOString().substring(11, 19))

const forecast = useForecast()
const eddy = useEddy()

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
</style>
