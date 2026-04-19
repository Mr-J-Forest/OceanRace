<template>
<section class="h-full flex gap-6 min-h-0">
        <!-- Left Panel: Controls (Fixed Width) -->
        <aside class="w-[360px] glass-panel flex flex-col h-full shrink-0 min-h-0">
          <transition name="fade">
            <div v-if="loadingInfo || predicting" class="absolute inset-0 z-50 bg-tech-bg/90 backdrop-blur-md flex flex-col items-center justify-center">
              <div class="radar-scan"></div>
              <p class="mt-4 font-mono text-tech-cyan animate-pulse">
                {{ predicting ? 'EXECUTING PREDICTION ENGINE...' : 'ANALYZING DATASET...' }}
              </p>
            </div>
          </transition>

          <div class="flex-1 overflow-y-auto custom-scrollbar p-5 flex flex-col gap-6 min-h-0">
            <div>
              <h2 class="panel-title"><Terminal class="w-5 h-5 text-tech-cyan" /> COMMAND CENTER</h2>
              <div class="space-y-4 mt-6">
                <div class="space-y-2">
                  <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest flex justify-between">
                    <span>Model Checkpoint</span><span class="text-tech-cyan">.pt</span>
                  </label>
                  <div class="relative group">
                    <Cpu class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-hover:text-tech-cyan transition-colors" />
                    <input type="text" v-model="modelPath" class="tech-input pl-9" />
                  </div>
                </div>

                <div class="space-y-2">
                  <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest flex justify-between">
                    <span>Target Dataset</span><span class="text-tech-cyan">.nc</span>
                  </label>
                  <div class="relative group">
                    <Database class="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500 group-hover:text-tech-cyan transition-colors" />
                    <input type="text" v-model="dataPath" class="tech-input pl-9" />
                  </div>
                  <button class="tech-btn ghost-btn w-full mt-3 flex items-center justify-center gap-2 group" @click="loadDataInfo">
                    <Search class="w-4 h-4 group-hover:scale-110 transition-transform" /> VERIFY DATASET
                  </button>
                </div>
              </div>
            </div>

            <transition name="slide-down">
              <div v-if="datasetInfo" class="p-4 rounded-xl bg-slate-900/60 border border-tech-cyan/20 mt-auto">
                <div class="flex items-center gap-2 mb-3 text-tech-cyan font-mono text-xs border-b border-tech-cyan/20 pb-2">
                  <CheckCircle2 class="w-4 h-4" />
                  <span>DATASET_LINKED</span>
                </div>
                <pre class="text-[10px] font-mono text-slate-400 overflow-x-auto custom-scrollbar pb-2 mb-4 leading-relaxed">{{ datasetInfo }}</pre>
                
                <div class="space-y-3">
                  <label class="text-xs font-mono text-slate-300 flex justify-between items-center">
                    <span>Start Offset</span>
                    <span class="px-2 py-0.5 rounded bg-tech-cyan/10 text-tech-cyan border border-tech-cyan/30">Idx: {{ startIdx }}</span>
                  </label>
                  <input type="range" v-model.number="startIdx" min="0" :max="maxIndex" class="tech-slider" />
                </div>

                <button class="tech-btn primary-btn w-full mt-6 flex items-center justify-center gap-2 text-sm" @click="runPrediction">
                  <Zap class="w-4 h-4 animate-pulse" /> START FORECAST
                </button>
              </div>
            </transition>
          </div>
        </aside>

        <!-- Right: 主图区可占满高度；折线图单独折叠 -->
        <main class="flex-1 flex flex-col gap-3 min-w-0 min-h-0 overflow-hidden">
          <div class="glass-panel flex flex-col flex-1 min-h-0 relative group overflow-hidden">
            <div class="absolute inset-0 bg-gradient-to-b from-tech-cyan/5 to-transparent pointer-events-none"></div>
            
            <div class="px-4 py-2.5 border-b border-slate-800/80 shrink-0 relative z-10 bg-tech-panel/40 backdrop-blur-sm flex flex-wrap items-center gap-x-4 gap-y-2">
              <div class="flex items-center gap-2 shrink-0">
                <MapIcon class="w-5 h-5 text-tech-cyan" />
                <h2 class="font-display text-lg tracking-widest text-white m-0">空间演变展示</h2>
              </div>

              <label
                v-if="hasResult && spatialWorkbenchTab === 'forecast'"
                class="flex items-center gap-2 shrink-0 cursor-pointer select-none rounded-lg border border-slate-600/80 bg-slate-900/50 px-2.5 py-1.5 hover:border-tech-cyan/40 transition-colors"
              >
                <input type="checkbox" v-model="showCurvePanel" class="w-3.5 h-3.5 accent-tech-cyan" />
                <span class="text-[10px] font-mono text-slate-300 tracking-wide">时序折线图</span>
              </label>
              
              <div v-if="hasResult" class="flex items-center gap-3 flex-1 min-w-[200px] max-w-xl">
                <button 
                  type="button"
                  class="w-7 h-7 shrink-0 rounded flex items-center justify-center transition-all"
                  :class="isPlaying ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500 hover:text-slate-900' : 'bg-tech-cyan/10 text-tech-cyan border border-tech-cyan/30 hover:bg-tech-cyan hover:text-slate-900'"
                  @click="togglePlay"
                  title="播放/暂停"
                >
                  <Pause v-if="isPlaying" class="w-3 h-3 fill-current" />
                  <Play v-else class="w-3 h-3 fill-current ml-0.5" />
                </button>
                
                <select v-model="playbackSpeed" @change="onSpeedChange" class="h-7 px-1 rounded bg-slate-900/60 border border-tech-cyan/20 text-[10px] font-mono text-tech-cyan focus:outline-none cursor-pointer shrink-0">
                  <option :value="0.5">0.5x</option>
                  <option :value="1.0">1.0x</option>
                  <option :value="2.0">2.0x</option>
                </select>
                
                <div class="flex-1 flex items-center gap-2 min-w-0">
                  <span class="text-[10px] font-mono text-slate-500 shrink-0">T+0</span>
                  <input
                    type="range"
                    v-model.number="currentStep"
                    min="0"
                    :max="totalSteps - 1"
                    @input="onStepSliderInput"
                    class="tech-slider h-1 flex-1 min-w-0 cursor-pointer"
                  />
                  <span class="text-[10px] font-mono text-slate-500 shrink-0">T+{{ Math.round(totalSteps * stepHours) }}h</span>
                </div>
              </div>

              <div v-if="hasResult" class="flex items-center gap-1.5 flex-wrap shrink-0">
                <button type="button" class="px-2 py-0.5 rounded text-[10px] font-mono border border-slate-600 text-slate-400 hover:border-tech-cyan hover:text-tech-cyan" @click="jumpForecastStep(-6)">−6步</button>
                <button type="button" class="px-2 py-0.5 rounded text-[10px] font-mono border border-slate-600 text-slate-400 hover:border-tech-cyan hover:text-tech-cyan" @click="jumpForecastStep(-1)">−1步</button>
                <button type="button" class="px-2 py-0.5 rounded text-[10px] font-mono border border-slate-600 text-slate-400 hover:border-tech-cyan hover:text-tech-cyan" @click="jumpForecastStep(1)">+1步</button>
                <button type="button" class="px-2 py-0.5 rounded text-[10px] font-mono border border-slate-600 text-slate-400 hover:border-tech-cyan hover:text-tech-cyan" @click="jumpForecastStep(6)">+6步</button>
                <div class="px-2.5 py-1 rounded-full bg-tech-cyan/10 border border-tech-cyan/40 text-tech-cyan font-mono text-[10px] flex items-center gap-1.5">
                  <span class="w-1.5 h-1.5 rounded-full bg-tech-cyan animate-pulse"></span>
                  T+{{ Math.round((currentStep + 1) * stepHours) }}h
                </div>
              </div>
            </div>

            <div v-if="hasResult" class="px-4 pt-2 pb-1.5 flex flex-wrap gap-1.5 border-b border-slate-800/40 shrink-0 justify-between items-center">
              <div class="flex gap-1.5 flex-wrap">
                <button
                  v-for="t in workbenchTabs"
                  :key="t.id"
                  type="button"
                  class="px-3 py-1.5 rounded-lg text-[11px] font-mono transition-all border"
                  :class="spatialWorkbenchTab === t.id ? 'bg-tech-cyan/15 border-tech-cyan/50 text-tech-cyan' : 'border-transparent text-slate-500 hover:text-slate-300'"
                  @click="spatialWorkbenchTab = t.id"
                >{{ t.label }}</button>
              </div>

              <!-- Streamline Toggle moved to tab bar to avoid overlapping the chart -->
              <div v-if="spatialWorkbenchTab === 'forecast'" class="flex items-center gap-2 bg-slate-900/40 border border-tech-cyan/20 px-2.5 py-1 rounded">
                <input type="checkbox" id="streamline-toggle" v-model="showQuiver" @change="onQuiverToggle" class="w-3.5 h-3.5 accent-tech-cyan cursor-pointer" />
                <label for="streamline-toggle" class="text-[10px] font-mono text-tech-cyan cursor-pointer select-none tracking-wider">洋流流线</label>
              </div>
            </div>

            <!-- 工作台内容：统一占满剩余高度 -->
            <div class="flex-1 min-h-0 flex flex-col relative z-10 overflow-hidden">
              <div v-if="!hasResult" class="flex-1 flex flex-col items-center justify-center text-slate-500">
                <div class="radar-scan opacity-30"></div>
                <p class="font-mono text-xs tracking-[0.2em] mt-6">等待后端解析数据</p>
              </div>

              <!-- 预报色斑 -->
              <div v-show="hasResult && spatialWorkbenchTab === 'forecast'" class="flex-1 min-h-0 flex flex-col relative p-2">
                <div id="spatial-chart" class="flex-1 min-h-[280px] w-full h-full"></div>
              </div>

              <!-- 实况·偏差：三列等分，每列内正方形绘图区 -->
              <div
                v-show="hasResult && spatialWorkbenchTab === 'compare'"
                class="flex-1 min-h-0 flex flex-col p-2 gap-2 overflow-hidden"
              >
                <div class="flex flex-wrap items-center gap-2 text-[11px] font-mono text-slate-400 shrink-0">
                  <span>对比变量</span>
                  <select v-model.number="compareVarIndex" @change="onCompareVarChange" class="bg-slate-900/80 border border-tech-cyan/30 rounded px-2 py-1 text-tech-cyan text-xs">
                    <option v-for="(vn, i) in compareVarOptions" :key="i" :value="i">{{ vn }}</option>
                  </select>
                  <span class="text-slate-500 text-[10px]">三幅独立比例尺，格点近似正方形</span>
                </div>
                <div class="flex-1 min-h-0 grid grid-cols-3 gap-2 md:gap-3 items-start">
                  <div class="min-h-0 flex flex-col items-center w-full">
                    <span class="text-[10px] font-mono text-slate-500 mb-1 shrink-0">预测</span>
                    <div class="w-full aspect-square min-h-[200px] max-h-[min(72vh,900px)] max-w-full">
                      <div id="compare-spatial-0" class="w-full h-full min-h-[180px]"></div>
                    </div>
                  </div>
                  <div class="min-h-0 flex flex-col items-center w-full">
                    <span class="text-[10px] font-mono text-slate-500 mb-1 shrink-0">实况</span>
                    <div class="w-full aspect-square min-h-[200px] max-h-[min(72vh,900px)] max-w-full">
                      <div id="compare-spatial-1" class="w-full h-full min-h-[180px]"></div>
                    </div>
                  </div>
                  <div class="min-h-0 flex flex-col items-center w-full">
                    <span class="text-[10px] font-mono text-slate-500 mb-1 shrink-0">预报−实况</span>
                    <div class="w-full aspect-square min-h-[200px] max-h-[min(72vh,900px)] max-w-full">
                      <div id="compare-spatial-2" class="w-full h-full min-h-[180px]"></div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- 异常预警 -->
              <div
                v-show="hasResult && spatialWorkbenchTab === 'warning'"
                class="flex-1 min-h-0 flex flex-col p-2 gap-3 overflow-hidden"
              >
                <!-- Threshold Configuration -->
                <div class="flex flex-wrap items-center gap-4 text-xs font-mono text-slate-400 shrink-0 bg-slate-900/60 p-2 rounded border border-tech-cyan/20">
                  <div class="flex items-center gap-2">
                    <span class="text-tech-cyan">SST阈值(℃)</span>
                    <input type="number" step="0.1" v-model.number="warningThresholds.SST" class="w-16 bg-slate-800 border border-slate-600 rounded px-1 py-0.5 text-slate-200 outline-none focus:border-tech-cyan" />
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-tech-cyan">SSS阈值(psu)</span>
                    <input type="number" step="0.1" v-model.number="warningThresholds.SSS" class="w-16 bg-slate-800 border border-slate-600 rounded px-1 py-0.5 text-slate-200 outline-none focus:border-tech-cyan" />
                  </div>
                  <div class="flex items-center gap-2">
                    <span class="text-tech-cyan">SSUV阈值(m/s)</span>
                    <input type="number" step="0.1" v-model.number="warningThresholds.SSUV" class="w-16 bg-slate-800 border border-slate-600 rounded px-1 py-0.5 text-slate-200 outline-none focus:border-tech-cyan" />
                  </div>
                  <button class="tech-btn primary-btn !py-1 !px-3 text-xs" @click="loadWarnings">
                    <Activity class="w-3 h-3 inline-block mr-1" /> 运行诊断
                  </button>
                </div>

                <!-- Main Content: Layout with Map and List -->
                <div v-if="loadingWarning" class="flex-1 flex flex-col items-center justify-center text-tech-cyan">
                  <div class="w-8 h-8 rounded-full border-2 border-tech-cyan border-t-transparent animate-spin mb-4"></div>
                  <span class="font-mono text-xs animate-pulse">正在扫描全时段数据并识别异常事件...</span>
                </div>
                <div v-else-if="warningsData" class="flex-1 min-h-0 flex flex-col md:flex-row gap-3 overflow-hidden">
                  <!-- Left: Map -->
                  <div class="flex-1 min-w-0 flex flex-col relative border border-slate-800 rounded bg-slate-900/40">
                    <div class="absolute top-2 left-2 z-10 flex gap-1">
                      <button 
                        v-for="v in Object.keys(warningsData.masks || {})" 
                        :key="v"
                        class="px-2 py-1 rounded text-[10px] font-mono transition-colors border"
                        :class="currentWarningVar === v ? 'bg-tech-cyan/20 text-tech-cyan border-tech-cyan' : 'bg-slate-800/80 text-slate-400 border-slate-700 hover:text-slate-200'"
                        @click="currentWarningVar = v; renderWarningHeatmap(v)"
                      >
                        {{ v }}
                      </button>
                    </div>
                    <div id="warning-spatial-chart" class="flex-1 min-h-0 w-full"></div>
                  </div>
                  <!-- Right: Event List -->
                  <div class="md:w-[340px] shrink-0 flex flex-col gap-2 bg-slate-900/40 border border-slate-800 rounded overflow-hidden">
                    <div class="px-3 py-2 border-b border-slate-800 bg-slate-800/50 text-xs font-mono text-slate-300 flex items-center justify-between">
                      <div class="flex items-center gap-1">
                        <TriangleAlert class="w-3.5 h-3.5 text-amber-500" />
                        <span>预警事件记录</span>
                      </div>
                      <span class="text-tech-cyan">{{ warningsData.events?.length || 0 }} 项</span>
                    </div>
                    <div class="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-2">
                      <div v-if="!warningsData.events?.length" class="text-center text-slate-500 text-xs mt-4">未检测到异常事件</div>
                      <div v-for="(ev, idx) in warningsData.events" :key="idx" class="p-2 rounded border border-slate-800 bg-slate-950/50 hover:border-slate-700 transition-colors">
                        <div class="flex items-center justify-between mb-1.5">
                          <span class="font-bold text-xs" :class="ev.level === '重度' ? 'text-red-400' : (ev.level === '中度' ? 'text-amber-500' : 'text-yellow-400')">
                            [{{ ev.level }}] {{ ev.type }}
                          </span>
                          <span class="text-[10px] font-mono text-slate-500">Lon: {{ ev.lon }}, Lat: {{ ev.lat }}</span>
                        </div>
                        <div class="text-[11px] text-slate-300 mb-1.5 leading-relaxed">{{ ev.desc }} (幅度: {{ ev.amplitude }})</div>
                        <div class="text-[10px] text-slate-400 bg-slate-800/40 p-1.5 rounded border-l-2" :class="ev.level === '重度' ? 'border-red-400' : (ev.level === '中度' ? 'border-amber-500' : 'border-yellow-400')">
                          处置参考：{{ ev.suggestion }}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <!-- 关联分析 -->
              <div
                v-show="hasResult && spatialWorkbenchTab === 'correlation'"
                class="flex-1 min-h-0 flex flex-col p-2 gap-3 overflow-hidden"
              >
                <!-- Controls -->
                <div class="flex flex-wrap items-center gap-4 text-xs font-mono text-slate-400 shrink-0 bg-slate-900/60 p-2 rounded border border-tech-cyan/20">
                  <div class="flex items-center gap-2">
                    <span>变量 1:</span>
                    <select v-model="corrVar1" class="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-tech-cyan outline-none focus:border-tech-cyan">
                      <option v-for="v in compareVarOptions" :key="v" :value="v">{{ v }}</option>
                    </select>
                  </div>
                  <div class="flex items-center gap-2">
                    <span>变量 2:</span>
                    <select v-model="corrVar2" class="bg-slate-800 border border-slate-600 rounded px-2 py-1 text-tech-cyan outline-none focus:border-tech-cyan">
                      <option v-for="v in compareVarOptions" :key="v" :value="v">{{ v }}</option>
                    </select>
                  </div>
                  <button class="tech-btn primary-btn !py-1 !px-3 text-xs" @click="loadCorrelation">
                    <Link class="w-3 h-3 inline-block mr-1" /> 执行分析
                  </button>
                </div>

                <div v-if="loadingCorr" class="flex-1 flex flex-col items-center justify-center text-tech-cyan">
                  <div class="w-8 h-8 rounded-full border-2 border-tech-cyan border-t-transparent animate-spin mb-4"></div>
                  <span class="font-mono text-xs animate-pulse">正在计算 Pearson 时空相关系数矩阵...</span>
                </div>
                <div v-else-if="correlationData" class="flex-1 min-h-0 flex flex-col md:flex-row gap-3 overflow-hidden">
                  <!-- Left: Map -->
                  <div class="flex-[2] min-w-0 border border-slate-800 rounded bg-slate-900/40 relative">
                    <div id="correlation-chart" class="absolute inset-0 w-full h-full p-2"></div>
                  </div>
                  <!-- Right: Analysis Text -->
                  <div class="flex-1 shrink-0 flex flex-col gap-3 min-w-[280px]">
                    <div class="bg-slate-900/40 border border-slate-800 rounded p-4 flex flex-col gap-3 h-full overflow-y-auto custom-scrollbar">
                      <h3 class="text-sm font-display text-tech-cyan flex items-center gap-2">
                        <Activity class="w-4 h-4" /> 关联影响指标
                      </h3>
                      <div class="grid grid-cols-3 gap-2">
                        <div class="bg-slate-950 p-2 rounded border border-slate-800">
                          <div class="text-[10px] text-slate-500 font-mono mb-1">全局平均相关</div>
                          <div class="text-base text-white font-mono">{{ correlationData.mean_corr }}</div>
                        </div>
                        <div class="bg-slate-950 p-2 rounded border border-slate-800">
                          <div class="text-[10px] text-slate-500 font-mono mb-1">平均绝对关联度</div>
                          <div class="text-base text-white font-mono">{{ correlationData.mean_abs_corr }}</div>
                        </div>
                        <div class="bg-slate-950 p-2 rounded border border-slate-800">
                          <div class="text-[10px] text-slate-500 font-mono mb-1">高强关联占比</div>
                          <div class="text-base text-white font-mono">{{ (correlationData.high_corr_ratio * 100).toFixed(1) }}%</div>
                        </div>
                      </div>
                      <div class="mt-2 text-xs text-slate-300 leading-relaxed bg-slate-800/40 p-3 rounded border-l-2 border-tech-cyan">
                        {{ correlationData.analysis_text }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- 折线图：可选显示，固定高度条带，仅在预报色斑(forecast)选项卡下展示 -->
          <div
            v-show="hasResult && showCurvePanel && spatialWorkbenchTab === 'forecast'"
            class="glass-panel shrink-0 flex flex-col border-t border-slate-800/60 overflow-hidden"
            style="height: clamp(240px, 32vh, 400px);"
          >
            <div class="px-4 py-2 border-b border-slate-800/80 shrink-0 bg-tech-panel/40 backdrop-blur-sm flex items-center justify-between gap-2">
              <div class="flex items-center gap-2 min-w-0">
                <LineChart class="w-5 h-5 text-tech-cyan shrink-0" />
                <h2 class="font-display text-sm md:text-base tracking-widest text-white m-0 truncate">{{ curveTitle }}</h2>
              </div>
              <span class="text-[9px] font-mono text-slate-500 shrink-0">点击预报色斑格点可切换单点序列</span>
            </div>
            <div class="flex-1 relative min-h-0 p-1">
              <div id="curve-chart" class="absolute inset-0"></div>
            </div>
          </div>
        </main>
      </section>
</template>

<script setup>
import { computed, watch, nextTick } from 'vue'
import {
  Terminal,
  Cpu,
  Database,
  Search,
  CheckCircle2,
  Zap,
  Map as MapIcon,
  Play,
  Pause,
  LineChart,
  TriangleAlert,
  Link,
  Activity
} from 'lucide-vue-next'
import { useForecast, resizeForecastCharts } from '../../composables/useForecast'

const {
  modelPath,
  dataPath,
  datasetInfo,
  maxIndex,
  startIdx,
  loadingInfo,
  predicting,
  hasResult,
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
  loadWarnings,
  loadCorrelation,
  renderWarningHeatmap
} = useForecast()

const workbenchTabs = [
  { id: 'forecast', label: '预报色斑' },
  { id: 'compare', label: '实况·偏差' },
  { id: 'warning', label: '异常预警' },
  { id: 'correlation', label: '关联分析' }
]

const compareVarOptions = computed(() => ['SST', 'SSS', 'SSU', 'SSV'])

const onCompareVarChange = () => {
  applyCompareVarChange()
}

watch([spatialWorkbenchTab, showCurvePanel], async () => {
  await nextTick()
  resizeForecastCharts(hasResult.value)
})
</script>
