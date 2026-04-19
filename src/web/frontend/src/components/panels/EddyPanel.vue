<template>
<section class="h-full flex gap-6">
        <aside class="w-[400px] glass-panel flex flex-col h-full shrink-0 p-5 gap-4 overflow-y-auto custom-scrollbar relative">
          <transition name="fade">
            <div v-if="eddyLoading || eddyPredicting" class="absolute inset-0 z-40 bg-tech-bg/90 backdrop-blur-md flex flex-col items-center justify-center">
              <div class="radar-scan"></div>
              <p class="mt-4 font-mono text-tech-cyan animate-pulse">
                {{ eddyPredicting ? 'PREDICTING EDDY FIELD...' : 'READING EDDY DATASET...' }}
              </p>
              <div v-if="eddyPredicting" class="mt-4 w-64">
                <div class="h-2 rounded-full bg-slate-800 border border-slate-700 overflow-hidden">
                  <div
                    class="h-full rounded-full bg-gradient-to-r from-cyan-300 to-sky-300 transition-all duration-300"
                    :style="{ width: `${eddyProgress}%` }"
                  ></div>
                </div>
                <div class="mt-2 text-center text-xs font-mono text-slate-300">
                  {{ eddyProgress }}%
                </div>
              </div>
            </div>
          </transition>

          <h2 class="panel-title"><Compass class="w-5 h-5 text-tech-cyan" /> EDDY DETECTION CENTER</h2>

          <div class="space-y-2">
            <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Model Checkpoint</label>
            <input type="text" v-model="eddyModelPath" class="tech-input" />
          </div>

          <div class="space-y-2">
            <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Target Clean NetCDF</label>
            <input type="text" v-model="eddyDataPath" class="tech-input" />
            <button class="tech-btn ghost-btn w-full mt-2 flex items-center justify-center gap-2" @click="loadEddyDataInfo">
              <Search class="w-4 h-4" /> LOAD DAYS
            </button>
          </div>

          <div v-if="eddyDatasetInfo" class="p-3 rounded-lg bg-slate-900/60 border border-tech-cyan/20">
            <pre class="text-[10px] font-mono text-slate-400 whitespace-pre-wrap">{{ eddyDatasetInfo }}</pre>
          </div>

          <div v-if="eddyDates.length > 0" class="p-4 rounded-xl border border-tech-cyan/20 bg-slate-950/40 space-y-3">
            <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Select Date</label>
            <div class="grid grid-cols-[1fr_auto_auto] gap-2 items-center">
              <input
                type="date"
                v-model="eddySelectedDate"
                :min="eddyMinDate"
                :max="eddyMaxDate"
                class="tech-input !h-10"
              />
              <button class="tech-btn ghost-btn !px-3 !py-2 text-xs" @click="shiftEddyDate(-1)">上一天</button>
              <button class="tech-btn ghost-btn !px-3 !py-2 text-xs" @click="shiftEddyDate(1)">下一天</button>
            </div>
            <div class="text-[11px] font-mono text-slate-400 flex items-center justify-between gap-2">
              <span>当前索引: <span class="text-tech-cyan">idx={{ eddyDayIndex }}</span></span>
              <span>可选范围: {{ eddyMinDate }} ~ {{ eddyMaxDate }}</span>
            </div>
            <p v-if="eddyDateHint" class="text-[10px] text-amber-300 font-mono leading-5">
              {{ eddyDateHint }}
            </p>
          </div>

          <button class="tech-btn primary-btn w-full mt-2 flex items-center justify-center gap-2" @click="runEddyPrediction" :disabled="eddyDates.length === 0 || eddyPredicting || !eddySelectedDate">
            <Zap class="w-4 h-4" /> PREDICT SELECTED DAY
          </button>

          <div v-if="eddyPredicting" class="space-y-1">
            <div class="h-1.5 rounded-full bg-slate-800 border border-slate-700 overflow-hidden">
              <div
                class="h-full rounded-full bg-gradient-to-r from-cyan-300 to-sky-300 transition-all duration-300"
                :style="{ width: `${eddyProgress}%` }"
              ></div>
            </div>
            <div class="text-[10px] font-mono text-slate-400 text-right">预测进度 {{ eddyProgress }}%</div>
          </div>

          <div v-if="eddyResult" class="p-4 rounded-xl border border-tech-cyan/25 bg-gradient-to-br from-slate-900/90 to-slate-800/50 text-xs font-mono text-slate-300 space-y-2">
            <div class="flex items-center justify-between text-tech-cyan">
              <span>DATE</span>
              <span>{{ eddyResult.day_label }}</span>
            </div>
            <div class="grid grid-cols-2 gap-2">
              <div class="rounded-lg border border-cyan-400/25 bg-cyan-500/10 px-3 py-2">
                <div class="text-[10px] text-slate-300">Cyclonic</div>
                <div class="text-lg text-cyan-200 mt-0.5">{{ eddyResult.cyclonic_count }}</div>
              </div>
              <div class="rounded-lg border border-rose-400/25 bg-rose-500/10 px-3 py-2">
                <div class="text-[10px] text-slate-300">Anticyclonic</div>
                <div class="text-lg text-rose-200 mt-0.5">{{ eddyResult.anticyclonic_count }}</div>
              </div>
            </div>
            <button class="tech-btn ghost-btn w-full mt-3 flex items-center justify-center gap-2" @click="downloadEddyInfo">
              <Download class="w-4 h-4" /> DOWNLOAD DAY INFO
            </button>
          </div>
        </aside>

        <main class="flex-1 min-w-0 flex flex-col gap-4">
          <div class="px-4 py-2 border-b border-slate-800/80 shrink-0 bg-tech-panel/40 backdrop-blur-sm flex items-center justify-between gap-2">
            <div class="flex gap-2 flex-wrap">
              <button
                type="button"
                class="px-4 py-1.5 rounded-lg text-xs font-mono transition-all border"
                :class="eddyWorkbenchTab === 'forecast' ? 'bg-tech-cyan/15 border-tech-cyan/50 text-tech-cyan' : 'border-transparent text-slate-500 hover:text-slate-300'"
                @click="eddyWorkbenchTab = 'forecast'"
              >实时识别</button>
              <button
                type="button"
                class="px-4 py-1.5 rounded-lg text-xs font-mono transition-all border"
                :class="eddyWorkbenchTab === 'track' ? 'bg-tech-cyan/15 border-tech-cyan/50 text-tech-cyan' : 'border-transparent text-slate-500 hover:text-slate-300'"
                @click="eddyWorkbenchTab = 'track'"
              >演变追踪</button>
            </div>
            <div v-if="eddyResult" class="px-3 py-1 rounded-full border border-tech-cyan/40 bg-tech-cyan/10 text-[11px] font-mono text-tech-cyan">
              {{ eddyResult.day_label }} | idx={{ eddyResult.day_index }}
            </div>
          </div>

          <!-- 实时识别 Tab -->
          <div v-show="eddyWorkbenchTab === 'forecast'" class="flex-1 flex flex-col gap-4 overflow-hidden">
            <div class="glass-panel p-4 flex-1 min-h-0 flex flex-col relative">
              <div class="flex items-center justify-between gap-3 mb-3">
                <div class="flex items-center gap-3">
                  <MapIcon class="w-5 h-5 text-tech-cyan" />
                  <h2 class="font-display text-lg tracking-widest text-white m-0">涡旋空间叠加展示</h2>
                </div>
              </div>

              <!-- 图例与提示 -->
              <div
                v-if="eddyResult"
                class="flex flex-wrap items-center justify-between gap-2 mb-2 shrink-0 pl-0.5"
              >
                <div class="flex items-center gap-2">
                  <span class="px-2 py-1 rounded border border-cyan-400/40 bg-slate-900/90 text-[10px] font-mono text-cyan-200 shadow-sm">蓝线=气旋边界</span>
                  <span class="px-2 py-1 rounded border border-rose-400/40 bg-slate-900/90 text-[10px] font-mono text-rose-200 shadow-sm">红线=反气旋边界</span>
                </div>
                <span class="text-xs font-mono text-slate-500">提示：点击图上涡旋中心(x)可查看全维度属性，并在“演变追踪”面板进行追踪。</span>
              </div>

              <div class="flex-1 relative min-h-0 bg-slate-900/25 rounded-xl border border-slate-700/60 overflow-hidden shadow-[inset_0_0_0_1px_rgba(6,182,212,0.08)] flex">
                <div class="absolute inset-0 pointer-events-none bg-gradient-to-b from-tech-cyan/5 to-transparent"></div>
                <div v-if="!eddyResult" class="absolute inset-0 flex items-center justify-center text-slate-500 font-mono text-sm">
                  先加载数据并选择日期，然后执行预测
                </div>
                <div v-show="!!eddyResult" id="eddy-chart" class="flex-1 min-w-0 min-h-0"></div>

                <!-- 右侧浮窗属性展示 -->
                <transition name="slide-left">
                  <div v-if="eddySelectedCenter" class="w-64 border-l border-slate-700/60 bg-slate-900/95 backdrop-blur shrink-0 flex flex-col p-4 z-10 shadow-lg">
                    <h3 class="text-sm font-display text-tech-cyan mb-4 border-b border-slate-700 pb-2">涡旋属性详情</h3>
                    <div class="space-y-3 font-mono text-xs">
                      <div class="flex justify-between">
                        <span class="text-slate-400">类型</span>
                        <span :class="eddySelectedCenter.class_id === 1 ? 'text-cyan-400' : 'text-rose-400'">
                          {{ eddySelectedCenter.class_id === 1 ? '气旋式' : '反气旋式' }}
                        </span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-400">经度 (Lon)</span>
                        <span class="text-slate-200">{{ Number(eddySelectedCenter.lon).toFixed(3) }}°E</span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-400">纬度 (Lat)</span>
                        <span class="text-slate-200">{{ Number(eddySelectedCenter.lat).toFixed(3) }}°N</span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-400">等效像素面积</span>
                        <span class="text-slate-200">{{ eddySelectedCenter.area }} px²</span>
                      </div>
                      <div class="flex justify-between">
                        <span class="text-slate-400">估算旋转角速度</span>
                        <span class="text-slate-200">{{ eddySelectedVelocity }} rad/d</span>
                      </div>
                    </div>
                    <button class="tech-btn primary-btn w-full mt-auto" @click="eddyWorkbenchTab = 'track'; loadEddyTrack()">
                      <Compass class="w-4 h-4 inline mr-1" /> 生成演变轨迹
                    </button>
                  </div>
                </transition>
              </div>
            </div>

            <div class="glass-panel p-4 shrink-0 flex flex-col border-t border-slate-800">
              <div class="flex items-center gap-3 mb-3">
                <LineChart class="w-5 h-5 text-tech-cyan" />
                <h2 class="font-display text-base tracking-wider text-white m-0">单日涡旋数量统计</h2>
              </div>
              <div v-if="!eddyResult" class="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm">
                预测后显示气旋/反气旋数量
              </div>
              <div v-else class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-3 rounded-xl border border-cyan-400/30 bg-gradient-to-br from-cyan-500/15 to-cyan-700/5">
                  <div class="text-[11px] font-mono text-slate-300">气旋涡旋 (Cyclonic)</div>
                  <div class="text-2xl font-mono text-cyan-300 mt-1">{{ Number(eddyResult?.cyclonic_count || 0) }}</div>
                </div>
                <div class="p-3 rounded-xl border border-rose-400/30 bg-gradient-to-br from-rose-500/15 to-rose-700/5">
                  <div class="text-[11px] font-mono text-slate-300">反气旋涡旋 (Anticyclonic)</div>
                  <div class="text-2xl font-mono text-rose-300 mt-1">{{ Number(eddyResult?.anticyclonic_count || 0) }}</div>
                </div>
                <div class="p-3 rounded-xl border border-emerald-400/30 bg-gradient-to-br from-emerald-500/15 to-emerald-700/5">
                  <div class="text-[11px] font-mono text-slate-300">总数 (Total)</div>
                  <div class="text-2xl font-mono text-emerald-300 mt-1">{{ Number(eddyResult?.cyclonic_count || 0) + Number(eddyResult?.anticyclonic_count || 0) }}</div>
                </div>
              </div>
            </div>
          </div>

          <!-- 演变追踪 Tab -->
          <div v-show="eddyWorkbenchTab === 'track'" class="flex-1 flex flex-col gap-4 overflow-hidden">
            <div class="glass-panel p-4 flex-1 min-h-0 flex flex-col relative">
              <div class="flex items-center justify-between gap-3 mb-3 border-b border-slate-700 pb-2">
                <div class="flex items-center gap-3">
                  <MapIcon class="w-5 h-5 text-tech-cyan" />
                  <h2 class="font-display text-lg tracking-widest text-white m-0">涡旋生命周期与路径追踪</h2>
                </div>
                <button class="tech-btn primary-btn !py-1 !px-3 text-xs" @click="loadEddyTrack" :disabled="!eddySelectedCenter">
                  <Activity class="w-3 h-3 inline-block mr-1" /> 重载轨迹
                </button>
              </div>

              <div v-if="!eddySelectedCenter" class="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm">
                请先在“实时识别”中点击选中一个涡旋中心
              </div>
              <div v-else-if="eddyTrackLoading" class="flex-1 flex flex-col items-center justify-center text-tech-cyan">
                <div class="w-8 h-8 rounded-full border-2 border-tech-cyan border-t-transparent animate-spin mb-4"></div>
                <span class="font-mono text-xs animate-pulse">正在回溯关联历史多时相结果，生成生命周期演变轨迹...</span>
              </div>
              <div v-else-if="eddyTrackData" class="flex-1 min-h-0 flex gap-4">
                <!-- Track List -->
                <div class="w-64 border border-slate-800 rounded bg-slate-900/50 flex flex-col overflow-hidden">
                  <div class="p-3 bg-slate-800/80 text-xs font-bold text-slate-300 border-b border-slate-700">演变节点记录 (生命周期: {{ eddyTrackData.nodes?.length }}天)</div>
                  <div class="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-2">
                    <div v-for="(node, idx) in eddyTrackData.nodes" :key="idx" class="p-2 border border-slate-700 rounded bg-slate-950/50 text-[10px] font-mono">
                      <div class="text-tech-cyan mb-1 flex justify-between">
                        <span>Day: {{ node.day_index }}</span>
                        <span :class="node.class_id===1?'text-cyan-400':'text-rose-400'">{{ node.class_id===1?'气旋':'反气旋' }}</span>
                      </div>
                      <div class="text-slate-400">Lon: {{ node.lon.toFixed(3) }}, Lat: {{ node.lat.toFixed(3) }}</div>
                      <div class="text-slate-400">面积: {{ node.area }} px² | 位移: {{ node.shift.toFixed(2) }} px</div>
                    </div>
                  </div>
                </div>
                <!-- Track Chart Placeholder -->
                <div class="flex-1 border border-slate-800 rounded bg-slate-900/50 flex flex-col relative min-h-0">
                  <div class="p-3 bg-slate-800/80 text-xs font-bold text-slate-300 border-b border-slate-700 shrink-0 z-10">形态演变时序曲线</div>
                  <div class="flex-1 relative min-h-0">
                    <div id="eddy-track-chart" class="absolute inset-0"></div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </main>
      </section>
</template>

<script setup>
import { watch, nextTick } from 'vue'
import { Compass, Search, Zap, Download, Map as MapIcon, LineChart, Activity } from 'lucide-vue-next'
import { useEddy } from '../../composables/useEddy'

const {
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
  eddySelectedDiameter,
  eddySelectedVelocity,
  eddyWorkbenchTab,
  eddyTrackData,
  eddyTrackLoading,
  loadEddyTrack,
  loadEddyDataInfo,
  shiftEddyDate,
  runEddyPrediction,
  downloadEddyInfo,
  resizeEddyChart
} = useEddy()

watch(eddyWorkbenchTab, async (newTab) => {
  await nextTick()
  resizeEddyChart()
})
</script>
