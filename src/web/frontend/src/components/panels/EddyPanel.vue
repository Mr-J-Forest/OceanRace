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
          <div class="glass-panel p-4 h-[70%] min-h-[360px] flex flex-col">
            <div class="flex items-center justify-between gap-3 mb-3">
              <div class="flex items-center gap-3">
                <MapIcon class="w-5 h-5 text-tech-cyan" />
                <h2 class="font-display text-lg tracking-widest text-white m-0">涡旋日预测图（ADT + 边界 + 中心）</h2>
              </div>
              <div v-if="eddyResult" class="px-3 py-1 rounded-full border border-tech-cyan/40 bg-tech-cyan/10 text-[11px] font-mono text-tech-cyan">
                {{ eddyResult.day_label }} | idx={{ eddyResult.day_index }}
              </div>
            </div>

            <!-- 图例放在图表容器外，避免与 Plotly 坐标轴/画布重合 -->
            <div
              v-if="eddyResult"
              class="flex flex-wrap items-center gap-2 mb-2 shrink-0 pl-0.5"
              aria-label="边界图例"
            >
              <span class="px-2 py-1 rounded border border-cyan-400/40 bg-slate-900/90 text-[10px] font-mono text-cyan-200 shadow-sm">蓝线=气旋边界</span>
              <span class="px-2 py-1 rounded border border-rose-400/40 bg-slate-900/90 text-[10px] font-mono text-rose-200 shadow-sm">红线=反气旋边界</span>
            </div>

            <div class="flex-1 relative min-h-0 bg-slate-900/25 rounded-xl border border-slate-700/60 overflow-hidden shadow-[inset_0_0_0_1px_rgba(6,182,212,0.08)]">
              <div class="absolute inset-0 pointer-events-none bg-gradient-to-b from-tech-cyan/5 to-transparent"></div>
              <div v-if="!eddyResult" class="absolute inset-0 flex items-center justify-center text-slate-500 font-mono text-sm">
                先加载数据并选择日期，然后执行预测
              </div>
              <div v-show="!!eddyResult" id="eddy-chart" class="absolute inset-0"></div>
            </div>
          </div>

          <div class="glass-panel p-4 h-[30%] min-h-[170px] flex flex-col">
            <div class="flex items-center gap-3 mb-3">
              <LineChart class="w-5 h-5 text-tech-cyan" />
              <h2 class="font-display text-base tracking-wider text-white m-0">涡旋数量统计</h2>
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
        </main>
      </section>
</template>

<script setup>
import { Compass, Search, Zap, Download, Map as MapIcon, LineChart } from 'lucide-vue-next'
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
  loadEddyDataInfo,
  shiftEddyDate,
  runEddyPrediction,
  downloadEddyInfo
} = useEddy()
</script>
