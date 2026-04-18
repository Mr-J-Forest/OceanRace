<template>
  <section class="h-full flex gap-6">
    <aside class="w-[380px] glass-panel flex flex-col h-full shrink-0 p-5 gap-4 overflow-y-auto custom-scrollbar">
      <h2 class="panel-title"><ShieldAlert class="w-5 h-5 text-tech-cyan" /> ANOMALY INSPECTOR</h2>

      <div class="space-y-2">
        <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Labels JSON</label>
        <input type="text" v-model="anomalyLabelsPath" class="tech-input" />
      </div>

      <div class="space-y-2">
        <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Events JSON</label>
        <input type="text" v-model="anomalyEventsPath" class="tech-input" />
      </div>

      <div class="space-y-2">
        <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Manifest JSON</label>
        <input type="text" v-model="anomalyManifestPath" class="tech-input" />
      </div>

      <div class="space-y-2">
        <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Processed（path.txt 或目录）</label>
        <input type="text" v-model="anomalyProcessedPath" class="tech-input" />
      </div>

      <div class="space-y-2">
        <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Split</label>
        <select v-model="anomalySplit" class="tech-input">
          <option value="train">train</option>
          <option value="val">val</option>
          <option value="test">test</option>
        </select>
      </div>

      <button class="tech-btn primary-btn w-full flex items-center justify-center gap-2" @click="loadAnomalyOverview">
        <Search class="w-4 h-4" /> LOAD LABELS & HITS
      </button>

      <p v-if="anomalyError" class="text-xs text-rose-400 font-mono leading-relaxed">{{ anomalyError }}</p>
      <p v-if="anomalyLoading" class="text-xs text-tech-cyan font-mono animate-pulse">LOADING ANOMALY POINTS...</p>
    </aside>

    <main class="flex-1 glass-panel p-3 overflow-hidden flex flex-col min-w-0">
      <div class="flex items-center justify-between gap-2 mb-2">
        <div class="flex items-center gap-3">
          <Database class="w-5 h-5 text-tech-cyan" />
          <h2 class="font-display text-base tracking-wider text-white m-0">风-浪异常智能识别与灾害预警</h2>
        </div>
        <div v-if="anomalyData" class="px-2.5 py-0.5 rounded border text-[10px] font-mono" :class="anomalyRiskClass">
          {{ anomalyRiskLevel }}预警 | 风险分 {{ anomalyRiskScore }}
        </div>
      </div>

      <div v-if="!anomalyData" class="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm tracking-widest">
        请先加载 labels/events
      </div>

      <template v-else>
        <div class="grid grid-cols-2 md:grid-cols-5 gap-2 mb-1.5">
          <div class="px-2.5 py-1.5 rounded-lg bg-slate-900/60 border border-slate-700">
            <div class="text-[10px] text-slate-400 font-mono">SAMPLES</div>
            <div class="text-base text-white font-mono leading-tight">{{ anomalyData.num_samples }}</div>
          </div>
          <div class="px-2.5 py-1.5 rounded-lg bg-slate-900/60 border border-slate-700">
            <div class="text-[10px] text-slate-400 font-mono">POSITIVE</div>
            <div class="text-base text-amber-400 font-mono leading-tight">{{ anomalyData.num_positive }}</div>
          </div>
          <div class="px-2.5 py-1.5 rounded-lg bg-slate-900/60 border border-slate-700">
            <div class="text-[10px] text-slate-400 font-mono">MATCHED_POSITIVE</div>
            <div class="text-base text-emerald-400 font-mono leading-tight">{{ anomalyData.matched_positive }}</div>
          </div>
          <div class="px-2.5 py-1.5 rounded-lg bg-slate-900/60 border border-slate-700">
            <div class="text-[10px] text-slate-400 font-mono">EVENTS_HIT</div>
            <div class="text-base text-tech-cyan font-mono leading-tight">{{ anomalyData.matched_event_count }}</div>
          </div>
          <div class="px-2.5 py-1.5 rounded-lg bg-slate-900/60 border border-slate-700">
            <div class="text-[10px] text-slate-400 font-mono">RISK_LEVEL</div>
            <div class="text-base font-mono leading-tight" :class="anomalyRiskClass">{{ anomalyRiskLevel }}</div>
          </div>
        </div>

        <div class="text-[10px] text-slate-400 font-mono mb-1.5">
          split={{ anomalyData.split }} | positive_ratio={{ (anomalyData.positive_ratio * 100).toFixed(2) }}% |
          matched_positive_ratio={{ (anomalyData.matched_positive_ratio * 100).toFixed(2) }}%
        </div>

        <div class="text-[10px] text-tech-cyan/90 font-mono mb-1.5">
          实时窗口终点: {{ anomalyLatestTimeText }} | 当前选择: {{ anomalySelectedTimeText }}
        </div>

        <div class="flex items-center gap-1.5 mb-1.5 overflow-x-auto custom-scrollbar pb-1">
          <button
            class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
            :class="anomalyView === 'monitor' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
            @click="anomalyView = 'monitor'"
          >
            1. 实况监测与基准
          </button>
          <button
            class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
            :class="anomalyView === 'detect' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
            @click="anomalyView = 'detect'"
          >
            2. 智能识别与回溯
          </button>
          <button
            class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
            :class="anomalyView === 'typhoon' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
            @click="anomalyView = 'typhoon'"
          >
            3. 台风关联与评估
          </button>
          <button
            class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
            :class="anomalyView === 'warning' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
            @click="anomalyView = 'warning'"
          >
            4. 分级预警发布
          </button>
        </div>

        <div v-if="anomalyView === 'monitor'" class="flex-1 overflow-auto custom-scrollbar space-y-2">
          <div class="grid grid-cols-1 md:grid-cols-3 gap-2">
            <div class="p-2 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-[10px] text-slate-400 font-mono">实时风速估计</div>
              <div class="text-lg text-white font-mono mt-0.5">{{ anomalyMonitor.windNow }} m/s</div>
              <div class="text-[10px] text-slate-400 font-mono mt-0.5">基准 {{ anomalyMonitor.windBand }}</div>
            </div>
            <div class="p-2 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-[10px] text-slate-400 font-mono">实时波高估计</div>
              <div class="text-lg text-white font-mono mt-0.5">{{ anomalyMonitor.waveNow }} m</div>
              <div class="text-[10px] text-slate-400 font-mono mt-0.5">基准 {{ anomalyMonitor.waveBand }}</div>
            </div>
            <div class="p-2 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-[10px] text-slate-400 font-mono">24H 监测状态</div>
              <div class="text-lg font-mono mt-0.5" :class="anomalyRiskClass">{{ anomalyMonitor.status }}</div>
              <div class="text-[10px] text-slate-400 font-mono mt-0.5">终点 {{ anomalyLatestTimeText }}</div>
            </div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="glass-panel p-3 border border-slate-700/60">
              <div class="text-xs font-mono text-slate-300 mb-2">风速实况精细图</div>
              <div class="relative">
                <div v-if="anomalySnapshotLoading" class="absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-tech-cyan bg-slate-950/45">精细图加载中...</div>
                <div v-else-if="anomalySnapshotError" class="absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-rose-300 bg-slate-950/45 px-3 text-center">{{ anomalySnapshotError }}</div>
                <div v-else-if="!anomalySnapshot" class="absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-slate-400 bg-slate-950/45">请选择时间点加载精细图</div>
                <div id="anomaly-wind-map" class="h-[210px]" :class="anomalySnapshotLoading ? 'opacity-25' : 'opacity-100'"></div>
              </div>
            </div>
            <div class="glass-panel p-3 border border-slate-700/60">
              <div class="text-xs font-mono text-slate-300 mb-2">波高实况精细图</div>
              <div class="relative">
                <div v-if="anomalySnapshotLoading" class="absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-tech-cyan bg-slate-950/45">精细图加载中...</div>
                <div v-else-if="anomalySnapshotError" class="absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-rose-300 bg-slate-950/45 px-3 text-center">{{ anomalySnapshotError }}</div>
                <div v-else-if="!anomalySnapshot" class="absolute inset-0 z-10 flex items-center justify-center text-xs font-mono text-slate-400 bg-slate-950/45">请选择时间点加载精细图</div>
                <div id="anomaly-wave-map" class="h-[210px]" :class="anomalySnapshotLoading ? 'opacity-25' : 'opacity-100'"></div>
              </div>
            </div>
          </div>

          <div class="glass-panel p-2.5 border border-slate-700/60">
            <div class="text-[11px] font-mono text-slate-300 mb-1">24小时节点选择</div>
            <div class="text-[10px] text-slate-500 font-mono mb-1.5">点击节点切换下方精细图</div>
            <div id="anomaly-window-chart" class="h-[96px]"></div>
            <div class="text-[10px] text-slate-400 font-mono mt-1.5">样本 {{ anomalySelectedIndexText }} | 时间 {{ anomalySelectedTimeText }}</div>
          </div>

          <div class="glass-panel p-2.5 border border-slate-700/60">
            <div class="text-[11px] font-mono text-slate-300 mb-1">风浪实况与基准对比</div>
            <div id="anomaly-monitor-chart" class="h-[110px]"></div>
          </div>
        </div>

        <div v-else-if="anomalyView === 'detect'" class="flex-1 overflow-auto custom-scrollbar space-y-2">
          <div class="glass-panel p-2.5 border border-slate-700/60">
            <div class="flex items-center justify-between gap-2 mb-1.5">
              <div class="text-[11px] font-mono text-slate-300">异常事件时序曲线</div>
              <div class="text-[11px] font-mono text-tech-cyan">{{ anomalyTracebackRangeText }}</div>
            </div>
            <div id="anomaly-timeline-chart" class="h-[180px]"></div>
            <div class="mt-2 pt-2 border-t border-slate-700/60">
              <div class="flex items-center justify-between gap-2 mb-1.5">
                <div class="text-[11px] font-mono text-slate-300">总体回溯一周 | 当前窗口长度 3 天</div>
                <div class="text-[10px] font-mono text-slate-400">步长 {{ anomalyTracebackStepHours }}h</div>
              </div>
              <input
                type="range"
                class="w-full accent-cyan-400"
                min="0"
                :max="anomalyTracebackMaxStartHour"
                :step="anomalyTracebackStepHours"
                v-model.number="anomalyTracebackStartHour"
              />
              <div class="flex items-center justify-between text-[10px] text-slate-400 font-mono mt-1">
                <span>最近7天最早</span>
                <span>左右拖动选择3天回溯窗口</span>
                <span>最近时刻</span>
              </div>
            </div>
          </div>

          <div class="overflow-auto custom-scrollbar rounded-lg border border-slate-700 bg-slate-900/40">
            <table class="w-full text-xs font-mono">
              <thead class="sticky top-0 bg-slate-900/90 text-slate-300">
                <tr>
                  <th class="text-left p-2 border-b border-slate-700">样本</th>
                  <th class="text-left p-2 border-b border-slate-700">时间</th>
                  <th class="text-left p-2 border-b border-slate-700">异常幅度</th>
                  <th class="text-left p-2 border-b border-slate-700">真实风速均值</th>
                  <th class="text-left p-2 border-b border-slate-700">真实波高均值</th>
                  <th class="text-left p-2 border-b border-slate-700">标签</th>
                  <th class="text-left p-2 border-b border-slate-700">影响范围</th>
                  <th class="text-left p-2 border-b border-slate-700">持续时长</th>
                  <th class="text-left p-2 border-b border-slate-700">事件命中</th>
                </tr>
              </thead>
              <tbody>
                <tr v-for="row in anomalyTracebackRows" :key="`${row.index}-${row.timestamp}`" class="odd:bg-slate-900/20 even:bg-slate-800/20">
                  <td class="p-2 border-b border-slate-800">{{ row.index }}</td>
                  <td class="p-2 border-b border-slate-800">{{ row.time }}</td>
                  <td class="p-2 border-b border-slate-800" :class="row.amplitude >= 2.4 ? 'text-rose-300' : 'text-amber-300'">{{ row.amplitude.toFixed(2) }} 倍</td>
                  <td class="p-2 border-b border-slate-800 text-cyan-200">{{ Number.isFinite(row.windMean) ? row.windMean.toFixed(3) : '-' }}</td>
                  <td class="p-2 border-b border-slate-800 text-cyan-100">{{ Number.isFinite(row.waveMean) ? row.waveMean.toFixed(3) : '-' }}</td>
                  <td class="p-2 border-b border-slate-800" :class="row.labelSignal === 1 ? 'text-rose-300' : 'text-slate-400'">{{ row.labelSignal }}</td>
                  <td class="p-2 border-b border-slate-800">{{ row.scope }}</td>
                  <td class="p-2 border-b border-slate-800">{{ row.duration }} 小时</td>
                  <td class="p-2 border-b border-slate-800">{{ row.eventHits }}</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>

        <div v-else-if="anomalyView === 'typhoon'" class="flex-1 overflow-auto custom-scrollbar space-y-3">
          <div class="glass-panel p-3 border border-slate-700/60">
            <div class="text-xs font-mono text-slate-300 mb-2">风险区域空间分布</div>
            <div id="anomaly-riskmap-chart" class="h-[240px]"></div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="p-3 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-xs font-mono text-slate-300 mb-2">台风-风浪异常耦合</div>
              <div class="space-y-2 text-xs font-mono">
                <div v-for="cp in anomalyCouplings" :key="cp.name" class="p-2 rounded border border-slate-700 bg-slate-900/60">
                  <div class="text-tech-cyan">{{ cp.name }} | 耦合度 {{ cp.score }}</div>
                  <div class="text-slate-400 mt-1">路径速度 {{ cp.speed }} kt，强度 {{ cp.intensity }} kt</div>
                </div>
              </div>
            </div>
            <div class="p-3 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-xs font-mono text-slate-300 mb-2">历史相似案例库</div>
              <div class="space-y-2 text-xs font-mono">
                <div v-for="item in anomalyCases" :key="item.id" class="p-2 rounded border border-slate-700 bg-slate-900/60">
                  <div class="text-white">{{ item.id }}</div>
                  <div class="text-slate-400 mt-1">相似度 {{ item.similarity }} | 窗口 {{ item.window }}</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div v-else class="flex-1 overflow-auto custom-scrollbar space-y-3">
          <div class="p-3 rounded-lg border" :class="anomalyRiskClass">
            <div class="text-sm font-mono">当前{{ anomalyWarning.level }}色预警 | 风险分 {{ anomalyRiskScore }}</div>
            <div class="text-xs font-mono mt-1 opacity-90">数据时效：{{ anomalyDataTimeText }}</div>
            <div class="text-xs font-mono mt-1 opacity-90">发布时间：{{ anomalyIssueTimeText }}</div>
            <div class="text-xs font-mono mt-1 opacity-90">推送对象：{{ anomalyWarning.targets }}</div>
          </div>

          <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div class="p-3 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-xs font-mono text-slate-300 mb-2">应对建议</div>
              <div class="space-y-2 text-xs font-mono">
                <div v-for="a in anomalyWarning.actions" :key="a" class="p-2 rounded bg-slate-900/60 border border-slate-700">{{ a }}</div>
              </div>
            </div>
            <div class="p-3 rounded-lg border border-slate-700 bg-slate-900/50">
              <div class="text-xs font-mono text-slate-300 mb-2">预警流程留痕</div>
              <div class="space-y-2 text-xs font-mono">
                <div v-for="log in anomalyWarningLogs" :key="log.id" class="p-2 rounded bg-slate-900/60 border border-slate-700">
                  <div class="text-slate-200">{{ log.issuedAt || log.time }} | 数据时效 {{ log.dataTime || '-' }} | {{ log.action }}</div>
                  <div class="text-slate-400 mt-1">{{ log.note }}</div>
                </div>
              </div>
            </div>
          </div>

          <div class="p-3 rounded-lg border border-slate-700 bg-slate-900/50">
            <div class="flex items-center justify-between mb-2">
              <div class="text-xs font-mono text-slate-300">标准化预警简报</div>
              <button class="tech-btn ghost-btn px-3 py-1 text-[11px]" @click="copyAnomalyBrief">复制简报</button>
            </div>
            <textarea v-model="anomalyBrief" class="tech-input h-36 !leading-6"></textarea>
          </div>
        </div>
      </template>
    </main>
  </section>
</template>

<script setup>
import { ShieldAlert, Search, Database } from 'lucide-vue-next'
import { useAnomaly } from '../../composables/useAnomaly'

const {
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
  anomalyLatestTimeText,
  anomalySelectedTimeText,
  anomalySelectedIndexText,
  anomalyTracebackRows,
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
  copyAnomalyBrief
} = useAnomaly()
</script>
