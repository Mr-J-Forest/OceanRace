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
      <section v-if="activeModule === 'forecast'" class="h-full flex gap-6">
        
        <!-- Left Panel: Controls (Fixed Width) -->
        <aside class="w-[360px] glass-panel flex flex-col h-full shrink-0">
          <transition name="fade">
            <div v-if="loadingInfo || predicting" class="absolute inset-0 z-50 bg-tech-bg/90 backdrop-blur-md flex flex-col items-center justify-center">
              <div class="radar-scan"></div>
              <p class="mt-4 font-mono text-tech-cyan animate-pulse">
                {{ predicting ? 'EXECUTING PREDICTION ENGINE...' : 'ANALYZING DATASET...' }}
              </p>
            </div>
          </transition>

          <div class="flex-1 overflow-y-auto custom-scrollbar p-5 flex flex-col gap-6">
            <div>
              <h2 class="panel-title"><Terminal class="w-5 h-5 text-tech-cyan" /> COMMAND CENTER</h2>
              <div class="space-y-4 mt-6">
                <!-- Inputs -->
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

            <!-- Context Info -->
            <transition name="slide-down">
              <div v-if="datasetInfo" class="p-4 rounded-xl bg-slate-900/60 border border-tech-cyan/20">
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

                <button class="tech-btn primary-btn w-full mt-6 flex items-center justify-center gap-2 text-sm shadow-[0_0_20px_rgba(6,182,212,0.15)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)]" @click="runPrediction">
                  <Zap class="w-4 h-4 animate-pulse" /> START FORECAST
                </button>
              </div>
            </transition>

            <div class="mt-auto pt-6 border-t border-slate-800/50">
              <h2 class="panel-title text-sm"><AlertTriangle class="w-4 h-4 text-amber-500" /> ANOMALY STATUS</h2>
              <div class="p-4 rounded-lg border border-dashed border-amber-500/20 bg-amber-500/5 text-center">
                <p class="text-xs text-amber-500/60 font-mono tracking-wide">SYSTEM OFFLINE</p>
              </div>
            </div>
          </div>
        </aside>

        <!-- Right Flex Area: Charts -->
        <main class="flex-1 flex flex-col gap-6 min-w-0">
          
          <!-- Top: Spatial Evolution -->
          <div class="glass-panel flex-1 flex flex-col min-h-0 relative group">
            <div class="absolute inset-0 bg-gradient-to-b from-tech-cyan/5 to-transparent pointer-events-none"></div>
            
            <div class="px-5 py-3 border-b border-slate-800/80 flex items-center justify-between shrink-0 relative z-10 bg-tech-panel/40 backdrop-blur-sm">
              <div class="flex items-center gap-3">
                <MapIcon class="w-5 h-5 text-tech-cyan" />
                <h2 class="font-display text-lg tracking-widest text-white m-0">空间演变展示</h2>
              </div>
              <div v-if="hasResult" class="flex items-center gap-3">
                <div class="px-4 py-1 rounded-full bg-tech-cyan/10 border border-tech-cyan/40 text-tech-cyan font-mono text-sm shadow-[0_0_15px_rgba(6,182,212,0.2)] flex items-center gap-2">
                  <span class="w-2 h-2 rounded-full bg-tech-cyan animate-pulse"></span>
                  T+{{ (currentStep + 1) * STEP_HOURS }}H
                </div>
              </div>
            </div>

            <div class="flex-1 relative p-1 pb-4 flex flex-col min-h-0 z-10">
              <div v-if="!hasResult" class="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                <div class="radar-scan opacity-30"></div>
                <p class="font-mono text-xs tracking-[0.2em] mt-6">等待后端解析数据</p>
              </div>

              <div v-show="hasResult" class="flex-1 w-full h-full min-h-0 relative">
                <div id="spatial-chart" class="absolute inset-0"></div>
              </div>
            </div>
            
            <!-- Floating Playback Bar -->
            <div v-if="hasResult" class="absolute bottom-6 left-1/2 -translate-x-1/2 w-3/5 max-w-xl px-4 py-2.5 rounded-full bg-slate-900/60 backdrop-blur-md border border-tech-cyan/20 shadow-lg flex items-center gap-4 z-20 transition-all duration-300 hover:bg-slate-900/80 hover:border-tech-cyan/40">
              <button 
                class="w-10 h-10 shrink-0 rounded-full flex items-center justify-center transition-all"
                :class="isPlaying ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500 hover:text-slate-900' : 'bg-tech-cyan/10 text-tech-cyan border border-tech-cyan/30 hover:bg-tech-cyan hover:text-slate-900'"
                @click="togglePlay"
              >
                <Pause v-if="isPlaying" class="w-4 h-4 fill-current" />
                <Play v-else class="w-4 h-4 fill-current ml-1" />
              </button>
              
              <div class="flex-1 flex flex-col gap-1.5 opacity-80 hover:opacity-100 transition-opacity">
                <div class="flex justify-between text-[10px] font-mono font-medium tracking-wider">
                  <span class="text-slate-500">当前 (T+0H)</span>
                  <span class="text-tech-cyan/80">预测帧 {{ currentStep + 1 }}/{{ totalSteps }}</span>
                  <span class="text-slate-500">未来 (T+{{ totalSteps * STEP_HOURS }}H)</span>
                </div>
                <input
                  type="range"
                  v-model.number="currentStep"
                  min="0"
                  :max="totalSteps - 1"
                  @input="onStepSliderInput"
                  class="tech-slider h-1.5"
                />
              </div>
            </div>
          </div>

          <!-- Bottom: Regional Trends Curve -->
          <div class="glass-panel h-[35%] min-h-[280px] flex flex-col shrink-0">
            <div class="px-5 py-3 border-b border-slate-800/80 shrink-0 bg-tech-panel/40 backdrop-blur-sm">
              <div class="flex items-center gap-3">
                <LineChart class="w-5 h-5 text-tech-cyan" />
                <h2 class="font-display text-lg tracking-widest text-white m-0">区域变化趋势</h2>
              </div>
            </div>

            <div class="flex-1 relative p-1 min-h-0">
              <div v-if="!hasResult" class="absolute inset-0 flex flex-col items-center justify-center text-slate-500">
                <LineChart class="w-8 h-8 opacity-20 mb-3" />
                <p class="font-mono text-xs tracking-widest">暂无趋势序列数据</p>
              </div>

              <div v-show="hasResult" class="w-full h-full relative">
                <div id="curve-chart" class="absolute inset-0"></div>
              </div>
            </div>
          </div>
        </main>
      </section>

      <section v-else-if="activeModule === 'eddy'" class="h-full flex gap-6 bg-slate-900/20 rounded-2xl p-2">
        <aside class="w-[380px] glass-panel flex flex-col h-full shrink-0 p-5 gap-4 overflow-y-auto custom-scrollbar relative">
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

          <h2 class="panel-title"><Compass class="w-5 h-5 text-tech-cyan" /> EDDY DETECTION</h2>

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

          <div v-if="eddyDates.length > 0" class="space-y-2">
            <label class="text-[10px] font-mono text-slate-400 uppercase tracking-widest">Select Day</label>
            <select v-model.number="eddyDayIndex" class="tech-input">
              <option v-for="(d, idx) in eddyDates" :key="`${d}-${idx}`" :value="idx">
                {{ d }} (idx={{ idx }})
              </option>
            </select>
          </div>

          <button class="tech-btn primary-btn w-full mt-2 flex items-center justify-center gap-2" @click="runEddyPrediction" :disabled="eddyDates.length === 0 || eddyPredicting">
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

          <div v-if="eddyResult" class="p-3 rounded-lg border border-tech-cyan/20 bg-slate-900/50 text-xs font-mono text-slate-300">
            <div>date: {{ eddyResult.day_label }} (idx={{ eddyResult.day_index }})</div>
            <div>cyclonic: {{ eddyResult.cyclonic_count }}</div>
            <div>anticyclonic: {{ eddyResult.anticyclonic_count }}</div>
            <button class="tech-btn ghost-btn w-full mt-3 flex items-center justify-center gap-2" @click="downloadEddyInfo">
              <Download class="w-4 h-4" /> DOWNLOAD DAY INFO
            </button>
          </div>
        </aside>

        <main class="flex-1 min-w-0 flex flex-col gap-4">
          <div class="glass-panel p-4 h-[68%] min-h-[320px] flex flex-col">
            <div class="flex items-center gap-3 mb-3">
              <MapIcon class="w-5 h-5 text-tech-cyan" />
              <h2 class="font-display text-lg tracking-widest text-white m-0">涡旋日预测图（ADT + 边界 + 中心）</h2>
            </div>

            <div class="flex-1 relative min-h-0 bg-slate-900/25 rounded-lg border border-slate-700/60">
              <div v-if="!eddyResult" class="absolute inset-0 flex items-center justify-center text-slate-500 font-mono text-sm">
                先加载数据并选择日期，然后执行预测
              </div>
              <div v-show="!!eddyResult" id="eddy-chart" class="absolute inset-0"></div>
            </div>
          </div>

          <div class="glass-panel p-4 h-[32%] min-h-[150px] flex flex-col">
            <div class="flex items-center gap-3 mb-3">
              <LineChart class="w-5 h-5 text-tech-cyan" />
              <h2 class="font-display text-base tracking-wider text-white m-0">涡旋数量统计</h2>
            </div>

            <div v-if="!eddyResult" class="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm">
              预测后显示气旋/反气旋数量
            </div>

            <div v-else class="grid grid-cols-1 md:grid-cols-3 gap-3">
              <div class="p-3 rounded-lg border border-cyan-400/30 bg-cyan-500/10">
                <div class="text-[11px] font-mono text-slate-300">气旋涡旋 (Cyclonic)</div>
                <div class="text-2xl font-mono text-cyan-300 mt-1">{{ Number(eddyResult?.cyclonic_count || 0) }}</div>
              </div>
              <div class="p-3 rounded-lg border border-rose-400/30 bg-rose-500/10">
                <div class="text-[11px] font-mono text-slate-300">反气旋涡旋 (Anticyclonic)</div>
                <div class="text-2xl font-mono text-rose-300 mt-1">{{ Number(eddyResult?.anticyclonic_count || 0) }}</div>
              </div>
              <div class="p-3 rounded-lg border border-emerald-400/30 bg-emerald-500/10">
                <div class="text-[11px] font-mono text-slate-300">总数 (Total)</div>
                <div class="text-2xl font-mono text-emerald-300 mt-1">{{ Number(eddyResult?.cyclonic_count || 0) + Number(eddyResult?.anticyclonic_count || 0) }}</div>
              </div>
            </div>
          </div>
        </main>
      </section>

      <section v-else-if="activeModule === 'anomaly'" class="h-full flex gap-6">
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

        <main class="flex-1 glass-panel p-5 overflow-hidden flex flex-col min-w-0">
          <div class="flex items-center justify-between gap-3 mb-4">
            <div class="flex items-center gap-3">
              <Database class="w-5 h-5 text-tech-cyan" />
              <h2 class="font-display text-lg tracking-widest text-white m-0">风-浪异常智能识别与灾害预警</h2>
            </div>
            <div v-if="anomalyData" class="px-3 py-1 rounded border text-[11px] font-mono" :class="anomalyRiskClass">
              {{ anomalyRiskLevel }}预警 | 风险分 {{ anomalyRiskScore }}
            </div>
          </div>

          <div v-if="!anomalyData" class="flex-1 flex items-center justify-center text-slate-500 font-mono text-sm tracking-widest">
            请先加载 labels/events
          </div>

          <template v-else>
            <div class="grid grid-cols-2 md:grid-cols-5 gap-3 mb-4">
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                <div class="text-[10px] text-slate-400 font-mono">SAMPLES</div>
                <div class="text-xl text-white font-mono">{{ anomalyData.num_samples }}</div>
              </div>
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                <div class="text-[10px] text-slate-400 font-mono">POSITIVE</div>
                <div class="text-xl text-amber-400 font-mono">{{ anomalyData.num_positive }}</div>
              </div>
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                <div class="text-[10px] text-slate-400 font-mono">MATCHED_POSITIVE</div>
                <div class="text-xl text-emerald-400 font-mono">{{ anomalyData.matched_positive }}</div>
              </div>
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                <div class="text-[10px] text-slate-400 font-mono">EVENTS_HIT</div>
                <div class="text-xl text-tech-cyan font-mono">{{ anomalyData.matched_event_count }}</div>
              </div>
              <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                <div class="text-[10px] text-slate-400 font-mono">RISK_LEVEL</div>
                <div class="text-xl font-mono" :class="anomalyRiskClass">{{ anomalyRiskLevel }}</div>
              </div>
            </div>

            <div class="text-xs text-slate-400 font-mono mb-3">
              split={{ anomalyData.split }} | positive_ratio={{ (anomalyData.positive_ratio * 100).toFixed(2) }}% | matched_positive_ratio={{ (anomalyData.matched_positive_ratio * 100).toFixed(2) }}%
            </div>

            <div class="flex-1 overflow-auto custom-scrollbar rounded-lg border border-slate-700 bg-slate-900/40">
              <table class="w-full text-xs font-mono">
                <thead class="sticky top-0 bg-slate-900/90 text-slate-300">
                  <tr>
                    <th class="text-left p-2 border-b border-slate-700">index</th>
                    <th class="text-left p-2 border-b border-slate-700">timestamp</th>
                    <th class="text-left p-2 border-b border-slate-700">matched</th>
                    <th class="text-left p-2 border-b border-slate-700">event_hits</th>
                  </tr>
                </thead>
                <tbody>
                  <tr v-for="row in anomalyData.points" :key="`${row.index}-${row.timestamp}`" class="odd:bg-slate-900/20 even:bg-slate-800/20">
                    <td class="p-2 border-b border-slate-800">{{ row.index }}</td>
                    <td class="p-2 border-b border-slate-800">{{ row.timestamp }}</td>
                    <td class="p-2 border-b border-slate-800" :class="row.matched ? 'text-emerald-300' : 'text-slate-500'">{{ row.matched ? 'yes' : 'no' }}</td>
                    <td class="p-2 border-b border-slate-800">{{ Array.isArray(row.event_hits) && row.event_hits.length ? row.event_hits.join(', ') : '-' }}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          </template>
        </main>
      </section>

      <section v-else class="h-full flex items-center justify-center">
        <div class="glass-panel p-10 max-w-lg text-center relative overflow-hidden group">
          <div class="absolute inset-0 bg-gradient-to-br from-tech-cyan/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <component :is="modules.find(m => m.key === activeModule)?.icon" class="w-16 h-16 mx-auto text-slate-600 mb-6" />
          <h2 class="font-display text-2xl text-white mb-4 tracking-wider">{{ getModuleLabel(activeModule) }}</h2>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch, markRaw } from 'vue'
import axios from 'axios'
import Plotly from 'plotly.js-dist-min'
import { 
  Activity, Terminal, Cpu, Database, Search, CheckCircle2, 
  Zap, AlertTriangle, Map as MapIcon, Play, Pause, LineChart, Lock, Download,
  Waves, Compass, ShieldAlert
} from 'lucide-vue-next'

const modules = [
  { key: 'eddy', label: '涡旋检测', icon: markRaw(Compass) },
  { key: 'forecast', label: '要素预测', icon: markRaw(Waves) },
  { key: 'anomaly', label: '异常检测', icon: markRaw(ShieldAlert) }
]

const activeModule = ref('forecast')

const modelPath = ref('models/forecast_model.pt')
const dataPath = ref('data/processed/element_forecasting/示例数据.nc')
const datasetInfo = ref('')
const maxIndex = ref(0)
const startIdx = ref(0)

const eddyModelPath = ref('outputs/final_results/eddy_detection/meta4_mask_retrain_20260413_bg/checkpoints/best.pt')
const eddyDataPath = ref('data/processed/eddy_detection/19930101_20241231_clean.nc')
const eddyDatasetInfo = ref('')
const eddyDates = ref([])
const eddyDayIndex = ref(0)
const eddyLoading = ref(false)
const eddyPredicting = ref(false)
const eddyResult = ref(null)
const eddyProgress = ref(0)

const loadingInfo = ref(false)
const predicting = ref(false)
const hasResult = ref(false)
const sessionId = ref('')

const totalSteps = ref(0)
const currentStep = ref(0)

const isPlaying = ref(false)
let playInterval = null
let eddyProgressTimer = null
const colorRangeCache = new Map()

const API_BASE = '/api'
const currentTime = ref(new Date().toISOString().substring(11, 19))
const STEP_HOURS = 1

const anomalyLabelsPath = ref('outputs/anomaly_detection/labels_competition.json')
const anomalyEventsPath = ref('outputs/anomaly_detection/events_competition.json')
const anomalyManifestPath = ref('data/processed/splits/anomaly_detection_competition.json')
const anomalySplit = ref('test')
const anomalyLoading = ref(false)
const anomalyError = ref('')
const anomalyData = ref(null)
const anomalyRiskScore = ref(0)
const anomalyRiskLevel = ref('低')
const anomalyRiskClass = ref('text-emerald-300 border-emerald-400/40 bg-emerald-400/10')

let clockInterval
onMounted(() => {
  clockInterval = setInterval(() => {
    currentTime.value = new Date().toISOString().substring(11, 19)
  }, 1000)

  loadDefaultDataPath()
  loadEddyDefaults()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  clearInterval(clockInterval)
  stopPlay()
  if (eddyProgressTimer) {
    clearInterval(eddyProgressTimer)
    eddyProgressTimer = null
  }
  window.removeEventListener('resize', handleResize)
})

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

watch(activeModule, (moduleKey) => {
  if (moduleKey !== 'forecast') {
    stopPlay()
  }
})

const switchModule = (moduleKey) => {
  activeModule.value = moduleKey
}

const getModuleLabel = (moduleKey) => {
  const found = modules.find((m) => m.key === moduleKey)
  return found ? found.label : '模块'
}

const handleResize = () => {
  if (activeModule.value === 'forecast' && hasResult.value) {
    Plotly.Plots.resize('spatial-chart')
    Plotly.Plots.resize('curve-chart')
  }
  if (activeModule.value === 'eddy' && eddyResult.value) {
    Plotly.Plots.resize('eddy-chart')
  }
}

const loadDefaultDataPath = async () => {
  try {
    const res = await axios.get(`${API_BASE}/default-data-path`)
    if (res.data.path) dataPath.value = res.data.path
  } catch (err) {
    // no-op
  }
}

const loadEddyDefaults = async () => {
  try {
    const res = await axios.get(`${API_BASE}/eddy/default-paths`)
    if (res.data?.model_path) eddyModelPath.value = res.data.model_path
    if (res.data?.data_path) eddyDataPath.value = res.data.data_path
  } catch (err) {
    try {
      const fallback = await axios.get(`${API_BASE}/eddy/default-data-path`)
      if (fallback.data?.path) eddyDataPath.value = fallback.data.path
    } catch (fallbackErr) {
      // no-op
    }
  }
}

const loadDataInfo = async () => {
  loadingInfo.value = true
  try {
    const res = await axios.post(`${API_BASE}/dataset-info`, { data_path: dataPath.value })
    maxIndex.value = res.data.max_index
    datasetInfo.value = res.data.info
  } catch (err) {
    datasetInfo.value = `[ERR] 通信阻断: ${err.response?.data?.detail || err.message}`
  } finally {
    loadingInfo.value = false
  }
}

const loadEddyDataInfo = async () => {
  eddyLoading.value = true
  try {
    const res = await axios.post(`${API_BASE}/eddy/dataset-info`, {
      data_path: eddyDataPath.value
    })
    eddyDates.value = Array.isArray(res.data.dates) ? res.data.dates : []
    eddyDayIndex.value = 0
    eddyDatasetInfo.value = res.data.info || `可选天数: ${eddyDates.value.length}`
  } catch (err) {
    eddyDatasetInfo.value = `[ERR] ${err.response?.data?.detail || err.message}`
    eddyDates.value = []
  } finally {
    eddyLoading.value = false
  }
}

const runPrediction = async () => {
  predicting.value = true
  hasResult.value = false
  stopPlay()
  colorRangeCache.clear()

  try {
    const res = await axios.post(`${API_BASE}/predict`, {
      model_path: modelPath.value,
      data_path: dataPath.value,
      start_idx: startIdx.value
    })

    sessionId.value = res.data.session_id
    totalSteps.value = res.data.steps
    currentStep.value = 0
    hasResult.value = true

    await nextTick()
    await Promise.all([loadStepData(), loadCurveData()])
  } catch (err) {
    alert(`核心推理引擎故障: ${err.response?.data?.detail || err.message}`)
  } finally {
    predicting.value = false
  }
}

const runEddyPrediction = async () => {
  if (eddyDates.value.length === 0) return
  eddyPredicting.value = true
  startEddyProgress()
  eddyResult.value = null
  try {
    const res = await axios.post(`${API_BASE}/eddy/predict-day`, {
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
  } catch (err) {
    resetEddyProgress()
    alert(`涡旋预测失败: ${err.response?.data?.detail || err.message}`)
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

const loadAnomalyOverview = async () => {
  anomalyLoading.value = true
  anomalyError.value = ''
  try {
    const res = await axios.post(`${API_BASE}/anomaly/inspect`, {
      labels_json: anomalyLabelsPath.value,
      events_json: anomalyEventsPath.value,
      manifest_path: anomalyManifestPath.value,
      split: anomalySplit.value
    })
    anomalyData.value = res.data

    const score = Math.round((Number(res.data?.matched_positive_ratio || 0) * 70) + (Number(res.data?.positive_ratio || 0) * 30))
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
  } catch (err) {
    anomalyError.value = `异常模块加载失败: ${err.response?.data?.detail || err.message}`
  } finally {
    anomalyLoading.value = false
  }
}

const loadStepData = async () => {
  if (!sessionId.value) return
  try {
    const res = await axios.get(`${API_BASE}/predict/${sessionId.value}/step/${currentStep.value}`)
    renderSpatialPlot(res.data)
    updateVerticalLineOnCurve()
  } catch (err) {
    console.error('数据游标读取失败', err)
  }
}

const loadCurveData = async () => {
  if (!sessionId.value) return
  try {
    const res = await axios.get(`${API_BASE}/predict/${sessionId.value}/curve`)
    renderCurvePlot(res.data.data)
  } catch (err) {
    console.error('加载长效趋势数据失败', err)
  }
}

const onStepSliderInput = () => {
  stopPlay()
  loadStepData()
}

const togglePlay = () => {
  if (isPlaying.value) {
    stopPlay()
    return
  }

  isPlaying.value = true
  if (currentStep.value >= totalSteps.value - 1) currentStep.value = 0
  loadStepData()

  playInterval = setInterval(() => {
    if (currentStep.value >= totalSteps.value - 1) {
      stopPlay()
      return
    }
    currentStep.value += 1
    loadStepData()
  }, 1200)
}

const stopPlay = () => {
  isPlaying.value = false
  if (playInterval) clearInterval(playInterval)
  playInterval = null
}

const getChartLayoutBase = (title) => ({
  title: { text: title, font: { color: '#06b6d4', size: 14, family: 'Orbitron, sans-serif' } },
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#94a3b8', family: 'JetBrains Mono, monospace' },
  margin: { l: 40, r: 20, t: 50, b: 30 },
  xaxis: { showgrid: false, zeroline: false },
  yaxis: { showgrid: false, zeroline: false }
})

const getVariableRenderStyle = (varNameRaw) => {
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

const renderSpatialPlot = (stepData, containerId = 'spatial-chart') => {
  const container = document.getElementById(containerId)
  if (!container) return

  const plotItems = mergeVectorFieldsSpatial(stepData?.data || [])
  const plots = []
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
    const xEnd = xStart + blockWidth * 0.85
    const cbX = xEnd + 0.01
    const titleX = (xStart + xEnd) / 2

    plots.push({
      z: smoothedGrid,
      type: 'heatmap',
      zsmooth: style.isMask ? false : 'best',
      colorscale: style.colorscale,
      ...colorRange,
      hoverongaps: false,
      showscale: true,
      colorbar: {
        len: 0.65,
        thickness: 8,
        x: cbX,
        y: 0.57,
        tickfont: { color: '#94a3b8', size: 10, family: 'JetBrains Mono, monospace' },
        outlinewidth: 0,
        bgcolor: 'rgba(3,7,18,0.6)',
        bordercolor: 'rgba(6,182,212,0.3)',
        borderwidth: 1,
        title: { text: style.unit ? `${style.displayName}<br>${style.unit}` : style.displayName, font: { color: '#06b6d4', size: 10 } }
      },
      xaxis: `x${index + 1 > 1 ? index + 1 : ''}`,
      yaxis: `y${index + 1 > 1 ? index + 1 : ''}`,
      name: v.var,
      hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4', font: { color: '#06b6d4', family: 'JetBrains Mono, monospace' } }
    })

    annotations.push({
      text: `[ ${style.displayName} ] ${style.title}`,
      x: titleX,
      y: 1.05,
      xref: 'paper', yref: 'paper',
      xanchor: 'center', yanchor: 'bottom',
      showarrow: false,
      font: { color: '#06b6d4', size: 12, family: 'Orbitron, sans-serif' }
    })
  })

  // Set explicit manual domains to dynamically place all charts in a single row
  const layout = {
    ...getChartLayoutBase(''),
    annotations
  }

  for (let i = 0; i < plotItems.length; i++) {
    const ax = i === 0 ? '' : (i + 1)
    const blockWidth = 1.0 / N
    const xStart = i * blockWidth
    const xEnd = xStart + blockWidth * 0.85

    layout[`xaxis${ax}`] = { domain: [xStart, xEnd], anchor: `y${ax}`, showticklabels: true, tickfont: { color: '#475569', size: 9 }, gridcolor: 'rgba(30, 41, 59, 0.5)', zeroline: false }
    layout[`yaxis${ax}`] = { domain: [0.25, 0.90], anchor: `x${ax}`, autorange: 'reversed', showticklabels: true, tickfont: { color: '#475569', size: 9 }, gridcolor: 'rgba(30, 41, 59, 0.5)', zeroline: false }
  }

  Plotly.react(container, plots, layout, { responsive: true, displayModeBar: false })
}

const _findField2D = (items, fieldName) => {
  const found = (items || []).find((v) => String(v?.var || '').toUpperCase() === fieldName)
  return Array.isArray(found?.data) ? found.data : null
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
    const cyclonicCenters = centers.filter((c) => Number(c?.[2]) === 1)
    const anticyclonicCenters = centers.filter((c) => Number(c?.[2]) === 2)
    const otherCenters = centers.filter((c) => !Number.isFinite(Number(c?.[2])) || Number(c?.[2]) <= 0)

    if (cyclonicCenters.length > 0) {
      traces.push({
        x: cyclonicCenters.map((c) => Number(c?.[1])),
        y: cyclonicCenters.map((c) => Number(c?.[0])),
        type: 'scatter',
        mode: 'markers',
        marker: { color: cyclonicColor, size: 4, symbol: 'x' },
        name: 'cyclonic-center',
        hovertemplate: 'cyclonic center<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>'
      })
    }

    if (anticyclonicCenters.length > 0) {
      traces.push({
        x: anticyclonicCenters.map((c) => Number(c?.[1])),
        y: anticyclonicCenters.map((c) => Number(c?.[0])),
        type: 'scatter',
        mode: 'markers',
        marker: { color: anticyclonicColor, size: 4, symbol: 'x' },
        name: 'anticyclonic-center',
        hovertemplate: 'anticyclonic center<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>'
      })
    }

    if (otherCenters.length > 0) {
      traces.push({
        x: otherCenters.map((c) => Number(c?.[1])),
        y: otherCenters.map((c) => Number(c?.[0])),
        type: 'scatter',
        mode: 'markers',
        marker: { color: '#f8fafc', size: 7, symbol: 'diamond-open', line: { width: 1.2, color: '#0f172a' } },
        name: 'center',
        hovertemplate: 'center<br>x=%{x:.1f}, y=%{y:.1f}<extra></extra>'
      })
    }
  }

  const layout = {
    ...getChartLayoutBase(`Eddy Prediction | ${payload.day_label || '-'}`),
    margin: { l: 40, r: 20, t: 50, b: 40 },
    xaxis: { title: 'longitude index', showgrid: false, zeroline: false, constrain: 'domain' },
    yaxis: {
      title: 'latitude index',
      autorange: 'reversed',
      showgrid: false,
      zeroline: false,
      scaleanchor: 'x',
      scaleratio: 1,
      constrain: 'domain'
    },
    showlegend: false
  }

  Plotly.react(container, traces, layout, { responsive: true, displayModeBar: false })
}

const renderCurvePlot = (curveData, containerId = 'curve-chart') => {
  const container = document.getElementById(containerId)
  if (!container) return

  const plotItems = mergeVectorFieldsCurve(curveData)
  const plots = []
  const annotations = []
  const N = Math.max(1, plotItems.length)

  plotItems.forEach((v, index) => {
    const hours = Array.from({ length: v.means.length }, (_, i) => (i + 1) * STEP_HOURS)
    const style = getVariableRenderStyle(v.var)
    const titleX = (index * (1.0 / N)) + (1.0 / N) * 0.5

    plots.push({
      x: hours, y: v.means, type: 'scatter', mode: 'lines+markers',
      line: { color: style.lineColor || (style.displayName === 'SSUV' ? '#8b5cf6' : '#06b6d4'), width: 2, shape: 'spline' },
      marker: { color: style.markerColor || (style.displayName === 'SSUV' ? '#a78bfa' : '#22d3ee'), size: 4 },
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
    grid: { rows: 1, columns: N, pattern: 'independent', xgap: 0.08 }
  }

  for (let i = 0; i < plotItems.length; i++) {
    const ax = i === 0 ? '' : (i + 1)
    layout[`xaxis${ax}`] = { gridcolor: 'rgba(30, 41, 59, 0.5)', tickfont: { color: '#64748b' } }
    layout[`yaxis${ax}`] = { gridcolor: 'rgba(30, 41, 59, 0.5)', tickfont: { color: '#64748b' } }
  }

  Plotly.react(container, plots, layout, { responsive: true, displayModeBar: false })
}

const updateVerticalLineOnCurve = (containerId = 'curve-chart', axisCount = 4, stepIndex = currentStep.value) => {
  const container = document.getElementById(containerId)
  if (!container) return

  const currentHour = (stepIndex + 1) * STEP_HOURS
  const shapes = []

  for (let i = 1; i <= Math.max(1, axisCount); i++) {
    shapes.push({
      type: 'line',
      x0: currentHour, x1: currentHour, y0: 0, y1: 1,
      yref: 'paper', xref: `x${i === 1 ? '' : i}`,
      line: { color: 'rgba(245, 158, 11, 0.8)', width: 2, dash: 'dot' }
    })
  }

  Plotly.relayout(container, { shapes })
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
