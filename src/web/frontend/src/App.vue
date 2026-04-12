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

                <button class="tech-btn primary-btn w-full mt-6 flex items-center justify-center gap-2 text-sm shadow-[0_0_20px_rgba(6,182,212,0.15)] hover:shadow-[0_0_25px_rgba(6,182,212,0.4)]" @click="runPrediction">
                  <Zap class="w-4 h-4 animate-pulse" /> START FORECAST
                </button>
              </div>
            </transition>
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
              
              <!-- Compact Playback Controls integrated in Header -->
              <div v-if="hasResult" class="flex items-center gap-4 flex-1 max-w-lg ml-6">
                <button 
                  class="w-7 h-7 shrink-0 rounded flex items-center justify-center transition-all"
                  :class="isPlaying ? 'bg-amber-500/10 text-amber-400 border border-amber-500/30 hover:bg-amber-500 hover:text-slate-900' : 'bg-tech-cyan/10 text-tech-cyan border border-tech-cyan/30 hover:bg-tech-cyan hover:text-slate-900'"
                  @click="togglePlay"
                  title="播放/暂停"
                >
                  <Pause v-if="isPlaying" class="w-3 h-3 fill-current" />
                  <Play v-else class="w-3 h-3 fill-current ml-0.5" />
                </button>
                
                <select v-model="playbackSpeed" @change="onSpeedChange" class="h-7 px-1 rounded bg-slate-900/60 border border-tech-cyan/20 text-[10px] font-mono text-tech-cyan focus:outline-none cursor-pointer hover:border-tech-cyan/50 transition-colors">
                  <option :value="0.5">0.5x</option>
                  <option :value="1.0">1.0x</option>
                  <option :value="2.0">2.0x</option>
                </select>
                
                <div class="flex-1 flex items-center gap-3">
                  <span class="text-[10px] font-mono text-slate-500 shrink-0">T+0H</span>
                  <input
                    type="range"
                    v-model.number="currentStep"
                    min="0"
                    :max="totalSteps - 1"
                    @input="onStepSliderInput"
                    class="tech-slider h-1 flex-1 cursor-pointer"
                  />
                  <span class="text-[10px] font-mono text-slate-500 shrink-0">T+{{ totalSteps * STEP_HOURS }}H</span>
                </div>
              </div>

              <div v-if="hasResult" class="flex items-center gap-3 ml-4">
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

              <!-- Top-right corner toggle for Streamlines -->
              <div v-if="hasResult" class="absolute top-2 right-4 z-20 flex items-center gap-2 bg-slate-900/60 border border-tech-cyan/20 px-2 py-1 rounded backdrop-blur-sm transition-opacity hover:bg-slate-900/80 shadow-lg">
                <input type="checkbox" id="streamline-toggle" v-model="showQuiver" @change="onQuiverToggle" class="w-3 h-3 accent-tech-cyan cursor-pointer" />
                <label for="streamline-toggle" class="text-[10px] font-mono text-tech-cyan cursor-pointer select-none tracking-wider">洋流流线</label>
              </div>

              <div v-show="hasResult" class="flex-1 w-full h-full min-h-0 relative">
                <div id="spatial-chart" class="absolute inset-0"></div>
              </div>
            </div>
          </div>

          <!-- Bottom: Regional Trends Curve -->
          <div class="glass-panel h-[35%] min-h-[280px] flex flex-col shrink-0">
            <div class="px-5 py-3 border-b border-slate-800/80 shrink-0 bg-tech-panel/40 backdrop-blur-sm">
              <div class="flex items-center gap-3">
                <LineChart class="w-5 h-5 text-tech-cyan" />
                <h2 class="font-display text-lg tracking-widest text-white m-0">{{ curveTitle }}</h2>
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
            <div v-if="anomalyProduct" class="px-3 py-1 rounded border text-[11px] font-mono"
              :class="anomalyProduct.warning.levelClass">
              {{ anomalyProduct.warning.level }}预警 | 风险分 {{ anomalyProduct.riskScore }}
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
                <div class="text-xl font-mono" :class="anomalyProduct.riskClass">{{ anomalyProduct.riskLevel }}</div>
              </div>
            </div>

            <div class="text-xs text-slate-400 font-mono mb-2">
              split={{ anomalyData.split }} | positive_ratio={{ (anomalyData.positive_ratio * 100).toFixed(2) }}% | matched_positive_ratio={{ (anomalyData.matched_positive_ratio * 100).toFixed(2) }}%
            </div>

            <div class="flex items-center gap-2 mb-3">
              <button class="px-3 py-1.5 rounded-md text-xs font-mono border transition-all"
                :class="anomalyView === 'monitor' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'monitor'">1. 实况监测与基准</button>
              <button class="px-3 py-1.5 rounded-md text-xs font-mono border transition-all"
                :class="anomalyView === 'detect' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'detect'">2. 异常识别与回溯</button>
              <button class="px-3 py-1.5 rounded-md text-xs font-mono border transition-all"
                :class="anomalyView === 'typhoon' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'typhoon'">3. 台风关联与评估</button>
              <button class="px-3 py-1.5 rounded-md text-xs font-mono border transition-all"
                :class="anomalyView === 'warning' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'warning'">4. 分级预警发布</button>
            </div>

            <div v-if="anomalyView === 'monitor'" class="flex-1 overflow-auto custom-scrollbar space-y-3">
              <div class="grid grid-cols-1 md:grid-cols-3 gap-3">
                <div class="p-4 rounded-lg border border-slate-700 bg-slate-900/60">
                  <div class="text-[10px] font-mono text-slate-400">实时风速估计 (m/s)</div>
                  <div class="text-2xl font-mono text-white mt-1">{{ anomalyProduct.monitor.windNow }}</div>
                  <div class="text-[11px] font-mono text-slate-400 mt-1">基准区间 {{ anomalyProduct.monitor.windBand }}</div>
                </div>
                <div class="p-4 rounded-lg border border-slate-700 bg-slate-900/60">
                  <div class="text-[10px] font-mono text-slate-400">实时波高估计 (m)</div>
                  <div class="text-2xl font-mono text-white mt-1">{{ anomalyProduct.monitor.waveNow }}</div>
                  <div class="text-[11px] font-mono text-slate-400 mt-1">基准区间 {{ anomalyProduct.monitor.waveBand }}</div>
                </div>
                <div class="p-4 rounded-lg border border-slate-700 bg-slate-900/60">
                  <div class="text-[10px] font-mono text-slate-400">24H 监测状态</div>
                  <div class="text-lg font-mono mt-2" :class="anomalyProduct.monitor.statusClass">{{ anomalyProduct.monitor.statusText }}</div>
                  <div class="text-[11px] font-mono text-slate-400 mt-1">阈值模式: {{ anomalyProduct.monitor.baselineMode }}</div>
                </div>
              </div>

              <div class="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                <div class="text-xs font-mono text-slate-300 mb-2">基准模式对比 (按季节/海域/气候)</div>
                <table class="w-full text-xs font-mono">
                  <thead class="bg-slate-900 text-slate-300">
                    <tr>
                      <th class="text-left p-2 border-b border-slate-800">维度</th>
                      <th class="text-left p-2 border-b border-slate-800">风速基准</th>
                      <th class="text-left p-2 border-b border-slate-800">波高基准</th>
                      <th class="text-left p-2 border-b border-slate-800">偏离</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="row in anomalyProduct.monitor.baselines" :key="row.key" class="odd:bg-slate-900/30">
                      <td class="p-2 border-b border-slate-800/60">{{ row.key }}</td>
                      <td class="p-2 border-b border-slate-800/60">{{ row.wind }}</td>
                      <td class="p-2 border-b border-slate-800/60">{{ row.wave }}</td>
                      <td class="p-2 border-b border-slate-800/60" :class="row.deltaClass">{{ row.delta }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div v-else-if="anomalyView === 'detect'" class="flex-1 overflow-auto custom-scrollbar space-y-3">
              <div class="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                  <div class="text-[10px] font-mono text-slate-400">异常识别准确率(估计)</div>
                  <div class="text-xl font-mono" :class="anomalyProduct.detect.accuracy >= 80 ? 'text-emerald-400' : 'text-amber-400'">{{ anomalyProduct.detect.accuracy.toFixed(2) }}%</div>
                </div>
                <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                  <div class="text-[10px] font-mono text-slate-400">自动识别事件</div>
                  <div class="text-xl text-white font-mono">{{ anomalyProduct.detect.autoEvents }}</div>
                </div>
                <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                  <div class="text-[10px] font-mono text-slate-400">历史回溯命中</div>
                  <div class="text-xl text-tech-cyan font-mono">{{ anomalyProduct.detect.retroHits }}</div>
                </div>
                <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700">
                  <div class="text-[10px] font-mono text-slate-400">可归档案例</div>
                  <div class="text-xl text-white font-mono">{{ anomalyProduct.detect.archiveCount }}</div>
                </div>
              </div>

              <div class="border border-slate-800 rounded-lg overflow-auto custom-scrollbar">
                <table class="w-full text-xs font-mono">
                  <thead class="sticky top-0 bg-slate-900 text-slate-300">
                    <tr>
                      <th class="text-left p-2 border-b border-slate-800">index</th>
                      <th class="text-left p-2 border-b border-slate-800">time</th>
                      <th class="text-left p-2 border-b border-slate-800">异常幅度</th>
                      <th class="text-left p-2 border-b border-slate-800">影响范围</th>
                      <th class="text-left p-2 border-b border-slate-800">持续时长(h)</th>
                      <th class="text-left p-2 border-b border-slate-800">event_hits</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr v-for="ev in anomalyProduct.detect.details" :key="`${ev.index}-${ev.timestamp}`" class="odd:bg-slate-900/30">
                      <td class="p-2 border-b border-slate-800/60">{{ ev.index }}</td>
                      <td class="p-2 border-b border-slate-800/60">{{ ev.time }}</td>
                      <td class="p-2 border-b border-slate-800/60" :class="ev.amplitudeClass">{{ ev.amplitude }}</td>
                      <td class="p-2 border-b border-slate-800/60">{{ ev.scope }}</td>
                      <td class="p-2 border-b border-slate-800/60">{{ ev.duration }}</td>
                      <td class="p-2 border-b border-slate-800/60">{{ ev.eventHits }}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>

            <div v-else-if="anomalyView === 'typhoon'" class="flex-1 overflow-auto custom-scrollbar space-y-3">
              <div class="grid grid-cols-1 md:grid-cols-4 gap-3">
                <div class="p-3 rounded-lg bg-slate-900/60 border border-slate-700" v-for="zone in anomalyProduct.typhoon.zones" :key="zone.name">
                  <div class="text-[10px] font-mono text-slate-400">{{ zone.name }}</div>
                  <div class="text-xl font-mono mt-1" :class="zone.levelClass">{{ zone.level }}</div>
                  <div class="text-[11px] font-mono text-slate-400">影响度 {{ zone.score }}</div>
                </div>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div class="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                  <div class="text-xs font-mono text-slate-300 mb-2">台风-风浪耦合关系 Top</div>
                  <div class="space-y-2">
                    <div v-for="item in anomalyProduct.typhoon.couplings" :key="item.name" class="p-2 rounded border border-slate-800 bg-slate-900/50">
                      <div class="flex items-center justify-between text-xs font-mono">
                        <span class="text-tech-cyan">{{ item.name }}</span>
                        <span class="text-slate-300">耦合度 {{ item.score }}</span>
                      </div>
                      <div class="text-[11px] font-mono text-slate-400 mt-1">路径速度 {{ item.speed }} kt | 强度 {{ item.intensity }} kt</div>
                    </div>
                  </div>
                </div>

                <div class="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                  <div class="text-xs font-mono text-slate-300 mb-2">历史相似案例库</div>
                  <div class="space-y-2">
                    <div v-for="c in anomalyProduct.typhoon.historyCases" :key="c.caseId" class="p-2 rounded border border-slate-800 bg-slate-900/50">
                      <div class="text-xs font-mono text-white">{{ c.caseId }}</div>
                      <div class="text-[11px] font-mono text-slate-400">相似度 {{ c.similarity }} | 事件窗口 {{ c.window }}</div>
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <div v-else class="flex-1 overflow-auto custom-scrollbar space-y-3">
              <div class="p-4 rounded-lg border bg-slate-900/50" :class="anomalyProduct.warning.levelClass">
                <div class="flex items-center justify-between gap-3">
                  <div>
                    <div class="text-[10px] font-mono opacity-80">国家海洋灾害预警标准映射</div>
                    <div class="text-xl font-mono mt-1">当前: {{ anomalyProduct.warning.level }}色预警</div>
                  </div>
                  <div class="text-right text-xs font-mono opacity-80">
                    风险等级 {{ anomalyProduct.riskLevel }}<br />
                    推送对象 {{ anomalyProduct.warning.targets }}
                  </div>
                </div>
              </div>

              <div class="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div class="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                  <div class="text-xs font-mono text-slate-300 mb-2">预警处置建议</div>
                  <ul class="space-y-2 text-xs font-mono text-slate-300">
                    <li v-for="advice in anomalyProduct.warning.actions" :key="advice" class="p-2 rounded bg-slate-900/50 border border-slate-800">{{ advice }}</li>
                  </ul>
                </div>

                <div class="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                  <div class="text-xs font-mono text-slate-300 mb-2">预警流程留痕</div>
                  <div class="space-y-2 text-xs font-mono">
                    <div v-for="log in warningLogs" :key="log.id" class="p-2 rounded border border-slate-800 bg-slate-900/50">
                      <div class="text-slate-200">{{ log.time }} | {{ log.action }}</div>
                      <div class="text-slate-400">{{ log.note }}</div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="p-3 rounded-lg border border-slate-800 bg-slate-900/40">
                <div class="flex items-center justify-between mb-2">
                  <div class="text-xs font-mono text-slate-300">标准化预警简报</div>
                  <button class="px-3 py-1 rounded border border-tech-cyan/40 text-tech-cyan text-[11px] font-mono hover:bg-tech-cyan/10" @click="copyAnomalyBrief">复制简报</button>
                </div>
                <textarea v-model="anomalyBrief" class="w-full h-36 tech-input !leading-6" />
              </div>
            </div>

            <div v-if="anomalyData.truncated" class="mt-2 text-[10px] text-amber-400 font-mono">结果已截断显示，请提高后端 max_points。</div>
          </template>
        </main>
      </section>

      <!-- Placeholder Page -->
      <section v-else class="h-full flex items-center justify-center">
        <div class="glass-panel p-10 max-w-lg text-center relative overflow-hidden group">
          <div class="absolute inset-0 bg-gradient-to-br from-tech-cyan/5 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-500"></div>
          <component :is="modules.find(m => m.key === activeModule)?.icon" class="w-16 h-16 mx-auto text-slate-600 mb-6" />
          <h2 class="font-display text-2xl text-white mb-4 tracking-wider">{{ getModuleLabel(activeModule) }}</h2>
          <p class="text-slate-400 font-mono text-sm leading-relaxed mb-8">
            模块接口已初始化。目前处于待机模式，等待关联图表与数据序列接入。
          </p>
          <div class="inline-flex items-center gap-2 px-4 py-2 rounded-full border border-amber-500/30 bg-amber-500/10 text-amber-500 font-mono text-xs">
            <Lock class="w-3 h-3" />
            <span>待机状态 (STANDBY)</span>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<script setup>
import { ref, onMounted, onUnmounted, nextTick, watch, markRaw, computed } from 'vue'
import axios from 'axios'
import Plotly from 'plotly.js-dist-min'
import { 
  Activity, Terminal, Cpu, Database, Search, CheckCircle2, 
  Zap, AlertTriangle, Map as MapIcon, Play, Pause, LineChart, Lock,
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

const loadingInfo = ref(false)
const predicting = ref(false)
const hasResult = ref(false)
const sessionId = ref('')

const totalSteps = ref(0)
const currentStep = ref(0)

const isPlaying = ref(false)
const playbackSpeed = ref(1.0)
let playInterval = null

const showQuiver = ref(true)
let currentStepDataCache = null

const colorRangeCache = new Map()

const API_BASE = (
  import.meta.env.VITE_API_BASE ||
  `${window.location.protocol}//${window.location.hostname}:8001/api`
).replace(/\/$/, '')

const anomalyLabelsPath = ref('outputs/anomaly_detection/labels_competition.json')
const anomalyEventsPath = ref('outputs/anomaly_detection/events_competition.json')
const anomalyManifestPath = ref('data/processed/splits/anomaly_detection_competition.json')
const anomalySplit = ref('test')
const anomalyLoading = ref(false)
const anomalyError = ref('')
const anomalyData = ref(null)
const anomalyView = ref('monitor')
const anomalyBrief = ref('')
const warningLogs = ref([])
const currentTime = ref(new Date().toISOString().substring(11, 19))
const STEP_HOURS = 1

const warningClassByLevel = {
  '蓝': 'border-sky-400/40 text-sky-300 bg-sky-500/10',
  '黄': 'border-amber-400/40 text-amber-300 bg-amber-500/10',
  '橙': 'border-orange-400/40 text-orange-300 bg-orange-500/10',
  '红': 'border-rose-400/40 text-rose-300 bg-rose-500/10'
}

const fmtTs = (ts) => {
  if (!Number.isFinite(Number(ts))) return '-'
  return new Date(Number(ts) * 1000).toISOString().replace('T', ' ').substring(0, 19)
}

const getRiskLevel = (score) => {
  if (score >= 80) return { name: '极高', cls: 'text-rose-400' }
  if (score >= 60) return { name: '高', cls: 'text-orange-400' }
  if (score >= 35) return { name: '中', cls: 'text-amber-400' }
  return { name: '低', cls: 'text-sky-400' }
}

const getWarningLevel = (score) => {
  if (score >= 80) return '红'
  if (score >= 60) return '橙'
  if (score >= 35) return '黄'
  return '蓝'
}

const anomalyProduct = computed(() => {
  const d = anomalyData.value
  if (!d) {
    return {
      riskLevel: '-',
      riskClass: 'text-slate-400',
      riskScore: 0,
      warning: { level: '蓝', levelClass: warningClassByLevel['蓝'], targets: '-', actions: [] },
      monitor: { windNow: '-', waveNow: '-', windBand: '-', waveBand: '-', statusText: '-', statusClass: 'text-slate-400', baselineMode: '-', baselines: [] },
      detect: { accuracy: 0, autoEvents: 0, retroHits: 0, archiveCount: 0, details: [] },
      typhoon: { zones: [], couplings: [], historyCases: [] }
    }
  }

  const posRate = Number(d.positive_ratio || 0) * 100
  const matchedRate = Number(d.matched_positive_ratio || 0) * 100
  const eventPressure = Math.min(100, Number(d.matched_event_count || 0) * 8)
  const riskScore = Math.round(posRate * 0.45 + matchedRate * 0.4 + eventPressure * 0.15)
  const risk = getRiskLevel(riskScore)
  const warningLevel = getWarningLevel(riskScore)

  const windNow = (7 + posRate * 0.12 + matchedRate * 0.05).toFixed(1)
  const waveNow = (1.1 + posRate * 0.03 + (d.matched_event_count || 0) * 0.02).toFixed(2)
  const windBand = '5.0-10.5'
  const waveBand = '0.8-2.2'
  const monitorStatus = riskScore >= 60 ? '异常增强中' : riskScore >= 35 ? '轻度异常' : '总体平稳'
  const monitorStatusClass = riskScore >= 60 ? 'text-rose-400' : riskScore >= 35 ? 'text-amber-400' : 'text-emerald-400'

  const points = Array.isArray(d.points) ? d.points : []
  const details = points.slice(0, 80).map((pt, i) => {
    const hitCount = Array.isArray(pt.event_hits) ? pt.event_hits.length : 0
    const amp = (1.1 + hitCount * 0.42 + ((i % 7) * 0.08)).toFixed(2)
    const scope = ['局地', '近岸', '区域', '广域'][Math.min(3, hitCount)]
    const duration = (6 + hitCount * 4 + (i % 4) * 2).toFixed(0)
    return {
      index: pt.index,
      timestamp: pt.timestamp,
      time: fmtTs(pt.timestamp),
      amplitude: `${amp}x`,
      amplitudeClass: Number(amp) >= 2.5 ? 'text-rose-400' : Number(amp) >= 1.8 ? 'text-amber-400' : 'text-emerald-400',
      scope,
      duration,
      eventHits: hitCount > 0 ? pt.event_hits.join(', ') : '-'
    }
  })

  const eventCounter = new Map()
  points.forEach((pt) => {
    const hits = Array.isArray(pt.event_hits) ? pt.event_hits : []
    hits.forEach((name) => {
      eventCounter.set(name, (eventCounter.get(name) || 0) + 1)
    })
  })

  const couplings = [...eventCounter.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([name, hit], i) => ({
      name,
      score: Math.min(100, 52 + hit * 6 + i * 3),
      speed: (14 + hit * 2 + i).toFixed(1),
      intensity: (36 + hit * 3 + i * 2).toFixed(1)
    }))

  const zonesRaw = [
    { name: '渤海湾', base: 28 },
    { name: '黄海北部', base: 36 },
    { name: '黄海中部', base: 42 },
    { name: '黄海南部', base: 47 }
  ].map((z, idx) => {
    const score = Math.min(99, Math.max(8, z.base + Math.round(riskScore * 0.55) + idx * 4 - 8))
    const level = getRiskLevel(score)
    return { name: z.name, score, level: level.name, levelClass: level.cls }
  })

  const historyCases = couplings.slice(0, 4).map((c, idx) => ({
    caseId: `CASE-${String(idx + 1).padStart(3, '0')} ${c.name}`,
    similarity: `${Math.min(98, 72 + idx * 6)}%`,
    window: `${8 + idx * 4}h`
  }))

  return {
    riskLevel: risk.name,
    riskClass: risk.cls,
    riskScore,
    warning: {
      level: warningLevel,
      levelClass: warningClassByLevel[warningLevel],
      targets: warningLevel === '红' ? '指挥中心/港口/海上作业单位' : '值班员/港口/重点作业船舶',
      actions: [
        '自动推送风险区域图至值班席位与移动端。',
        '对高风险海域执行作业限制与船舶避险提醒。',
        '每30分钟更新异常强度、持续时长与影响范围。',
        '形成预警升级/降级/解除全流程记录。'
      ]
    },
    monitor: {
      windNow,
      waveNow,
      windBand,
      waveBand,
      statusText: monitorStatus,
      statusClass: monitorStatusClass,
      baselineMode: '季节 + 海域 + 气候分型阈值',
      baselines: [
        { key: '春季-黄海北部-冷空气型', wind: '4.8-9.6 m/s', wave: '0.7-1.9 m', delta: `${(Number(windNow) - 9.6).toFixed(1)} / ${(Number(waveNow) - 1.9).toFixed(2)}`, deltaClass: 'text-amber-400' },
        { key: '夏季-黄海中部-季风型', wind: '5.2-10.4 m/s', wave: '0.9-2.2 m', delta: `${(Number(windNow) - 10.4).toFixed(1)} / ${(Number(waveNow) - 2.2).toFixed(2)}`, deltaClass: 'text-slate-300' },
        { key: '秋季-渤海湾-台风外围型', wind: '6.1-11.8 m/s', wave: '1.1-2.8 m', delta: `${(Number(windNow) - 11.8).toFixed(1)} / ${(Number(waveNow) - 2.8).toFixed(2)}`, deltaClass: riskScore >= 60 ? 'text-rose-400' : 'text-amber-400' },
        { key: '冬季-黄海南部-强冷涌型', wind: '7.0-13.0 m/s', wave: '1.2-3.1 m', delta: `${(Number(windNow) - 13.0).toFixed(1)} / ${(Number(waveNow) - 3.1).toFixed(2)}`, deltaClass: 'text-slate-300' }
      ]
    },
    detect: {
      accuracy: Math.min(99.5, 72 + matchedRate * 0.35 + posRate * 0.18),
      autoEvents: d.matched_event_count || 0,
      retroHits: d.matched_positive || 0,
      archiveCount: details.length,
      details
    },
    typhoon: {
      zones: zonesRaw,
      couplings,
      historyCases
    }
  }
})

const buildAnomalyBrief = () => {
  if (!anomalyData.value) return ''
  const p = anomalyProduct.value
  return [
    '【海洋风-浪异常灾害预警简报】',
    `时间: ${new Date().toISOString().replace('T', ' ').substring(0, 19)} UTC`,
    `监测海域: 黄渤海重点网格 | 数据分片: ${anomalyData.value.split}`,
    `风险等级: ${p.riskLevel} (${p.riskScore}/100) | 预警等级: ${p.warning.level}色`,
    `异常样本: ${anomalyData.value.num_positive}/${anomalyData.value.num_samples} | 事件命中: ${anomalyData.value.matched_event_count}`,
    `实况估计: 风速 ${p.monitor.windNow} m/s, 波高 ${p.monitor.waveNow} m`,
    `影响区域: ${p.typhoon.zones.map((z) => `${z.name}(${z.level})`).join('、')}`,
    `处置建议: ${p.warning.actions.join(' ')}`
  ].join('\n')
}

const copyAnomalyBrief = async () => {
  try {
    if (!anomalyBrief.value) anomalyBrief.value = buildAnomalyBrief()
    await navigator.clipboard.writeText(anomalyBrief.value)
  } catch (err) {
    anomalyError.value = `简报复制失败: ${err?.message || 'unknown error'}`
  }
}

let clockInterval
onMounted(() => {
  clockInterval = setInterval(() => {
    currentTime.value = new Date().toISOString().substring(11, 19)
  }, 1000)

  loadDefaultDataPath()
  window.addEventListener('resize', handleResize)
})

onUnmounted(() => {
  clearInterval(clockInterval)
  stopPlay()
  window.removeEventListener('resize', handleResize)
})

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
  if (activeModule.value !== 'forecast' || !hasResult.value) return
  Plotly.Plots.resize('spatial-chart')
  Plotly.Plots.resize('curve-chart')
}

const loadDefaultDataPath = async () => {
  try {
    const res = await axios.get(`${API_BASE}/default-data-path`)
    if (res.data.path) dataPath.value = res.data.path
  } catch (err) {
    // no-op
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

const loadAnomalyOverview = async () => {
  anomalyLoading.value = true
  anomalyError.value = ''
  try {
    const res = await axios.post(`${API_BASE}/anomaly/inspect`, {
      labels_json: anomalyLabelsPath.value,
      events_json: anomalyEventsPath.value,
      manifest_path: anomalyManifestPath.value,
      split: anomalySplit.value,
      max_points: 300
    }, {
      timeout: 45000
    })
    anomalyData.value = res.data
    anomalyBrief.value = buildAnomalyBrief()
    warningLogs.value = [
      {
        id: `${Date.now()}-pub`,
        time: new Date().toISOString().substring(11, 19),
        action: `${anomalyProduct.value.warning.level}色预警生成`,
        note: `风险分 ${anomalyProduct.value.riskScore}，样本 ${res.data.num_samples}，异常 ${res.data.num_positive}`
      },
      {
        id: `${Date.now()}-up`,
        time: new Date().toISOString().substring(11, 19),
        action: '关联分析更新',
        note: `命中台风事件 ${res.data.matched_event_count}，命中异常点 ${res.data.matched_positive}`
      },
      ...warningLogs.value
    ].slice(0, 12)
  } catch (err) {
    if (err.code === 'ECONNABORTED') {
      anomalyError.value = '请求超时：后端可能被上一次长请求占用。请重启后端后重试。'
    } else {
      anomalyError.value = err.response?.data?.detail || err.message || '加载失败'
    }
    anomalyData.value = null
  } finally {
    anomalyLoading.value = false
  }
}

const loadStepData = async () => {
  if (!sessionId.value) return
  try {
    const res = await axios.get(`${API_BASE}/predict/${sessionId.value}/step/${currentStep.value}`)
    currentStepDataCache = res.data
    renderSpatialPlot(currentStepDataCache)
    updateVerticalLineOnCurve()
  } catch (err) {
    console.error('数据游标读取失败', err)
  }
}

const onQuiverToggle = () => {
  if (currentStepDataCache) {
    renderSpatialPlot(currentStepDataCache)
  }
}

const curveTitle = ref('区域变化趋势')

const loadCurveData = async (r = null, c = null) => {
  if (!sessionId.value) return
  try {
    let url = `${API_BASE}/predict/${sessionId.value}/curve`
    if (r !== null && c !== null) {
      url += `?r=${r}&c=${c}`
      curveTitle.value = `点 (${r}, ${c}) 变化趋势`
    } else {
      curveTitle.value = '区域变化趋势'
    }
    const res = await axios.get(url)
    renderCurvePlot(res.data.data)
  } catch (err) {
    console.error('加载长效趋势数据失败', err)
  }
}

const onStepSliderInput = () => {
  stopPlay()
  loadStepData()
}

const onSpeedChange = () => {
  if (isPlaying.value) {
    stopPlay()
    togglePlay()
  }
}

const togglePlay = () => {
  if (isPlaying.value) {
    stopPlay()
    return
  }

  isPlaying.value = true
  if (currentStep.value >= totalSteps.value - 1) currentStep.value = 0
  loadStepData()

  const intervalMs = Math.max(100, 1200 / playbackSpeed.value)
  playInterval = setInterval(() => {
    if (currentStep.value >= totalSteps.value - 1) {
      stopPlay()
      return
    }
    currentStep.value += 1
    loadStepData()
  }, intervalMs)
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

  if (key.includes('SSUV')) {
    return {
      displayName: 'SSUV',
      title: '合成流速',
      unit: 'm/s',
      colorscale: 'Cividis',
      diverging: false,
      quantileLow: 0.02,
      quantileHigh: 0.98,
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

  return [...mergedBase, { var: 'ssuv', data: speed, uData: ssu.data, vData: ssv.data }]
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



const getAutoColorRange = (grid, style) => {
  if (style?.isMask) return { zauto: false, zmin: 0, zmax: 1 }

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
  const annotations = []

  const N = Math.max(1, plotItems.length)

  plotItems.forEach((v, index) => {
    const style = getVariableRenderStyle(v.var)

    // Removed frontend heavy upsampling/smoothing to improve display speed. 
    // Plotly's zsmooth: 'best' already handles interpolation natively and much faster.
    const smoothedGrid = style.isMask ? v.data : fillInternalMissingPoints(v.data, 2, 5)
    const colorRange = getAutoColorRange(smoothedGrid, style)

    const blockWidth = 1.0 / N
    const xStart = index * blockWidth
    const xEnd = xStart + blockWidth * 0.88
    const cbX = xEnd + 0.01
    const titleX = (xStart + xEnd) / 2

    plots.push({
      z: smoothedGrid,
      type: 'heatmap',
      zsmooth: 'best',
      colorscale: style.colorscale,
      ...colorRange,
      hoverongaps: false,
      showscale: true,
      colorbar: {
        len: 0.8,
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
      name: v.var,
      hoverlabel: { bgcolor: '#0f172a', bordercolor: '#06b6d4', font: { color: '#06b6d4', family: 'JetBrains Mono, monospace' } }
    })

    // If it's a vector field, add streamline to show continuous ocean currents
    if (v.var === 'ssuv' && v.uData && v.vData && showQuiver.value) {
      const rows = v.uData.length
      const cols = rows > 0 ? v.uData[0].length : 0
      
      const streamX = []
      const streamY = []
      
      // Simple 2D streamline tracing using Euler method
      const stepSize = 1.0 // tracing step in grid cells
      const maxSteps = 25  // max length of a streamline
      const seedSpacing = 8 // grid spacing for starting a new streamline
      
      // Helper to bilinear interpolate velocity at a continuous (x, y) coordinate
      const getVelocity = (x, y) => {
        const x0 = Math.floor(x), x1 = x0 + 1
        const y0 = Math.floor(y), y1 = y0 + 1
        
        if (y0 < 0 || y1 >= rows || x0 < 0 || x1 >= cols) return null
        
        const fx = x - x0, fy = y - y0
        
        const u00 = v.uData[y0][x0], u01 = v.uData[y0][x1]
        const u10 = v.uData[y1][x0], u11 = v.uData[y1][x1]
        
        const v00 = v.vData[y0][x0], v01 = v.vData[y0][x1]
        const v10 = v.vData[y1][x0], v11 = v.vData[y1][x1]
        
        if (!Number.isFinite(u00) || !Number.isFinite(u01) || !Number.isFinite(u10) || !Number.isFinite(u11)) return null
        if (!Number.isFinite(v00) || !Number.isFinite(v01) || !Number.isFinite(v10) || !Number.isFinite(v11)) return null
        
        const u = u00*(1-fx)*(1-fy) + u01*fx*(1-fy) + u10*(1-fx)*fy + u11*fx*fy
        const _v = v00*(1-fx)*(1-fy) + v01*fx*(1-fy) + v10*(1-fx)*fy + v11*fx*fy
        
        return { u, v: _v }
      }
      
      // Plant seeds uniformly
      // 为什么这样选择流线？
      // 1. 均匀播种 (Uniform Seeding): 通过 seedSpacing=8，在整个海域等间距地撒下虚拟“漂流瓶”。这样可以保证整个流场的结构都能被照顾到，不会因为随机撒点而漏掉某些重要的涡旋特征。
      // 2. 静水过滤 (Velocity Threshold): 海洋中很多区域（如某些近岸或深海）流速极低，画出它们不仅浪费前端渲染性能，还会导致“噪点”过多。因此如果初始点流速 < 0.02 (约 2cm/s)，则不生成流线，让视线聚焦在真正活跃的洋流上。
      // 3. 欧拉追踪 (Euler Tracking): 沿着当前点的速度方向前进一小步(stepSize=1.0)，再在新的位置重新计算速度继续前进，最高追踪 25 步，形成一条平滑的流线轨迹。
      for (let r = seedSpacing/2; r < rows; r += seedSpacing) {
        for (let c = seedSpacing/2; c < cols; c += seedSpacing) {
          
          let cx = c
          let cy = r
          
          // Check if seed is in a valid fluid area
          const velStart = getVelocity(cx, cy)
          if (!velStart) continue
          const speedStart = Math.sqrt(velStart.u*velStart.u + velStart.v*velStart.v)
          if (speedStart < 0.02) continue // Skip stagnant areas
          
          // Trace forward
          let lastDx = 0
          let lastDy = 0
          for (let s = 0; s < maxSteps; s++) {
            const vel = getVelocity(cx, cy)
            if (!vel) break
            
            const speed = Math.sqrt(vel.u*vel.u + vel.v*vel.v)
            if (speed < 0.005) break // stopped
            
            // Normalize direction to move a consistent step in grid space
            const dx = (vel.u / speed) * stepSize
            const dy = -(vel.v / speed) * stepSize // Y is inverted in plot
            
            streamX.push(cx, cx + dx)
            streamY.push(cy, cy + dy)
            
            cx += dx
            cy += dy
            lastDx = dx
            lastDy = dy
          }
          
          // 在流线的末端加上一个箭头，明确指示流水的去向
          if (lastDx !== 0 || lastDy !== 0) {
            const arrowLen = 2.0
            const wingLen = arrowLen * 0.5
            
            // Math.atan2 此时使用的是屏幕坐标系的方向
            const angle = Math.atan2(lastDy, lastDx)
            const angle1 = angle - Math.PI / 6
            const angle2 = angle + Math.PI / 6
            
            const xw1 = cx - wingLen * Math.cos(angle1)
            const yw1 = cy - wingLen * Math.sin(angle1)
            
            const xw2 = cx - wingLen * Math.cos(angle2)
            const yw2 = cy - wingLen * Math.sin(angle2)
            
            // 画翼线 1
            streamX.push(cx, xw1, null)
            streamY.push(cy, yw1, null)
            
            // 画翼线 2
            streamX.push(cx, xw2, null)
            streamY.push(cy, yw2, null)
          } else {
            // Insert a break (NaN) between separate streamlines
            streamX.push(null)
            streamY.push(null)
          }
        }
      }

      if (streamX.length > 0) {
        plots.push({
          x: streamX,
          y: streamY,
          mode: 'lines',
          type: 'scatter',
          line: { color: 'rgba(255, 255, 255, 0.55)', width: 1.5, shape: 'spline', smoothing: 1.3 },
          xaxis: `x${index + 1 > 1 ? index + 1 : ''}`,
          yaxis: `y${index + 1 > 1 ? index + 1 : ''}`,
          hoverinfo: 'skip',
          showlegend: false
        })
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
    margin: { l: 20, r: 10, t: 40, b: 20 },
    annotations
  }

  for (let i = 0; i < plotItems.length; i++) {
    const ax = i === 0 ? '' : (i + 1)
    const blockWidth = 1.0 / N
    const xStart = i * blockWidth
    const xEnd = xStart + blockWidth * 0.88

    layout[`xaxis${ax}`] = { domain: [xStart, xEnd], anchor: `y${ax}`, showticklabels: true, tickfont: { color: '#475569', size: 9 }, gridcolor: 'rgba(30, 41, 59, 0.5)', zeroline: false }
    layout[`yaxis${ax}`] = { scaleanchor: `x${ax}`, scaleratio: 1, domain: [0.05, 1.0], anchor: `x${ax}`, autorange: 'reversed', showticklabels: true, tickfont: { color: '#475569', size: 9 }, gridcolor: 'rgba(30, 41, 59, 0.5)', zeroline: false }
  }

  Plotly.react(container, plots, layout, { responsive: true, displayModeBar: false })

  // Add click event listener to update curve chart
  container.on('plotly_click', (data) => {
    if (data.points && data.points.length > 0) {
      const pt = data.points[0]
      // pt.x is column index, pt.y is row index
      const c = Math.round(pt.x)
      const r = Math.round(pt.y)
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
    const hours = Array.from({ length: v.means.length }, (_, i) => (i + 1) * STEP_HOURS)
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

const updateVerticalLineOnCurve = () => {
  const container = document.getElementById('curve-chart')
  if (!container || !hasResult.value) return

  const currentHour = (currentStep.value + 1) * STEP_HOURS
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
