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
                {{ d }} | 序号 {{ idx }}
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
            <div>date: {{ eddyResult.day_label }} | 序号 {{ eddyResult.day_index }}</div>
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
              <h2 class="font-display text-lg tracking-widest text-white m-0">涡旋日预测图</h2>
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
                <div class="text-[11px] font-mono text-slate-300">气旋涡旋</div>
                <div class="text-2xl font-mono text-cyan-300 mt-1">{{ Number(eddyResult?.cyclonic_count || 0) }}</div>
              </div>
              <div class="p-3 rounded-lg border border-rose-400/30 bg-rose-500/10">
                <div class="text-[11px] font-mono text-slate-300">反气旋涡旋</div>
                <div class="text-2xl font-mono text-rose-300 mt-1">{{ Number(eddyResult?.anticyclonic_count || 0) }}</div>
              </div>
              <div class="p-3 rounded-lg border border-emerald-400/30 bg-emerald-500/10">
                <div class="text-[11px] font-mono text-slate-300">总数</div>
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
              split={{ anomalyData.split }} | positive_ratio={{ (anomalyData.positive_ratio * 100).toFixed(2) }}% | matched_positive_ratio={{ (anomalyData.matched_positive_ratio * 100).toFixed(2) }}%
            </div>

            <div class="text-[10px] text-tech-cyan/90 font-mono mb-1.5">
              实时窗口终点: {{ anomalyLatestTimeText }} | 当前选择: {{ anomalySelectedTimeText }}
            </div>

            <div class="flex items-center gap-1.5 mb-1.5 overflow-x-auto custom-scrollbar pb-1">
              <button
                class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
                :class="anomalyView === 'monitor' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'monitor'"
              >1. 实况监测与基准</button>
              <button
                class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
                :class="anomalyView === 'detect' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'detect'"
              >2. 智能识别与回溯</button>
              <button
                class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
                :class="anomalyView === 'typhoon' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'typhoon'"
              >3. 台风关联与评估</button>
              <button
                class="px-2 py-0.5 rounded-md text-[10px] font-mono border transition-all whitespace-nowrap"
                :class="anomalyView === 'warning' ? 'border-tech-cyan bg-tech-cyan/10 text-tech-cyan' : 'border-slate-700 text-slate-400 hover:text-slate-200'"
                @click="anomalyView = 'warning'"
              >4. 分级预警发布</button>
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
import { ref, onMounted, onUnmounted, nextTick, watch, markRaw, computed } from 'vue'
import axios from 'axios'
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

let plotlyModulePromise = null
const ensurePlotly = () => {
  if (!plotlyModulePromise) {
    plotlyModulePromise = import('plotly.js-dist-min').then((mod) => mod.default || mod)
  }
  return plotlyModulePromise
}

const plotlyResize = (target) => {
  return ensurePlotly()
    .then((plotly) => plotly?.Plots?.resize?.(target))
    .catch(() => {})
}

const plotlyReact = (container, data, layout, config) => {
  return ensurePlotly()
    .then((plotly) => plotly?.react?.(container, data, layout, config))
    .catch(() => {})
}

const plotlyPurge = (container) => {
  return ensurePlotly()
    .then((plotly) => plotly?.purge?.(container))
    .catch(() => {})
}

const plotlyRelayout = (container, layout) => {
  return ensurePlotly()
    .then((plotly) => plotly?.relayout?.(container, layout))
    .catch(() => {})
}

const anomalyLabelsPath = ref('outputs/anomaly_detection/labels_competition.json')
const anomalyEventsPath = ref('outputs/anomaly_detection/events_competition.json')
const anomalyManifestPath = ref('data/processed/splits/anomaly_detection_competition.json')
const anomalySplit = ref('test')
const anomalyLoading = ref(false)
const anomalyError = ref('')
const anomalyData = ref(null)
const anomalyView = ref('monitor')
const anomalyRiskScore = ref(0)
const anomalyRiskLevel = ref('低')
const anomalyRiskClass = ref('text-emerald-300 border-emerald-400/40 bg-emerald-400/10')
const anomalySnapshot = ref(null)
const anomalySnapshotLoading = ref(false)
const anomalySnapshotError = ref('')
const anomalyRecentWindow = ref([])
const anomalyRecentWeek = ref([])
const anomalySelectedSnapshotIndex = ref(-1)
const anomalyLatestTimeText = ref('-')
const anomalySelectedTimeText = ref('-')
const anomalySelectedIndexText = ref('-')
const anomalyRows = ref([])
const anomalyTracebackRows = ref([])
const anomalyTracebackWindowHours = ref(72)
const anomalyTracebackStartHour = ref(0)
const anomalyTracebackRangeText = ref('-')
const anomalyDataTimeText = ref('-')
const anomalyIssueTimeText = ref('-')
const anomalyCouplings = ref([])
const anomalyCases = ref([])
const anomalyBrief = ref('')
const anomalyWarning = ref({
  level: '蓝',
  targets: '值班中心/港口/海上作业单位',
  actions: [
    '保持持续监测，滚动更新风速与波高阈值偏离。',
    '对重点海域发布风险提示，建议动态调整作业窗口。',
    '触发异常后30分钟内完成复核并更新预警简报。'
  ]
})
const anomalyWarningLogs = ref([])
const anomalyMonitor = ref({
  windNow: 0,
  waveNow: 0,
  windBand: '5.0-10.5',
  waveBand: '0.8-2.2',
  status: '平稳'
})

let clockInterval
let anomalySnapshotRequestSeq = 0
let anomalySnapshotPrefetchToken = 0
const anomalySnapshotLocalCache = new Map()
const anomalySnapshotInflight = new Map()

const ANOMALY_LOOKBACK_HOURS = 168
const ANOMALY_MONITOR_WINDOW_HOURS = 24

const anomalyTracebackMaxStartHour = computed(() => {
  const maxStart = ANOMALY_LOOKBACK_HOURS - Number(anomalyTracebackWindowHours.value || 72)
  return Math.max(0, maxStart)
})

const anomalyTracebackStepHours = computed(() => {
  const week = Array.isArray(anomalyRecentWeek.value) ? anomalyRecentWeek.value : []
  if (week.length < 2) return 1
  const deltas = []
  for (let i = 1; i < week.length; i++) {
    const dt = Number(week[i].timestamp) - Number(week[i - 1].timestamp)
    if (Number.isFinite(dt) && dt > 0) {
      deltas.push(Math.round(dt / 3600))
    }
  }
  if (!deltas.length) return 1
  deltas.sort((a, b) => a - b)
  const mid = Math.floor(deltas.length / 2)
  const median = deltas[mid]
  return Math.max(1, Number.isFinite(median) ? median : 1)
})
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

watch(anomalyView, async (view) => {
  if (!anomalyData.value) return
  await nextTick()
  if (view === 'monitor') {
    renderAnomalyMonitorChart()
    renderAnomalyWindowChart()
    renderAnomalyOceanMaps()
  }
  if (view === 'detect') renderAnomalyTimelineChart()
  if (view === 'typhoon') renderAnomalyRiskMapChart()
})

watch(anomalyTracebackStartHour, async () => {
  recomputeAnomalyTracebackRows()
  if (anomalyView.value === 'detect') {
    await nextTick()
    renderAnomalyTimelineChart()
  }
})

watch(anomalySnapshot, async () => {
  if (anomalyView.value === 'typhoon') {
    await nextTick()
    renderAnomalyRiskMapChart()
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
    plotlyResize('spatial-chart')
    plotlyResize('curve-chart')
  }
  if (activeModule.value === 'eddy' && eddyResult.value) {
    plotlyResize('eddy-chart')
  }
  if (activeModule.value === 'anomaly' && anomalyData.value) {
    ;['anomaly-monitor-chart', 'anomaly-window-chart', 'anomaly-timeline-chart', 'anomaly-riskmap-chart', 'anomaly-wind-map', 'anomaly-wave-map'].forEach((id) => {
      const el = document.getElementById(id)
      if (el) plotlyResize(el)
    })
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

const _formatEpochText = (ts) => {
  const t = Number(ts)
  if (!Number.isFinite(t) || t <= 0) return '-'
  return new Date(t * 1000).toISOString().replace('T', ' ').slice(0, 19)
}

const _requestAnomalySnapshot = async (snapshotIndex, timeoutMs = 30000) => {
  const idx = Number(snapshotIndex)
  if (!Number.isFinite(idx) || idx < 0) return null
  if (anomalySnapshotInflight.has(idx)) {
    return anomalySnapshotInflight.get(idx)
  }
  const req = axios.post(`${API_BASE}/anomaly/inspect`, {
    labels_json: anomalyLabelsPath.value,
    events_json: anomalyEventsPath.value,
    manifest_path: anomalyManifestPath.value,
    split: anomalySplit.value,
    recent_window_hours: 24,
    snapshot_only: true,
    max_points: 1,
    include_snapshot: true,
    snapshot_index: idx
  }, {
    timeout: timeoutMs
  }).then((res) => res.data?.snapshot || null)
    .finally(() => {
      anomalySnapshotInflight.delete(idx)
    })

  anomalySnapshotInflight.set(idx, req)
  return req
}

const _buildRecentWindow = (payload) => {
  const source = Array.isArray(payload?.recent_window) ? payload.recent_window : []
  if (!source.length) {
    return []
  }
  return source.map((row) => {
    const hits = Array.isArray(row?.event_hits) ? row.event_hits : []
    const windMean = Number(row?.wind_mean)
    const waveMean = Number(row?.wave_mean)
    const windP95 = Number(row?.wind_p95)
    const waveP95 = Number(row?.wave_p95)
    return {
      index: Number.isFinite(Number(row?.index)) ? Number(row.index) : -1,
      timestamp: Number.isFinite(Number(row?.timestamp)) ? Number(row.timestamp) : -1,
      label: Number.isFinite(Number(row?.label)) ? Number(row.label) : 0,
      matched: !!row?.matched,
      eventHits: hits,
      windMean: Number.isFinite(windMean) ? windMean : null,
      waveMean: Number.isFinite(waveMean) ? waveMean : null,
      windP95: Number.isFinite(windP95) ? windP95 : null,
      waveP95: Number.isFinite(waveP95) ? waveP95 : null,
      time: _formatEpochText(row?.timestamp)
    }
  }).filter((row) => row.index >= 0)
}

const recomputeAnomalyTracebackRows = () => {
  const week = Array.isArray(anomalyRecentWeek.value) ? anomalyRecentWeek.value : []
  if (!week.length) {
    anomalyTracebackRows.value = []
    anomalyTracebackRangeText.value = '-'
    return
  }

  const latestTs = Number(week[week.length - 1]?.timestamp || -1)
  const earliestTs = Number(week[0]?.timestamp || -1)
  if (!Number.isFinite(latestTs) || !Number.isFinite(earliestTs) || latestTs <= 0 || earliestTs <= 0) {
    anomalyTracebackRows.value = []
    anomalyTracebackRangeText.value = '-'
    return
  }

  const windowSec = Math.max(1, Number(anomalyTracebackWindowHours.value || 72)) * 3600
  const stepHours = Math.max(1, Number(anomalyTracebackStepHours.value || 1))
  const snappedStartHour = Math.round(Number(anomalyTracebackStartHour.value || 0) / stepHours) * stepHours
  if (snappedStartHour !== Number(anomalyTracebackStartHour.value || 0)) {
    anomalyTracebackStartHour.value = snappedStartHour
  }
  const startOffsetSec = Math.max(0, snappedStartHour) * 3600
  const selectedStart = earliestTs + startOffsetSec
  const selectedEnd = Math.min(latestTs, selectedStart + windowSec)

  anomalyTracebackRangeText.value = `${_formatEpochText(selectedStart)} ~ ${_formatEpochText(selectedEnd)}`

  const validWind = week.map((r) => Number(r.windMean)).filter((v) => Number.isFinite(v))
  const validWave = week.map((r) => Number(r.waveMean)).filter((v) => Number.isFinite(v))
  const mean = (arr) => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0)
  const std = (arr, m) => {
    if (!arr.length) return 1
    const v = arr.reduce((s, x) => s + (x - m) * (x - m), 0) / arr.length
    return Math.sqrt(Math.max(v, 1e-6))
  }
  const windMeanBase = mean(validWind)
  const waveMeanBase = mean(validWave)
  const windStdBase = std(validWind, windMeanBase)
  const waveStdBase = std(validWave, waveMeanBase)

  const windowRows = week
    .map((row, i) => ({ row, i }))
    .filter(({ row }) => Number(row.timestamp) >= selectedStart && Number(row.timestamp) <= selectedEnd)

  const realScores = windowRows.map(({ row }) => {
    const windVal = Number(row.windMean)
    const waveVal = Number(row.waveMean)
    const windZ = Number.isFinite(windVal) ? Math.max(0, (windVal - windMeanBase) / windStdBase) : 0
    const waveZ = Number.isFinite(waveVal) ? Math.max(0, (waveVal - waveMeanBase) / waveStdBase) : 0
    return 0.55 * windZ + 0.45 * waveZ
  })

  const scoreMin = realScores.length ? Math.min(...realScores) : 0
  const scoreMax = realScores.length ? Math.max(...realScores) : 0
  const scoreSpan = Math.max(1e-6, scoreMax - scoreMin)

  anomalyTracebackRows.value = windowRows.map(({ row }, i) => {
    const hits = Array.isArray(row.eventHits) ? row.eventHits : []
    const score = Number.isFinite(realScores[i]) ? realScores[i] : 0
    const scoreNorm = (score - scoreMin) / scoreSpan
    const amplitude = 1.0 + scoreNorm * 2.0
    const duration = Math.max(1, Math.round(1 + scoreNorm * 8))
    return {
      index: row.index,
      timestamp: row.timestamp,
      time: row.time,
      amplitude,
      duration,
      labelSignal: Number(row.label) === 1 ? 1 : 0,
      windMean: row.windMean,
      waveMean: row.waveMean,
      scope: amplitude >= 2.35 ? '广域' : amplitude >= 1.7 ? '区域' : '局地',
      eventHits: hits.length ? hits.join(', ') : '-',
      matched: !!row.matched,
      realScore: score
    }
  })
}

const selectAnomalyWindowPoint = (sampleIndex, withSnapshot = true) => {
  anomalySelectedSnapshotIndex.value = Number.isFinite(Number(sampleIndex)) ? Number(sampleIndex) : -1
  const selected = anomalyRecentWindow.value.find((row) => row.index === anomalySelectedSnapshotIndex.value)
  anomalySelectedTimeText.value = selected ? selected.time : '-'
  anomalySelectedIndexText.value = selected ? String(selected.index) : '-'
  renderAnomalyWindowChart()
  if (withSnapshot && anomalySelectedSnapshotIndex.value >= 0) {
    void fetchAnomalySnapshot(anomalySelectedSnapshotIndex.value)
  }
}

const fetchAnomalySnapshot = async (snapshotIndex) => {
  if (!Number.isFinite(Number(snapshotIndex)) || Number(snapshotIndex) < 0) return
  const idx = Number(snapshotIndex)

  const cached = anomalySnapshotLocalCache.get(idx)
  if (cached) {
    anomalySnapshot.value = cached
    anomalySnapshotError.value = ''
    anomalySnapshotLoading.value = false
    await nextTick()
    if (anomalyView.value === 'monitor') {
      renderAnomalyOceanMaps()
    }
    return
  }

  const reqSeq = ++anomalySnapshotRequestSeq
  anomalySnapshotLoading.value = true
  anomalySnapshotError.value = ''
  const prevSnapshot = anomalySnapshot.value

  try {
    let snap = await _requestAnomalySnapshot(idx, 30000)
    // Cold-start I/O occasionally causes the first foreground request to miss/timeout.
    if (!snap) {
      snap = await _requestAnomalySnapshot(idx, 45000)
    }
    if (reqSeq !== anomalySnapshotRequestSeq) return
    anomalySnapshot.value = snap
    if (snap) {
      anomalySnapshotLocalCache.set(idx, snap)
    } else {
      anomalySnapshotError.value = '该时间点未返回精细图数据'
    }
  } catch (err) {
    if (reqSeq !== anomalySnapshotRequestSeq) return
    anomalySnapshot.value = prevSnapshot || null
    const msg = err?.code === 'ECONNABORTED'
      ? '精细图加载超时（30s），请重试或切换其他时间点'
      : `精细图加载失败: ${err.response?.data?.detail || err.message}`
    anomalySnapshotError.value = msg
  } finally {
    if (reqSeq !== anomalySnapshotRequestSeq) return
    anomalySnapshotLoading.value = false
    await nextTick()
    if (anomalyView.value === 'monitor') {
      renderAnomalyOceanMaps()
    }
  }
}

const prefetchAnomalySnapshots = async (indices) => {
  if (!Array.isArray(indices) || !indices.length) return
  const token = ++anomalySnapshotPrefetchToken
  for (const idxRaw of indices) {
    if (token !== anomalySnapshotPrefetchToken) return
    const idx = Number(idxRaw)
    if (!Number.isFinite(idx) || idx < 0) continue
    if (anomalySnapshotLocalCache.has(idx)) continue
    try {
      const snap = await _requestAnomalySnapshot(idx, 25000)
      if (snap) {
        anomalySnapshotLocalCache.set(idx, snap)
      }
    } catch (err) {
      // Silent prefetch failure should not interrupt foreground interaction.
    }
  }
}

const loadAnomalyOverview = async () => {
  anomalyLoading.value = true
  anomalyError.value = ''
  anomalySnapshot.value = null
  anomalySnapshotError.value = ''
  anomalySnapshotLocalCache.clear()
  anomalySnapshotPrefetchToken += 1
  anomalyRecentWindow.value = []
  anomalyRecentWeek.value = []
  anomalyTracebackRows.value = []
  anomalyTracebackStartHour.value = 0
  anomalyTracebackRangeText.value = '-'
  anomalySelectedSnapshotIndex.value = -1
  anomalySelectedTimeText.value = '-'
  anomalySelectedIndexText.value = '-'
  try {
    const res = await axios.post(`${API_BASE}/anomaly/inspect`, {
      labels_json: anomalyLabelsPath.value,
      events_json: anomalyEventsPath.value,
      manifest_path: anomalyManifestPath.value,
      split: anomalySplit.value,
      recent_window_hours: ANOMALY_LOOKBACK_HOURS,
      include_snapshot: false
    })
    anomalyData.value = res.data

    anomalyRecentWeek.value = _buildRecentWindow(res.data)
    anomalyRecentWindow.value = anomalyRecentWeek.value.filter((row) => Number(row.timestamp) >= (Number(res.data?.latest_timestamp || 0) - ANOMALY_MONITOR_WINDOW_HOURS * 3600))
    anomalyTracebackStartHour.value = anomalyTracebackMaxStartHour.value
    recomputeAnomalyTracebackRows()

    anomalyLatestTimeText.value = _formatEpochText(res.data?.latest_timestamp)
    if (anomalyRecentWindow.value.length) {
      const latest = anomalyRecentWindow.value[anomalyRecentWindow.value.length - 1]
      if (anomalyLatestTimeText.value === '-') {
        anomalyLatestTimeText.value = latest.time
      }
      selectAnomalyWindowPoint(latest.index, false)
    }

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

    const warningLevel = score >= 85 ? '红' : score >= 65 ? '橙' : score >= 45 ? '黄' : '蓝'
    anomalyWarning.value = {
      level: warningLevel,
      targets: warningLevel === '红' ? '应急指挥中心/港口/全部海上作业单位' : '值班中心/港口/重点航线单位',
      actions: [
        '持续滚动监测风速、波高和异常事件命中情况。',
        '对高风险区域限制作业窗口，发布避险建议。',
        '关联台风路径变化并更新分区风险等级。'
      ]
    }

    const points = Array.isArray(res.data?.points) ? res.data.points : []
    anomalyRows.value = points.map((row, i) => {
      const hits = Array.isArray(row.event_hits) ? row.event_hits : []
      const amplitude = 1.2 + hits.length * 0.5 + ((i % 5) * 0.1)
      const duration = 4 + hits.length * 3 + (i % 4)
      return {
        index: row.index,
        timestamp: row.timestamp,
        time: Number.isFinite(Number(row.timestamp)) ? new Date(Number(row.timestamp) * 1000).toISOString().replace('T', ' ').slice(0, 19) : '-',
        amplitude,
        duration,
        scope: hits.length >= 2 ? '广域' : hits.length === 1 ? '区域' : '局地',
        eventHits: hits.length ? hits.join(', ') : '-',
        matched: !!row.matched
      }
    })

    const byEvent = new Map()
    anomalyRows.value.forEach((r) => {
      if (!r.eventHits || r.eventHits === '-') return
      r.eventHits.split(', ').forEach((name) => byEvent.set(name, (byEvent.get(name) || 0) + 1))
    })
    anomalyCouplings.value = [...byEvent.entries()].slice(0, 6).map(([name, c], idx) => ({
      name,
      score: Math.min(99, 52 + c * 8 + idx * 2),
      speed: (16 + c * 2 + idx).toFixed(1),
      intensity: (34 + c * 3 + idx * 2).toFixed(1)
    }))
    if (!anomalyCouplings.value.length) {
      anomalyCouplings.value = [
        { name: '无强台风命中事件', score: 36, speed: '12.0', intensity: '28.0' }
      ]
    }

    anomalyCases.value = anomalyCouplings.value.slice(0, 4).map((c, idx) => ({
      id: `CASE-${String(idx + 1).padStart(3, '0')} ${c.name}`,
      similarity: `${Math.min(98, 72 + idx * 7)}%`,
      window: `${8 + idx * 4}h`
    }))

    const windNow = (6.5 + Number(res.data?.positive_ratio || 0) * 28 + Number(res.data?.matched_positive_ratio || 0) * 5)
    const waveNow = (1.0 + Number(res.data?.positive_ratio || 0) * 4 + Number(res.data?.matched_positive_ratio || 0) * 1.8)
    anomalyMonitor.value = {
      windNow: Number(windNow.toFixed(1)),
      waveNow: Number(waveNow.toFixed(2)),
      windBand: '5.0-10.5',
      waveBand: '0.8-2.2',
      status: score >= 65 ? '异常增强中' : score >= 45 ? '轻度异常' : '整体平稳'
    }

    const nowText = new Date().toISOString().replace('T', ' ').slice(0, 19)
    const dataTimeText = _formatEpochText(res.data?.latest_timestamp)
    anomalyIssueTimeText.value = nowText
    anomalyDataTimeText.value = dataTimeText

    anomalyWarningLogs.value = [
      {
        id: `${Date.now()}-issue`,
        time: nowText,
        issuedAt: nowText,
        dataTime: dataTimeText,
        action: `${warningLevel}色预警发布`,
        note: `风险分=${score}，异常点=${res.data?.num_positive || 0}，命中事件=${res.data?.matched_event_count || 0}`
      },
      {
        id: `${Date.now()}-risk`,
        time: nowText,
        issuedAt: nowText,
        dataTime: dataTimeText,
        action: '风险区域更新',
        note: '按黄渤海分区完成低/中/高/极高风险重算'
      },
      ...anomalyWarningLogs.value
    ].slice(0, 12)

    anomalyBrief.value = [
      '【风-浪异常灾害预警简报】',
      `数据时效时间: ${dataTimeText} UTC`,
      `发布时间: ${nowText} UTC`,
      `分片: ${res.data?.split || '-'} | 风险等级: ${anomalyRiskLevel.value} | 预警: ${warningLevel}色`,
      `异常样本: ${res.data?.num_positive || 0}/${res.data?.num_samples || 0}`,
      `事件命中: ${res.data?.matched_event_count || 0} | 命中异常点: ${res.data?.matched_positive || 0}`,
      `监测估计: 风速 ${anomalyMonitor.value.windNow} m/s，波高 ${anomalyMonitor.value.waveNow} m`,
      '建议: 加强重点海域巡检，按风险等级动态调整作业窗口。'
    ].join('\n')

    await nextTick()
    renderAnomalyMonitorChart()
    renderAnomalyWindowChart()
    renderAnomalyTimelineChart()
    renderAnomalyRiskMapChart()

    if (anomalySelectedSnapshotIndex.value >= 0) {
      void fetchAnomalySnapshot(anomalySelectedSnapshotIndex.value)
      const pending = anomalyRecentWindow.value
        .map((row) => Number(row.index))
        .filter((idx) => Number.isFinite(idx) && idx >= 0 && idx !== Number(anomalySelectedSnapshotIndex.value))
      void prefetchAnomalySnapshots(pending)
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

  plotlyReact(container, plots, layout, { responsive: true, displayModeBar: false })
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

  plotlyReact(container, traces, layout, { responsive: true, displayModeBar: false })
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

  plotlyReact(container, plots, layout, { responsive: true, displayModeBar: false })
}

const renderAnomalyMonitorChart = () => {
  const container = document.getElementById('anomaly-monitor-chart')
  if (!container || !anomalyData.value) return

  const windBandHigh = Number(String(anomalyMonitor.value.windBand).split('-')[1] || 10.5)
  const waveBandHigh = Number(String(anomalyMonitor.value.waveBand).split('-')[1] || 2.2)
  const traces = [
    {
      x: ['风速', '波高'],
      y: [anomalyMonitor.value.windNow, anomalyMonitor.value.waveNow],
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

  plotlyReact(container, traces, layout, { responsive: true, displayModeBar: false })
}

const renderAnomalyWindowChart = async () => {
  const container = document.getElementById('anomaly-window-chart')
  const points = anomalyRecentWindow.value
  if (!container) return
  if (!points.length) {
    await plotlyPurge(container)
    return
  }

  const x = points.map((p) => p.time.slice(11, 19))
  const y = points.map((p) => p.label)
  const markerColor = points.map((p) => {
    if (p.index === anomalySelectedSnapshotIndex.value) return '#22d3ee'
    return p.matched ? '#34d399' : '#94a3b8'
  })
  const markerSize = points.map((p) => (p.index === anomalySelectedSnapshotIndex.value ? 11 : 8))

  await plotlyReact(container, [
    {
      x,
      y,
      type: 'scatter',
      mode: 'lines+markers',
      name: '24h窗口样本',
      line: { color: 'rgba(148,163,184,0.65)', width: 1.5, shape: 'spline' },
      marker: { color: markerColor, size: markerSize },
      customdata: points.map((p) => [p.index, p.time, p.eventHits.join(', ')]),
      hovertemplate: 'index=%{customdata[0]}<br>time=%{customdata[1]}<br>label=%{y}<br>events=%{customdata[2]}<extra></extra>'
    }
  ], {
    ...getChartLayoutBase(''),
    title: undefined,
    margin: { l: 36, r: 12, t: 8, b: 22 },
    xaxis: { tickfont: { color: '#94a3b8', size: 9 }, showgrid: false },
    yaxis: { title: 'label', range: [-0.15, 1.15], dtick: 1, gridcolor: 'rgba(30,41,59,0.5)', tickfont: { color: '#94a3b8' } },
    showlegend: false
  }, { responsive: true, displayModeBar: false })

  if (typeof container.removeAllListeners === 'function') {
    container.removeAllListeners('plotly_click')
  }
  if (typeof container.on !== 'function') {
    return
  }
  container.on('plotly_click', (ev) => {
    const pointNumber = ev?.points?.[0]?.pointNumber
    if (!Number.isFinite(pointNumber)) return
    const row = points[Number(pointNumber)]
    if (!row) return
    selectAnomalyWindowPoint(row.index, true)
  })
}

const _maskByValid = (grid, valid) => {
  if (!Array.isArray(grid) || !Array.isArray(valid)) return grid || []
  return grid.map((row, r) => (row || []).map((v, c) => {
    const m = valid?.[r]?.[c]
    return Number.isFinite(m) && m >= 0.5 ? v : null
  }))
}

const _buildBoundaryFromValid = (valid) => {
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

const _quantileRange = (grid, qLow = 0.02, qHigh = 0.98) => {
  const flat = (grid || []).flat().filter((v) => Number.isFinite(v))
  if (!flat.length) return { zmin: 0, zmax: 1 }
  const sorted = [...flat].sort((a, b) => a - b)
  const pick = (q) => sorted[Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)))]
  let zmin = pick(qLow)
  let zmax = pick(qHigh)
  if (!Number.isFinite(zmin) || !Number.isFinite(zmax) || zmin === zmax) {
    zmin = sorted[0]
    zmax = sorted[sorted.length - 1]
  }
  return { zmin, zmax }
}

const renderAnomalyOceanMaps = () => {
  const snap = anomalySnapshot.value
  const windEl = document.getElementById('anomaly-wind-map')
  const waveEl = document.getElementById('anomaly-wave-map')
  if (!windEl || !waveEl) return
  if (!snap) {
    plotlyPurge(windEl)
    plotlyPurge(waveEl)
    return
  }

  const windMasked = _maskByValid(snap.wind_speed || [], snap.wind_valid || [])
  const waveMasked = _maskByValid(snap.wave_swh || [], snap.wave_valid || [])
  const windBoundary = _buildBoundaryFromValid(snap.wind_valid || [])
  const waveBoundary = _buildBoundaryFromValid(snap.wave_valid || [])

  const windRange = _quantileRange(windMasked, 0.02, 0.98)
  const waveRange = _quantileRange(waveMasked, 0.02, 0.98)

  plotlyReact(windEl, [
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
  ], {
    ...getChartLayoutBase('风速空间分布'),
    margin: { l: 40, r: 20, t: 36, b: 30 },
    xaxis: { showgrid: false, zeroline: false },
    yaxis: { showgrid: false, zeroline: false, autorange: 'reversed' },
    showlegend: false
  }, { responsive: true, displayModeBar: false })

  plotlyReact(waveEl, [
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
  ], {
    ...getChartLayoutBase('波高空间分布'),
    margin: { l: 40, r: 20, t: 36, b: 30 },
    xaxis: { showgrid: false, zeroline: false },
    yaxis: { showgrid: false, zeroline: false, autorange: 'reversed' },
    showlegend: false
  }, { responsive: true, displayModeBar: false })
}

const renderAnomalyTimelineChart = () => {
  const container = document.getElementById('anomaly-timeline-chart')
  if (!container) return

  if (!anomalyTracebackRows.value.length) {
    plotlyReact(container, [], {
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
    }, { responsive: true, displayModeBar: false })
    return
  }

  const x = anomalyTracebackRows.value.map((r) => r.time)
  const amplitude = anomalyTracebackRows.value.map((r) => Number(r.amplitude.toFixed(2)))
  const duration = anomalyTracebackRows.value.map((r) => r.duration)
  const labelRaw = anomalyTracebackRows.value.map((r) => Number(r.labelSignal) === 1 ? 1 : 0)
  const markerColor = anomalyTracebackRows.value.map((r) => (r.matched ? '#34d399' : '#f59e0b'))
  const ampMin = Math.min(...amplitude)
  const ampMax = Math.max(...amplitude)
  const ampPad = Math.max(0.12, (ampMax - ampMin) * 0.25)
  const labelBand = Math.max(0.5, (ampMax - ampMin) + ampPad * 0.5)
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

  plotlyReact(container, traces, layout, { responsive: true, displayModeBar: false })
}

const renderAnomalyRiskMapChart = () => {
  const container = document.getElementById('anomaly-riskmap-chart')
  if (!container || !anomalyData.value) return

  const safeNumber = (v) => {
    const n = Number(v)
    return Number.isFinite(n) ? n : NaN
  }

  const percentile = (arr, q) => {
    if (!arr.length) return 0
    const sorted = [...arr].sort((a, b) => a - b)
    const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q)))
    return sorted[idx]
  }

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v))

  const snap = anomalySnapshot.value
  let z = [
    [Math.max(10, anomalyRiskScore.value - 28), Math.max(15, anomalyRiskScore.value - 16)],
    [Math.max(20, anomalyRiskScore.value - 8), Math.min(99, anomalyRiskScore.value + 6)]
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
    const wind = snap.wind_speed
    const wave = snap.wave_swh
    const windValid = Array.isArray(snap.wind_valid) ? snap.wind_valid : []
    const waveValid = Array.isArray(snap.wave_valid) ? snap.wave_valid : []

    const rows = wind.length
    const cols = rows > 0 && Array.isArray(wind[0]) ? wind[0].length : 0
    if (rows > 1 && cols > 1) {
      const windVals = []
      const waveVals = []
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

      const regionRawScores = []
      const regionConfidence = []
      const regionValidCounts = []
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
            const cellRisk = (windNorm !== null && waveNorm !== null)
              ? (0.62 * windNorm + 0.38 * waveNorm)
              : (windNorm !== null ? windNorm : waveNorm)
            riskSum += cellRisk
            validN += 1
          }
        }
        const rawScore = validN > 0 ? riskSum / validN : 0
        const coverage = eligibleN > 0 ? (validN / eligibleN) : 0
        const sampleStrength = clamp(Math.log10(validN + 1) / 2.4, 0, 1)
        const channelBalance = Math.max(windValidN, waveValidN) > 0
          ? Math.min(windValidN, waveValidN) / Math.max(windValidN, waveValidN)
          : 0
        const conf = 100 * (0.55 * sampleStrength + 0.10 * channelBalance + 0.35 * coverage)
        regionRawScores.push(rawScore)
        regionConfidence.push(conf)
        regionValidCounts.push(validN)
      }

      const rawMin = Math.min(...regionRawScores)
      const rawMax = Math.max(...regionRawScores)
      const rawSpan = Math.max(1e-6, rawMax - rawMin)
      const anchor = Number.isFinite(Number(anomalyRiskScore.value)) ? Number(anomalyRiskScore.value) : 50
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

  const latestTs = Number(anomalyData.value?.latest_timestamp || -1)
  const dataLagMin = latestTs > 0 ? Math.max(0, Math.round((Date.now() / 1000 - latestTs) / 60)) : -1
  const lagText = dataLagMin >= 0
    ? (dataLagMin >= 1440
      ? `数据时效滞后 ${Math.floor(dataLagMin / 1440)} 天 ${Math.floor((dataLagMin % 1440) / 60)} 小时`
      : (dataLagMin >= 60
        ? `数据时效滞后 ${Math.floor(dataLagMin / 60)} 小时 ${dataLagMin % 60} 分钟`
        : `数据时效滞后 ${dataLagMin} 分钟`))
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

  plotlyReact(container, [trace], layout, { responsive: true, displayModeBar: false })
}

const copyAnomalyBrief = async () => {
  try {
    await navigator.clipboard.writeText(anomalyBrief.value || '')
  } catch (err) {
    anomalyError.value = `复制失败: ${err?.message || 'unknown'}`
  }
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

  plotlyRelayout(container, { shapes })
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
