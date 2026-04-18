<template>
<section class="h-full flex gap-6">
        
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

                <button class="tech-btn primary-btn w-full mt-6 flex items-center justify-center gap-2 text-sm" @click="runPrediction">
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
</template>

<script setup>
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
  LineChart
} from 'lucide-vue-next'
import { useForecast } from '../../composables/useForecast'

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
  STEP_HOURS,
  loadDataInfo,
  runPrediction,
  togglePlay,
  onSpeedChange,
  onStepSliderInput,
  onQuiverToggle
} = useForecast()
</script>
