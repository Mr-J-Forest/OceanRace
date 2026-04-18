import { ref, computed } from 'vue'
import { anomalyData, anomalyProduct } from './useAnomaly'
import { eddyResult } from './useEddy'

export function useManager() {

  const managerNoticeDraft = ref('')
  const managerNoticeTarget = ref('港口调度中心')
  const managerNoticeChannel = ref('平台公告')
  const managerPublishRecords = ref<{ id: string; time: string; target: string; channel: string; status: string }[]>([])
  const managerAgencyByZone: Record<string, string> = {
    渤海湾: '渤海海事局值班中心',
    黄海北部: '北部海域联合指挥部',
    黄海中部: '黄海中部海洋监测站',
    黄海南部: '南部海事应急协同组'
  }

  const managerRiskSnapshot = computed(() => {
    if (anomalyData.value) {
      const p = anomalyProduct.value
      return {
        level: p.warning.level,
        riskScore: p.riskScore,
        highRiskZoneCount: p.typhoon.zones.filter((z: { score: number }) => z.score >= 60).length,
        focusWindow: p.riskScore >= 60 ? '未来12小时重点关注' : '未来24小时持续关注',
        statusText: p.monitor.statusText
      }
    }

    const totalEddy = Number(eddyResult.value?.cyclonic_count || 0) + Number(eddyResult.value?.anticyclonic_count || 0)
    const level = totalEddy >= 80 ? '橙' : totalEddy >= 40 ? '黄' : '蓝'
    return {
      level,
      riskScore: Math.min(95, Math.max(20, totalEddy)),
      highRiskZoneCount: totalEddy >= 40 ? 2 : 1,
      focusWindow: totalEddy >= 40 ? '未来12小时重点关注' : '未来24小时例行巡检',
      statusText: totalEddy >= 40 ? '涡旋活动增强' : '海况整体平稳'
    }
  })

  const managerZoneResponsibilities = computed(() => {
    const baseZones = anomalyData.value
      ? anomalyProduct.value.typhoon.zones
      : [
          { name: '渤海湾', score: 42, level: '中', levelClass: 'text-amber-400' },
          { name: '黄海北部', score: 58, level: '中', levelClass: 'text-amber-400' },
          { name: '黄海中部', score: 35, level: '低', levelClass: 'text-sky-400' },
          { name: '黄海南部', score: 28, level: '低', levelClass: 'text-sky-400' }
        ]
    return baseZones.map((z: { name: string; score: number; level?: string; levelClass?: string }) => ({
      ...z,
      agency: managerAgencyByZone[z.name] || '海洋综合值班中心',
      action: z.score >= 60 ? '发布重点预警并启动现场联动' : z.score >= 40 ? '加强巡查并滚动更新风险' : '维持常态监测'
    }))
  })

  const managerOneLineConclusion = computed(() => {
    const snap = managerRiskSnapshot.value
    const topZone = [...managerZoneResponsibilities.value].sort((a, b) => b.score - a.score)[0]
    const zoneName = topZone?.name || '重点海域'
    return `当前${zoneName}风险较高（${snap.level}色预警倾向，风险分${snap.riskScore}），建议${snap.focusWindow}并优先保障港口与近海作业安全。`
  })

  const managerBriefContent = computed(() => {
    const snap = managerRiskSnapshot.value
    const zoneLines = managerZoneResponsibilities.value
      .map((z: { name: string; score: number; agency: string; action: string }) => `- ${z.name}: 风险分${z.score} | 责任单位 ${z.agency} | 建议 ${z.action}`)
      .join('\n')
    const publishLines = managerPublishRecords.value.length
      ? managerPublishRecords.value.map((r) => `- ${r.time} | ${r.target} | ${r.channel} | ${r.status}`).join('\n')
      : '- 暂无发布记录'
    return [
      '【海洋风险管理简报】',
      `时间: ${new Date().toISOString().replace('T', ' ').substring(0, 19)} UTC`,
      `风险总览: ${snap.level}色倾向 | 风险分 ${snap.riskScore} | 高风险区域 ${snap.highRiskZoneCount} 个`,
      `一句话结论: ${managerOneLineConclusion.value}`,
      '',
      '【区域责任制】',
      zoneLines,
      '',
      '【预警发布记录】',
      publishLines
    ].join('\n')
  })

  const generateManagerNotice = () => {
    const snap = managerRiskSnapshot.value
    const top = [...managerZoneResponsibilities.value].sort((a, b) => b.score - a.score)[0]
    managerNoticeDraft.value = [
      '【海洋风险提示】',
      `当前风险等级建议：${snap.level}色预警（风险分 ${snap.riskScore}）。`,
      `重点关注区域：${top?.name || '重点海域'}，责任单位：${top?.agency || '海洋综合值班中心'}。`,
      `建议措施：${top?.action || '维持常态监测并及时复核。'}`,
      `请${managerNoticeTarget.value}通过${managerNoticeChannel.value}落实处置并在2小时内反馈。`
    ].join('\n')
  }

  const publishManagerNotice = () => {
    if (!managerNoticeDraft.value.trim()) generateManagerNotice()
    managerPublishRecords.value = [
      {
        id: `${Date.now()}-${Math.random().toString(36).slice(2, 7)}`,
        time: new Date().toISOString().replace('T', ' ').substring(0, 19),
        target: managerNoticeTarget.value,
        channel: managerNoticeChannel.value,
        status: '已发布'
      },
      ...managerPublishRecords.value
    ].slice(0, 20)
  }

  const exportManagerBrief = () => {
    const blob = new Blob([managerBriefContent.value], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `manager_brief_${new Date().toISOString().substring(0, 10)}.txt`
    document.body.appendChild(a)
    a.click()
    a.remove()
    URL.revokeObjectURL(url)
  }

  return {
    managerNoticeDraft,
    managerNoticeTarget,
    managerNoticeChannel,
    managerPublishRecords,
    managerRiskSnapshot,
    managerZoneResponsibilities,
    managerOneLineConclusion,
    managerBriefContent,
    generateManagerNotice,
    publishManagerNotice,
    exportManagerBrief
  }
}
