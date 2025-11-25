<template>
    <div class="react-view">
        <!-- 고정 헤더 영역 -->
        <div class="header-section">
            <!-- 탭 네비게이션 -->
            <div class="tab-navigation">
                <button class="tab-btn" :class="{ active: activeTab === 'input' }" @click="activeTab = 'input'"
                    type="button">
                    <span class="tab-icon">✏️</span>
                    <span class="tab-label">쿼리 입력</span>
                </button>
                <button class="tab-btn" :class="{ active: activeTab === 'summary', disabled: !hasExecutionData }"
                    :disabled="!hasExecutionData" @click="activeTab = 'summary'" type="button">
                    <span class="tab-icon">📊</span>
                    <span class="tab-label">실시간 요약</span>
                    <span v-if="reactStore.isRunning" class="live-badge">LIVE</span>
                </button>
                <button class="tab-btn" :class="{ active: activeTab === 'details', disabled: !reactStore.hasSteps }"
                    :disabled="!reactStore.hasSteps" @click="activeTab = 'details'" type="button">
                    <span class="tab-icon">🔍</span>
                    <span class="tab-label">상세 스텝</span>
                    <span v-if="reactStore.hasSteps" class="step-count">{{ reactStore.steps.length }}</span>
                </button>
            </div>

            <div v-if="reactStore.error" class="error-message">
                <strong>오류:</strong> {{ reactStore.error }}
            </div>
        </div>

        <!-- 메인 컨텐츠 영역 -->
        <div class="content-section">
            <!-- 쿼리 입력 탭 -->
            <transition name="fade" mode="out-in">
                <div v-if="activeTab === 'input'" key="input" class="tab-content">
                    <div class="centered-input">
                        <ReactInput :loading="reactStore.isRunning" :waiting-for-user="reactStore.isWaitingUser"
                            :question-to-user="reactStore.questionToUser" :current-question="reactStore.currentQuestion"
                            @start="handleStart" @respond="handleRespond" @cancel="handleCancel" />
                    </div>
                </div>

                <!-- 실시간 요약 탭 -->
                <div v-else-if="activeTab === 'summary'" key="summary" class="tab-content">
                    <!-- 실행 중 상태 표시 -->
                    <div v-if="reactStore.isRunning && !reactStore.hasSteps" class="loading-state">
                        <div class="spinner"></div>
                        <p>에이전트가 도구를 호출하며 SQL을 구성하고 있습니다...</p>
                    </div>

                    <!-- 요약 패널 -->
                    <ReactSummaryPanel v-if="reactStore.hasSteps || reactStore.partialSql || reactStore.finalSql"
                        :status="reactStore.status" :partial-sql="reactStore.latestPartialSql"
                        :final-sql="reactStore.finalSql" :validated-sql="reactStore.validatedSql"
                        :warnings="reactStore.warnings" :execution-result="reactStore.executionResult"
                        :collected-metadata="reactStore.collectedMetadata"
                        :remaining-tool-calls="reactStore.remainingToolCalls" :current-step="reactStore.steps.length"
                        :is-running="reactStore.isRunning" :latest-step="reactStore.latestStep" />
                </div>

                <!-- 상세 스텝 타임라인 탭 -->
                <div v-else-if="activeTab === 'details'" key="details" class="tab-content">
                    <ReactStepTimeline v-if="reactStore.hasSteps" :steps="reactStore.steps"
                        :is-running="reactStore.isRunning" />
                </div>
            </transition>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import ReactInput from '../components/react/ReactInput.vue'
import ReactStepTimeline from '../components/react/ReactStepTimeline.vue'
import ReactSummaryPanel from '../components/react/ReactSummaryPanel.vue'
import { useReactStore } from '../stores/react'

const reactStore = useReactStore()

type TabType = 'input' | 'summary' | 'details'
const activeTab = ref<TabType>('input')

// 실행 데이터가 있는지 확인
const hasExecutionData = computed(() =>
    reactStore.hasSteps || reactStore.partialSql || reactStore.finalSql || reactStore.isRunning
)

// 쿼리 시작 시 자동으로 요약 탭으로 전환
watch(() => reactStore.isRunning, (isRunning, wasRunning) => {
    if (isRunning && !wasRunning) {
        // 실행 시작됨 -> 요약 탭으로 자동 전환
        activeTab.value = 'summary'
    }
})

// 실행이 완료되고 스텝이 있으면 상세 탭 활성화 가능
watch(() => reactStore.hasSteps, (hasSteps) => {
    if (hasSteps && activeTab.value === 'input' && !reactStore.isRunning) {
        // 완료 후 입력 탭에 있다면 요약으로 이동
        activeTab.value = 'summary'
    }
})

// 취소 시 입력 탭으로 돌아가기
watch(() => reactStore.status, (status) => {
    if (status === 'idle' && activeTab.value !== 'input' && !hasExecutionData.value) {
        activeTab.value = 'input'
    }
})

async function handleStart(
    question: string,
    options: { maxToolCalls: number; maxSqlSeconds: number }
) {
    await reactStore.start(question, options)
    // 시작 후 자동으로 요약 탭으로 전환 (watch에서 처리됨)
}

async function handleRespond(answer: string) {
    await reactStore.continueWithResponse(answer)
}

function handleCancel() {
    reactStore.cancel()
    // 사용자가 취소한 경우 입력 탭으로 돌아갈 수 있도록 유지
}
</script>

<style scoped>
.react-view {
    display: flex;
    flex-direction: column;
    height: 100vh;
    background: linear-gradient(135deg, #f5f7fa 0%, #e8eef5 100%);
    overflow: hidden;
}

.header-section {
    flex-shrink: 0;
    background: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.06);
    padding: 0.75rem 1.5rem;
    z-index: 10;
}

/* 탭 네비게이션 */
.tab-navigation {
    display: flex;
    gap: 0.5rem;
    padding: 0.5rem;
    background: #f7fafc;
    border-radius: 10px;
    justify-content: center;
    flex-wrap: wrap;
}

.tab-btn {
    flex: 1;
    min-width: 120px;
    max-width: 200px;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 0.65rem 1rem;
    background: white;
    border: 2px solid transparent;
    border-radius: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #4a5568;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
}

.tab-btn:not(:disabled):hover {
    background: #edf2f7;
    border-color: #cbd5e0;
    transform: translateY(-2px);
}

.tab-btn.active {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    border-color: #667eea;
    color: white;
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
}

.tab-btn.disabled,
.tab-btn:disabled {
    opacity: 0.4;
    cursor: not-allowed;
}

.tab-btn:disabled:hover {
    transform: none;
    background: white;
}

.tab-icon {
    font-size: 1.1rem;
}

.tab-label {
    font-size: 0.9rem;
}

.live-badge {
    position: absolute;
    top: -8px;
    right: -8px;
    background: #ef4444;
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    padding: 0.25rem 0.5rem;
    border-radius: 12px;
    animation: pulse-badge 2s ease-in-out infinite;
}

.step-count {
    position: absolute;
    top: -8px;
    right: -8px;
    background: #667eea;
    color: white;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 0.25rem 0.5rem;
    border-radius: 50%;
    min-width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
}

@keyframes pulse-badge {

    0%,
    100% {
        opacity: 1;
        transform: scale(1);
    }

    50% {
        opacity: 0.8;
        transform: scale(1.05);
    }
}

.error-message {
    background: #ffebee;
    border-left: 4px solid #f44336;
    padding: 1rem 1.25rem;
    margin-top: 1rem;
    border-radius: 8px;
    color: #c62828;
    animation: slideIn 0.3s ease-out;
}

.content-section {
    flex: 1;
    overflow-y: auto;
    padding: 2rem;
    max-width: 1600px;
    width: 100%;
    margin: 0 auto;
}

.tab-content {
    animation: fadeIn 0.3s ease-out;
}

/* 쿼리 입력 중앙 정렬 */
.centered-input {
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 400px;
    width: 100%;
    max-width: 1200px;
    margin: 0 auto;
    padding: 2rem;
}

/* Fade 트랜지션 */
.fade-enter-active,
.fade-leave-active {
    transition: opacity 0.25s ease-out, transform 0.25s ease-out;
}

.fade-enter-from {
    opacity: 0;
    transform: translateY(10px);
}

.fade-leave-to {
    opacity: 0;
    transform: translateY(-10px);
}

.loading-state {
    text-align: center;
    padding: 4rem 2rem;
    background: white;
    border-radius: 16px;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
}

.loading-state p {
    color: #666;
    font-size: 1.1rem;
    margin: 0;
}

.spinner {
    width: 60px;
    height: 60px;
    margin: 0 auto 1.5rem;
    border: 4px solid #f3f3f3;
    border-top: 4px solid #667eea;
    border-radius: 50%;
    animation: spin 1s cubic-bezier(0.68, -0.55, 0.265, 1.55) infinite;
}

@keyframes spin {
    0% {
        transform: rotate(0deg);
    }

    100% {
        transform: rotate(360deg);
    }
}

@keyframes slideIn {
    from {
        opacity: 0;
        transform: translateY(-10px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* 스크롤바 스타일링 */
.content-section::-webkit-scrollbar {
    width: 8px;
}

.content-section::-webkit-scrollbar-track {
    background: transparent;
}

.content-section::-webkit-scrollbar-thumb {
    background: rgba(0, 0, 0, 0.2);
    border-radius: 4px;
}

.content-section::-webkit-scrollbar-thumb:hover {
    background: rgba(0, 0, 0, 0.3);
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .header-section {
        padding: 0.5rem 1rem;
    }

    .content-section {
        padding: 1rem;
    }

    .tab-navigation {
        gap: 0.5rem;
        padding: 0.5rem;
    }

    .tab-btn {
        min-width: 100px;
        padding: 0.5rem 0.75rem;
        font-size: 0.85rem;
    }

    .tab-icon {
        font-size: 1rem;
    }

    .tab-label {
        font-size: 0.85rem;
    }

    .centered-input {
        min-height: 300px;
        padding: 1rem;
        max-width: 100%;
    }
}
</style>
