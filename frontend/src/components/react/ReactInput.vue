<template>
    <div class="react-input">
        <!-- 헤더 (쿼리 입력 영역에만 표시) -->
        <div v-if="!waitingForUser" class="input-header">
            <h1>🧠 Neo4j ReAct Text2SQL</h1>
            <p>ReAct 에이전트의 단계별 추론 과정을 실시간으로 확인하세요</p>
        </div>

        <div v-if="!waitingForUser" class="input-container">
            <textarea v-model="question" @keydown.ctrl.enter.prevent="submitQuestion"
                placeholder="질문을 입력하세요... 예: '지난 분기 매출 Top 5 제품을 보여줘'" rows="3" :disabled="loading"></textarea>
            <div class="action-buttons">
                <button v-if="!loading" class="btn-primary" @click="submitQuestion" :disabled="!canSubmitQuestion">
                    <span class="btn-icon">🚀</span>
                    <span class="btn-text">ReAct 실행</span>
                </button>
                <button v-if="loading" class="btn-secondary" type="button" @click="emit('cancel')">
                    <span class="btn-icon">✕</span>
                    <span class="btn-text">중단</span>
                </button>
            </div>
        </div>

        <!-- 고급 설정 (쿼리 입력 시에만 표시) -->
        <div v-if="!waitingForUser" class="settings-section">
            <button class="settings-toggle" type="button" @click="showSettings = !showSettings">
                <span class="toggle-icon">⚙️</span>
                <span class="toggle-text">고급 설정</span>
                <span class="toggle-arrow" :class="{ expanded: showSettings }">▼</span>
            </button>
            <transition name="slide">
                <div v-if="showSettings" class="settings-panel">
                    <div class="setting-item">
                        <label for="maxToolCalls">
                            <span class="setting-label">최대 도구 호출 횟수</span>
                            <span class="setting-hint">에이전트가 사용할 수 있는 도구 호출 수 (1~100)</span>
                        </label>
                        <div class="setting-input-group">
                            <input id="maxToolCalls" v-model.number="maxToolCalls" type="number" min="1" max="100"
                                :disabled="loading" />
                            <span class="setting-unit">회</span>
                        </div>
                    </div>
                    <div class="setting-item">
                        <label for="maxSqlSeconds">
                            <span class="setting-label">SQL 실행 제한 시간</span>
                            <span class="setting-hint">최종 SQL 실행 최대 허용 시간 (1~3600초)</span>
                        </label>
                        <div class="setting-input-group">
                            <input id="maxSqlSeconds" v-model.number="maxSqlSeconds" type="number" min="1" max="3600"
                                :disabled="loading" />
                            <span class="setting-unit">초</span>
                        </div>
                    </div>
                </div>
            </transition>
        </div>

        <div v-else class="follow-up-wrapper">
            <div class="follow-up-question">
                <strong>에이전트 질문:</strong>
                <p>{{ questionToUser }}</p>
            </div>
            <div class="follow-up-container">
                <textarea v-model="userResponse" @keydown.ctrl.enter.prevent="submitUserResponse"
                    placeholder="추가 정보를 입력해주세요. (Ctrl+Enter 전송)" rows="3" :disabled="loading"></textarea>
                <div class="action-buttons">
                    <button class="btn-secondary" type="button" @click="emit('cancel')">
                        <span class="btn-icon">✕</span>
                        <span class="btn-text">중단</span>
                    </button>
                    <button class="btn-primary" @click="submitUserResponse" :disabled="!canSubmitUserResponse">
                        <span class="btn-icon">📤</span>
                        <span class="btn-text">{{ loading ? '전송 중...' : '답변 보내기' }}</span>
                    </button>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'

export interface ReactStartOptions {
    maxToolCalls: number
    maxSqlSeconds: number
}

const emit = defineEmits<{
    start: [question: string, options: ReactStartOptions]
    respond: [answer: string]
    cancel: []
}>()

const props = defineProps<{
    loading: boolean
    waitingForUser: boolean
    questionToUser: string | null
    currentQuestion: string
}>()

const question = ref(props.currentQuestion ?? '')
const userResponse = ref('')
const showSettings = ref(false)
const maxToolCalls = ref(30)
const maxSqlSeconds = ref(60)

const waitingForUser = computed(() => props.waitingForUser)

const canSubmitQuestion = computed(() => !!question.value.trim() && !props.loading)
const canSubmitUserResponse = computed(
    () => !!userResponse.value.trim() && !props.loading
)

watch(
    () => props.currentQuestion,
    newVal => {
        if (!props.loading && !waitingForUser.value) {
            question.value = newVal
        }
    }
)

watch(waitingForUser, isWaiting => {
    if (!isWaiting) {
        userResponse.value = ''
    }
})

function submitQuestion() {
    if (!canSubmitQuestion.value) return
    const trimmed = question.value.trim()
    question.value = trimmed
    emit('start', trimmed, {
        maxToolCalls: maxToolCalls.value,
        maxSqlSeconds: maxSqlSeconds.value
    })
}

function submitUserResponse() {
    if (!canSubmitUserResponse.value) return
    const trimmed = userResponse.value.trim()
    userResponse.value = trimmed
    emit('respond', trimmed)
}
</script>

<style scoped>
.react-input {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
    width: 100%;
}

.input-header {
    text-align: center;
    animation: fadeInDown 0.5s ease-out;
}

.input-header h1 {
    margin: 0 0 0.75rem 0;
    font-size: 2.5rem;
    color: #1a1a1a;
    font-weight: 700;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.input-header p {
    margin: 0;
    font-size: 1rem;
    color: #666;
    line-height: 1.6;
}

@keyframes fadeInDown {
    from {
        opacity: 0;
        transform: translateY(-20px);
    }

    to {
        opacity: 1;
        transform: translateY(0);
    }
}

.input-container,
.follow-up-container {
    display: flex;
    flex-direction: row;
    gap: 1rem;
    align-items: stretch;
}

.follow-up-wrapper {
    display: flex;
    flex-direction: column;
    gap: 1rem;
}

textarea {
    flex: 1;
    padding: 1rem 1.25rem;
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    font-size: 1rem;
    font-family: inherit;
    resize: none;
    height: 240px;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    background: #fafbfc;
    line-height: 1.6;
}

textarea:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    background: white;
}

textarea:disabled {
    background: #f5f5f5;
    cursor: not-allowed;
    opacity: 0.7;
}

.action-buttons {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    align-items: stretch;
    min-width: 150px;
}

/* 쿼리 입력 시 (버튼 1개) - textarea와 높이 맞춤 */
.input-container .action-buttons {
    height: 240px;
}

/* 후속 질문 시 (버튼 2개) - 더 많은 공간 필요 */
.follow-up-container .action-buttons {
    min-height: 120px;
}

.follow-up-container textarea {
    min-height: 120px;
}

.btn-primary {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 1rem 0.75rem;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    font-size: 0.9rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
    position: relative;
    overflow: hidden;
    white-space: nowrap;
}

.btn-primary .btn-icon {
    font-size: 1.8rem;
    line-height: 1;
}

.btn-primary .btn-text {
    font-size: 0.9rem;
    line-height: 1.3;
    font-weight: 600;
}

.btn-primary::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
    transition: left 0.5s;
}

.btn-primary:hover:not(:disabled)::before {
    left: 100%;
}

.btn-primary:hover:not(:disabled) {
    transform: translateX(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.btn-primary:active:not(:disabled) {
    transform: translateX(0);
}

.btn-primary:disabled {
    background: linear-gradient(135deg, #cbd5e0 0%, #a0aec0 100%);
    cursor: not-allowed;
    box-shadow: none;
}

.btn-secondary {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    padding: 1rem 0.75rem;
    background: white;
    color: #4a5568;
    border: 2px solid #e5e7eb;
    border-radius: 12px;
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    white-space: nowrap;
}

.btn-secondary .btn-icon {
    font-size: 1.5rem;
    line-height: 1;
}

.btn-secondary .btn-text {
    font-size: 0.9rem;
    line-height: 1.3;
    font-weight: 500;
}

.btn-secondary:hover:not(:disabled) {
    background: #f7fafc;
    border-color: #cbd5e0;
    transform: translateX(-2px);
}

.follow-up-question {
    background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
    border-left: 4px solid #f59e0b;
    padding: 1.25rem;
    border-radius: 12px;
    color: #78350f;
    box-shadow: 0 2px 8px rgba(245, 158, 11, 0.2);
}

.follow-up-question strong {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 1.05rem;
}

.follow-up-question strong::before {
    content: '💬';
    font-size: 1.2rem;
}

.follow-up-question p {
    margin: 0.75rem 0 0 0;
    white-space: pre-wrap;
    line-height: 1.6;
    color: #92400e;
}

/* 고급 설정 섹션 */
.settings-section {
    margin-top: 0.5rem;
}

.settings-toggle {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.6rem 1rem;
    background: transparent;
    border: 1px dashed #cbd5e0;
    border-radius: 8px;
    color: #718096;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.2s ease;
    width: 100%;
    justify-content: center;
}

.settings-toggle:hover {
    background: #f7fafc;
    border-color: #a0aec0;
    color: #4a5568;
}

.toggle-icon {
    font-size: 1rem;
}

.toggle-text {
    font-weight: 500;
}

.toggle-arrow {
    font-size: 0.7rem;
    transition: transform 0.2s ease;
    margin-left: auto;
}

.toggle-arrow.expanded {
    transform: rotate(180deg);
}

.settings-panel {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 1rem;
    margin-top: 0.75rem;
    padding: 1.25rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
}

.setting-item {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.setting-item label {
    display: flex;
    flex-direction: column;
    gap: 0.15rem;
}

.setting-label {
    font-size: 0.9rem;
    font-weight: 600;
    color: #2d3748;
}

.setting-hint {
    font-size: 0.75rem;
    color: #718096;
}

.setting-input-group {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.setting-input-group input {
    flex: 1;
    padding: 0.6rem 0.75rem;
    border: 2px solid #e2e8f0;
    border-radius: 8px;
    font-size: 0.9rem;
    font-family: inherit;
    transition: all 0.2s ease;
    background: white;
    max-width: 120px;
}

.setting-input-group input:focus {
    outline: none;
    border-color: #667eea;
    box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
}

.setting-input-group input:disabled {
    background: #edf2f7;
    cursor: not-allowed;
    opacity: 0.6;
}

.setting-unit {
    font-size: 0.85rem;
    color: #718096;
    min-width: 24px;
}

/* Slide 트랜지션 */
.slide-enter-active,
.slide-leave-active {
    transition: all 0.25s ease;
    overflow: hidden;
}

.slide-enter-from,
.slide-leave-to {
    opacity: 0;
    max-height: 0;
    margin-top: 0;
    padding: 0 1.25rem;
}

.slide-enter-to,
.slide-leave-from {
    opacity: 1;
    max-height: 200px;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
    .input-header h1 {
        font-size: 1.75rem;
    }

    .input-header p {
        font-size: 0.9rem;
    }

    .input-container,
    .follow-up-container {
        flex-direction: column;
    }

    textarea {
        height: 100px;
    }

    .action-buttons {
        flex-direction: row;
        min-width: unset;
        height: auto;
    }

    .btn-primary,
    .btn-secondary {
        flex: 1;
        flex-direction: row;
        justify-content: center;
        padding: 0.85rem 1rem;
        gap: 0.75rem;
    }

    .btn-primary .btn-icon,
    .btn-secondary .btn-icon {
        font-size: 1.2rem;
    }

    .btn-primary .btn-text,
    .btn-secondary .btn-text {
        font-size: 0.9rem;
    }

    .btn-primary:hover:not(:disabled),
    .btn-secondary:hover:not(:disabled) {
        transform: translateY(-2px);
    }
}
</style>
