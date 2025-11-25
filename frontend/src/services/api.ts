import axios from 'axios'

type ImportMetaWithEnv = ImportMeta & {
  env?: Record<string, string | undefined>
}

// Gateway를 통한 접근: http://localhost:9090/api
const metaEnv = ((import.meta as ImportMetaWithEnv).env ?? {}) as Record<string, string | undefined>
const API_BASE = metaEnv.VITE_API_URL || '/api'

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json'
  }
})

export interface AskRequest {
  question: string
  limit?: number
  visual_pref?: string[]
}

export interface AskResponse {
  sql: string
  table: {
    columns: string[]
    rows: any[][]
    row_count: number
    execution_time_ms: number
  }
  charts: Chart[]
  provenance: {
    tables: string[]
    columns: string[]
    neo4j_paths: string[]
    vector_matches: Array<{ node: string; score: number }>
    prompt_snapshot_id: string
  }
  perf: {
    embedding_ms: number
    graph_search_ms: number
    llm_ms: number
    sql_ms: number
    total_ms: number
  }
}

export interface Chart {
  title: string
  type: string
  description: string
  vega_lite?: any
}

export interface TableInfo {
  name: string
  schema: string
  description: string
  column_count: number
}

export interface ColumnInfo {
  name: string
  table_name: string
  dtype: string
  nullable: boolean
  description: string
}

export interface FeedbackRequest {
  prompt_snapshot_id: string
  original_sql: string
  corrected_sql?: string
  rating: number
  notes?: string
  approved: boolean
}

export interface ReactSQLCompleteness {
  is_complete: boolean
  missing_info: string
  confidence_level: string
}

export interface ReactToolCallModel {
  name: string
  raw_parameters_xml: string
  parameters: Record<string, any>
}

export interface ReactStepModel {
  iteration: number
  reasoning: string
  metadata_xml: string
  partial_sql: string
  sql_completeness: ReactSQLCompleteness
  tool_call: ReactToolCallModel
  tool_result?: string | null
  llm_output: string
}

export interface ReactExecutionResult {
  columns: string[]
  rows: any[][]
  row_count: number
  execution_time_ms: number
}

export interface ReactResponseModel {
  status: 'completed' | 'needs_user_input'
  final_sql?: string | null
  validated_sql?: string | null
  execution_result?: ReactExecutionResult | null
  steps: ReactStepModel[]
  collected_metadata: string
  partial_sql: string
  remaining_tool_calls: number
  session_state?: string | null
  question_to_user?: string | null
  warnings?: string[] | null
}

export interface ReactRequest {
  question: string
  dbms?: string | null
  max_tool_calls?: number
  execute_final_sql?: boolean
  max_iterations?: number | null
  session_state?: string | null
  user_response?: string | null
  max_sql_seconds?: number
  prefer_language?: string
}

export type ReactStreamEvent =
  | {
    event: 'step'
    step: ReactStepModel
    state: Record<string, any>
  }
  | {
    event: 'completed'
    response: ReactResponseModel
    state: Record<string, any>
  }
  | {
    event: 'needs_user_input'
    response: ReactResponseModel
    state: Record<string, any>
  }
  | {
    event: 'error'
    message: string
  }

export const apiService = {
  // 자연어 질의
  async ask(request: AskRequest): Promise<AskResponse> {
    const { data } = await api.post('/ask', request)
    return data
  },

  // 테이블 목록
  async getTables(search?: string, schema?: string, limit: number = 50): Promise<TableInfo[]> {
    const { data } = await api.get('/meta/tables', {
      params: { search, schema, limit }
    })
    return data
  },

  // 테이블 컬럼
  async getTableColumns(tableName: string, schema: string = 'public'): Promise<ColumnInfo[]> {
    const { data } = await api.get(`/meta/tables/${tableName}/columns`, {
      params: { schema }
    })
    return data
  },

  // 컬럼 검색
  async searchColumns(search: string, limit: number = 50): Promise<ColumnInfo[]> {
    const { data } = await api.get('/meta/columns', {
      params: { search, limit }
    })
    return data
  },

  // 피드백 제출
  async submitFeedback(feedback: FeedbackRequest): Promise<any> {
    const { data } = await api.post('/feedback', feedback)
    return data
  },

  // 피드백 통계
  async getFeedbackStats(): Promise<any> {
    const { data } = await api.get('/feedback/stats')
    return data
  },

  // 스키마 인제스천
  async ingestSchema(dbName: string, schema: string, clearExisting: boolean = false): Promise<any> {
    const { data } = await api.post('/ingest', {
      db_name: dbName,
      schema,
      clear_existing: clearExisting
    })
    return data
  },

  // 헬스체크
  async healthCheck(): Promise<any> {
    const { data } = await api.get('/health')
    return data
  },

  async *reactStream(
    request: ReactRequest,
    options: { signal?: AbortSignal } = {}
  ): AsyncGenerator<ReactStreamEvent, void, unknown> {
    const response = await fetch(`${API_BASE}/react`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(request),
      signal: options.signal
    })

    if (!response.ok || !response.body) {
      const message = await response.text()
      throw new Error(message || 'ReAct 요청에 실패했습니다.')
    }

    const reader = response.body.getReader()
    const decoder = new TextDecoder('utf-8')
    let buffer = ''

    try {
      while (true) {
        const { done, value } = await reader.read()
        if (done) {
          break
        }
        buffer += decoder.decode(value, { stream: true })
        let newlineIndex = buffer.indexOf('\n')
        while (newlineIndex !== -1) {
          const rawLine = buffer.slice(0, newlineIndex).trim()
          buffer = buffer.slice(newlineIndex + 1)
          if (rawLine) {
            try {
              const parsed = JSON.parse(rawLine) as ReactStreamEvent
              yield parsed
            } catch (error) {
              console.warn('ReAct 이벤트 파싱 실패', error, rawLine)
            }
          }
          newlineIndex = buffer.indexOf('\n')
        }
      }

      const remaining = buffer.trim()
      if (remaining) {
        try {
          const parsed = JSON.parse(remaining) as ReactStreamEvent
          yield parsed
        } catch (error) {
          console.warn('ReAct 마지막 이벤트 파싱 실패', error, remaining)
        }
      }
    } finally {
      reader.releaseLock()
    }
  }
}

// Export the axios instance for direct use
export { api }
export default apiService

