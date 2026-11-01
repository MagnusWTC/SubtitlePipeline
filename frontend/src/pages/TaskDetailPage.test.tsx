import { render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter, Route, Routes } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { LogItem, LogResponse, Task } from '../api'
import { TaskDetailPage } from './TaskDetailPage'

const apiMocks = vi.hoisted(() => ({
  getTask: vi.fn(),
  getTaskLogs: vi.fn(),
}))

vi.mock('../api', () => apiMocks)

function makeTask(): Task {
  return {
    id: 7,
    file_path: '/media/example.mkv',
    status: 'failed',
    stage: 'translate',
    progress: 60,
    retry_count: 1,
    max_retries: 1,
    cancel_requested: false,
    error_message: 'translation failed',
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:01:00Z',
  }
}

function makeLog(overrides: Partial<LogItem>): LogItem {
  return {
    id: 1,
    stage: 'translate',
    level: 'ERROR',
    message: 'translation failed',
    details: null,
    timestamp: '2026-01-01T00:01:00Z',
    ...overrides,
  }
}

function makeLogResponse(items: LogItem[]): LogResponse {
  return {
    items,
    total: items.length,
    page: 1,
    page_size: 20,
  }
}

function renderPage() {
  return render(
    <MemoryRouter initialEntries={['/tasks/7']}>
      <Routes>
        <Route path="/tasks/:taskId" element={<TaskDetailPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe('TaskDetailPage log diagnostics', () => {
  beforeEach(() => {
    apiMocks.getTask.mockReset()
    apiMocks.getTaskLogs.mockReset()
    apiMocks.getTask.mockResolvedValue(makeTask())
  })

  it('keeps details collapsed by default and shows key fields after expanding', async () => {
    const user = userEvent.setup()
    apiMocks.getTaskLogs.mockResolvedValue(
      makeLogResponse([
        makeLog({
          details: {
            error_code: 'LLM_RATE_LIMIT',
            http_status: 429,
            finish_reason: 'length',
            retry_mode: 'restart',
            chunk_start_index: 8,
            nested: { retryable: true },
          },
        }),
      ]),
    )

    renderPage()

    const summary = await screen.findByText('查看详情')
    const detailsElement = summary.closest('details')
    if (!detailsElement) {
      throw new Error('expected a details element')
    }
    expect(detailsElement).not.toHaveAttribute('open')

    await user.click(summary)

    expect(detailsElement).toHaveAttribute('open')
    expect(within(detailsElement).getByText('error_code', { selector: 'dt' })).toBeInTheDocument()
    expect(within(detailsElement).getByText('LLM_RATE_LIMIT')).toBeInTheDocument()
    expect(within(detailsElement).getByText('http_status', { selector: 'dt' })).toBeInTheDocument()
    expect(within(detailsElement).getByText('429')).toBeInTheDocument()
    expect(within(detailsElement).getByText('finish_reason', { selector: 'dt' })).toBeInTheDocument()
    expect(within(detailsElement).getByText('length')).toBeInTheDocument()
    expect(screen.getByTestId('log-details-json-1')).toHaveTextContent('"chunk_start_index": 8')
    expect(screen.getByTestId('log-details-json-1')).toHaveTextContent('"retryable": true')
  })

  it('shows a clear empty state without an expander when details are null or empty', async () => {
    apiMocks.getTaskLogs.mockResolvedValue(
      makeLogResponse([
        makeLog({ id: 1, message: 'normal log', details: null }),
        makeLog({ id: 2, message: 'empty detail log', details: {} }),
      ]),
    )

    renderPage()

    expect(await screen.findByText('normal log')).toBeInTheDocument()
    expect(screen.getByText('empty detail log')).toBeInTheDocument()
    expect(screen.getAllByText('无详情')).toHaveLength(2)
    expect(screen.queryByText('查看详情')).not.toBeInTheDocument()
  })
})
