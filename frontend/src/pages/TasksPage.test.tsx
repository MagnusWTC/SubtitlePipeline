import { act, render, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { Task, TaskActionResponse, TaskListResponse } from '../api'
import {
  resolveTaskPollingInterval,
  TASK_CANCEL_POLL_INTERVAL_MS,
  TASK_POLL_INTERVAL_MS,
  TasksPage,
} from './TasksPage'

const apiMocks = vi.hoisted(() => ({
  cancelTask: vi.fn(),
  checkResumeFeasibility: vi.fn(),
  getTasks: vi.fn(),
  retryTask: vi.fn(),
}))

vi.mock('../api', () => apiMocks)

function makeTask(overrides: Partial<Task> = {}): Task {
  return {
    id: 1,
    file_path: 'task.mp4',
    status: 'processing',
    stage: 'transcribe',
    progress: 35,
    retry_count: 0,
    max_retries: 3,
    cancel_requested: false,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
    ...overrides,
  }
}

function makeTaskList(items: Task[]): TaskListResponse {
  const statusCounts = items.reduce<Record<string, number>>((counts, task) => {
    counts[task.status] = (counts[task.status] ?? 0) + 1
    return counts
  }, {})

  return {
    items,
    total: items.length,
    page: 1,
    page_size: 20,
    status_counts: statusCounts,
  }
}

function renderTasksPage() {
  return render(
    <MemoryRouter>
      <TasksPage />
    </MemoryRouter>,
  )
}

describe('TasksPage cancellation', () => {
  beforeEach(() => {
    apiMocks.cancelTask.mockReset()
    apiMocks.checkResumeFeasibility.mockReset()
    apiMocks.getTasks.mockReset()
    apiMocks.retryTask.mockReset()
    apiMocks.getTasks.mockResolvedValue(makeTaskList([]))
    apiMocks.checkResumeFeasibility.mockResolvedValue({ can_resume: false, missing: [] })
  })

  it('calls the cancel API and immediately disables the button as 取消中', async () => {
    const user = userEvent.setup()
    apiMocks.getTasks.mockResolvedValue(makeTaskList([makeTask()]))
    let resolveCancel: (value: TaskActionResponse) => void = () => undefined
    apiMocks.cancelTask.mockReturnValue(
      new Promise<TaskActionResponse>((resolve) => {
        resolveCancel = resolve
      }),
    )

    renderTasksPage()

    const cancelButton = await screen.findByRole('button', { name: '取消' })
    await user.click(cancelButton)

    expect(apiMocks.cancelTask).toHaveBeenCalledWith(1)
    expect(screen.getByRole('button', { name: '取消中' })).toBeDisabled()

    await act(async () => {
      resolveCancel({
        id: 1,
        status: 'processing',
        stage: 'transcribe',
        progress: 35,
        cancel_requested: true,
        outcome: 'cancelling',
        message: '取消请求已接受，等待当前阶段安全停止',
      })
      await Promise.resolve()
    })

    await waitFor(() => expect(screen.getByRole('button', { name: '取消中' })).toBeDisabled())
  })

  it('shows an actionable error and restores the cancel button when the API fails', async () => {
    const user = userEvent.setup()
    apiMocks.getTasks.mockResolvedValue(makeTaskList([makeTask()]))
    apiMocks.cancelTask.mockRejectedValue(new Error('任务状态已变化'))

    renderTasksPage()

    await user.click(await screen.findByRole('button', { name: '取消' }))

    expect(await screen.findByRole('alert')).toHaveTextContent('取消任务失败：任务状态已变化')
    const retryCancelButton = screen.getByRole('button', { name: '取消' })
    expect(retryCancelButton).toBeEnabled()

    await user.click(retryCancelButton)
    expect(apiMocks.cancelTask).toHaveBeenCalledTimes(2)
  })

  it('renders processing tasks with cancel_requested as 取消中 on the first load', async () => {
    apiMocks.getTasks.mockResolvedValue(makeTaskList([makeTask({ cancel_requested: true })]))

    renderTasksPage()

    expect(await screen.findByText('task.mp4')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '取消中' })).toBeDisabled()
    expect(screen.queryByRole('button', { name: '取消' })).not.toBeInTheDocument()
  })

  it('applies a cancelled response to a pending task without waiting for polling', async () => {
    const user = userEvent.setup()
    apiMocks.getTasks.mockResolvedValue(makeTaskList([makeTask({ status: 'pending', stage: 'queued', progress: 0 })]))
    apiMocks.cancelTask.mockResolvedValue({
      id: 1,
      status: 'cancelled',
      stage: 'queued',
      progress: 0,
      cancel_requested: false,
      outcome: 'cancelled',
      message: '任务已取消',
    } satisfies TaskActionResponse)

    renderTasksPage()

    await user.click(await screen.findByRole('button', { name: '取消' }))

    expect(await screen.findByText('cancelled')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '重新执行' })).toBeEnabled()
    expect(screen.queryByRole('button', { name: '取消中' })).not.toBeInTheDocument()
  })

  it('renders every task status and includes completed and cancelled tasks in the total', async () => {
    apiMocks.getTasks.mockResolvedValue(
      makeTaskList([
        makeTask({ id: 1, file_path: 'pending.mp4', status: 'pending', stage: 'queued', progress: 0 }),
        makeTask({ id: 2, file_path: 'processing.mp4', status: 'processing' }),
        makeTask({ id: 3, file_path: 'failed.mp4', status: 'failed', stage: 'translate', progress: 60 }),
        makeTask({ id: 4, file_path: 'done.mp4', status: 'done', stage: 'output_finalize', progress: 100 }),
        makeTask({ id: 5, file_path: 'cancelled.mp4', status: 'cancelled', stage: 'translate', progress: 60 }),
      ]),
    )

    renderTasksPage()

    expect(await screen.findByText('pending.mp4')).toBeInTheDocument()
    expect(screen.getByText('processing.mp4')).toBeInTheDocument()
    expect(screen.getByText('failed.mp4')).toBeInTheDocument()
    expect(screen.getByText('done.mp4')).toBeInTheDocument()
    expect(screen.getByText('cancelled.mp4')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '全部 5' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: '已完成 1' })).toBeInTheDocument()
  })

  it('resets pagination when filtering completed tasks and allows restarting them', async () => {
    const user = userEvent.setup()
    const completedTask = makeTask({ file_path: 'done.mp4', status: 'done', progress: 100 })
    apiMocks.getTasks.mockImplementation(async (status, page) => ({
      ...makeTaskList(status === 'done' ? [completedTask] : [makeTask()]),
      page,
      total: status === 'done' ? 1 : 21,
      status_counts: { processing: 20, done: 1 },
    }))
    apiMocks.retryTask.mockResolvedValue(undefined)

    renderTasksPage()

    await screen.findByText('task.mp4')
    await user.click(screen.getByRole('button', { name: '下一页' }))
    await waitFor(() => expect(apiMocks.getTasks).toHaveBeenLastCalledWith(undefined, 2, 20))
    await user.click(screen.getByRole('button', { name: '已完成 1' }))

    expect(await screen.findByText('done.mp4')).toBeInTheDocument()
    expect(apiMocks.getTasks).toHaveBeenLastCalledWith('done', 1, 20)
    expect(screen.getByRole('button', { name: '上一页' })).toBeDisabled()
    expect(screen.getByRole('button', { name: '下一页' })).toBeDisabled()
    await user.click(screen.getByRole('button', { name: '重新执行' }))
    expect(apiMocks.retryTask).toHaveBeenCalledWith(1, 'restart')

    await user.click(screen.getByRole('button', { name: '全部 21' }))
    expect(await screen.findByText('task.mp4')).toBeInTheDocument()
    expect(apiMocks.getTasks).toHaveBeenLastCalledWith(undefined, 1, 20)
  })

  it('uses a one-second polling interval while cancellation is pending', async () => {
    vi.useFakeTimers()
    try {
      apiMocks.getTasks.mockResolvedValue(makeTaskList([makeTask({ cancel_requested: true })]))

      renderTasksPage()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(0)
      })

      expect(screen.getByText('task.mp4')).toBeInTheDocument()
      apiMocks.getTasks.mockClear()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(TASK_CANCEL_POLL_INTERVAL_MS - 1)
      })
      expect(apiMocks.getTasks).not.toHaveBeenCalled()

      await act(async () => {
        await vi.advanceTimersByTimeAsync(1)
      })
      expect(apiMocks.getTasks).toHaveBeenCalled()
    } finally {
      vi.useRealTimers()
    }
  })

  it('keeps the normal three-second interval when nothing is cancelling', () => {
    expect(resolveTaskPollingInterval([makeTask()])).toBe(TASK_POLL_INTERVAL_MS)
    expect(resolveTaskPollingInterval([makeTask({ cancel_requested: true })])).toBe(TASK_CANCEL_POLL_INTERVAL_MS)
  })
})
