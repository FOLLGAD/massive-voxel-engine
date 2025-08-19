import { WorkerMessageInit } from "./worker.types";

export class WorkerManager {
  private workers: Worker[];
  private instantWorkers: Worker[] = [];
  private onMessageHandler: ((event: MessageEvent) => void) | undefined;
  private messageHandlers: ((event: MessageEvent) => void)[] = [];
  private workerJobs: Map<Worker, number> = new Map();
  
  // --- Task tracking & scheduling ---
  private nextTaskId = 1;
  private maxJobsPerWorker = 1;
  private queuedTasks: Array<{
    id: number;
    data: { type: string } & { [key: string]: unknown };
    transferables?: Transferable[];
    instant: boolean;
    enqueueTimeMs: number;
  }> = [];
  private activeTasks: Map<number, {
    id: number;
    data: { type: string } & { [key: string]: unknown };
    worker: Worker;
    startTimeMs: number;
  }> = new Map();
  private workerActiveTaskIds: Map<Worker, number[]> = new Map();

  constructor(numWorkers: number = navigator.hardwareConcurrency || 4, initMsg: WorkerMessageInit) {
    const numInstantWorkers = 2;
    this.workers = this.init(numWorkers - numInstantWorkers, initMsg);
    this.instantWorkers = this.init(numInstantWorkers, initMsg);
  }

  private init(numWorkers: number = navigator.hardwareConcurrency || 4, initMsg: WorkerMessageInit) {
    const workers: Worker[] = [];
    for (let i = 0; i < numWorkers; i++) {
      try {
        const worker = new Worker("./src/worker.js");
        worker.onmessage = (event: MessageEvent) => this._onMessageHandler(event);
        worker.onerror = (error) => {
          console.error(`Worker ${i} error:`, error);
        };
        worker.postMessage(initMsg);
        workers.push(worker);
      } catch (error) {
        console.error(`❌ Failed to create worker ${i}:`, error);
      }
    }
    return workers;
  }

  private selectWorker(instant = false) {
    const workers = instant
      ? [...this.instantWorkers, ...this.workers]
      : this.workers;

    if (workers.length === 0) {
      console.error("❌ No workers available!");
      return null;
    }

    let min = this.workerJobs.get(workers[0]) ?? 0;
    let minWorker = workers[0];
    for (let i = 1; i < workers.length; i++) {
      if (min === 0) break;
      const jobs = this.workerJobs.get(workers[i]) ?? 0;
      if (jobs < min) {
        min = jobs;
        minWorker = workers[i];
      }
    }

    return minWorker;
  }

  private canAcceptMoreWork(worker: Worker | null | undefined): worker is Worker {
    if (!worker) return false;
    const current = this.workerJobs.get(worker) ?? 0;
    return current < this.maxJobsPerWorker;
  }

  private dispatchTaskFromQueue(): boolean {
    if (this.queuedTasks.length === 0) return false;

    // Prioritize instant tasks; then longest queued first
    const now = performance.now();
    this.queuedTasks.sort((a, b) => {
      if (a.instant !== b.instant) return a.instant ? -1 : 1;
      const ageA = now - a.enqueueTimeMs;
      const ageB = now - b.enqueueTimeMs;
      return ageB - ageA; // older first
    });

    for (let i = 0; i < this.queuedTasks.length; i++) {
      const task = this.queuedTasks[i];
      const candidate = this.selectWorker(task.instant);
      if (!this.canAcceptMoreWork(candidate)) continue;

      // Remove from queue and dispatch
      this.queuedTasks.splice(i, 1);
      this.postToWorker(candidate, task.data, task.transferables, task.id);
      return true;
    }

    return false;
  }

  private postToWorker(
    worker: Worker,
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    data: { type: string } & { [key: string]: any },
    transferables: Transferable[] | undefined,
    taskId: number
  ) {
    try {
      if (transferables && transferables.length > 0) {
        worker.postMessage(data, transferables);
      } else {
        worker.postMessage(data);
      }
    } catch (error) {
      console.error("❌ Error posting message to worker:", error);
      return;
    }

    // Track as active
    const startTimeMs = performance.now();
    this.activeTasks.set(taskId, { id: taskId, data, worker, startTimeMs });
    const list = this.workerActiveTaskIds.get(worker) ?? [];
    list.push(taskId);
    this.workerActiveTaskIds.set(worker, list);

    // Increment jobs for the selected worker
    this.workerJobs.set(worker, (this.workerJobs.get(worker) ?? 0) + 1);
  }

  async queueTask(
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    data: { type: string } & { [key: string]: any },
    transferables?: Transferable[],
    instant = false
  ) {
    const taskId = this.nextTaskId++;
    const candidate = this.selectWorker(instant);
    if (this.canAcceptMoreWork(candidate)) {
      this.postToWorker(candidate, data, transferables, taskId);
    } else {
      // Queue the task
      this.queuedTasks.push({ id: taskId, data, transferables, instant, enqueueTimeMs: performance.now() });
    }
  }

  private _onMessageHandler(event: MessageEvent) {
    // Call primary handler
    this.onMessageHandler?.(event);

    // Call all additional handlers
    this.messageHandlers.forEach(handler => handler(event));

    // Decrement job count when a task completes.
    // Our long-running tasks conclude with either 'chunkMeshUpdated' or 'chunksToUnload'.
    if (event.data.type === "chunkMeshUpdated" || event.data.type === "chunksToUnload") {
      const worker = event.target as Worker;
      const currentJobs = this.workerJobs.get(worker) ?? 0;
      if (currentJobs > 0) {
        this.workerJobs.set(worker, currentJobs - 1);
      }

      // Mark the oldest active task for this worker as complete (FIFO)
      const list = this.workerActiveTaskIds.get(worker) ?? [];
      const finishedTaskId = list.shift();
      this.workerActiveTaskIds.set(worker, list);
      if (finishedTaskId !== undefined) {
        this.activeTasks.delete(finishedTaskId);
      }

      // Try to dispatch next queued task
      this.dispatchTaskFromQueue();
    }
  }

  async setMessageHandler(fn: (event: MessageEvent) => void) {
    this.onMessageHandler = fn;
  }

  addMessageHandler(fn: (event: MessageEvent) => void) {
    this.messageHandlers.push(fn);
  }

  removeMessageHandler(fn: (event: MessageEvent) => void) {
    const index = this.messageHandlers.indexOf(fn);
    if (index > -1) {
      this.messageHandlers.splice(index, 1);
    }
  }

  getAvailableWorker(): Worker | null {
    // Return the worker with the least jobs
    const allWorkers = [...this.instantWorkers, ...this.workers];
    if (allWorkers.length === 0) return null;

    let min = this.workerJobs.get(allWorkers[0]) ?? 0;
    let minWorker = allWorkers[0];

    for (let i = 1; i < allWorkers.length; i++) {
      const jobs = this.workerJobs.get(allWorkers[i]) ?? 0;
      if (jobs < min) {
        min = jobs;
        minWorker = allWorkers[i];
      }
    }

    return minWorker;
  }

  // --- UI helpers ---
  getTaskSnapshot(maxQueued: number = 10): {
    queued: Array<{ id: number; type: string; ageMs: number; position?: unknown; }>;
    active: Array<{ id: number; type: string; ageMs: number; position?: unknown; }>;
  } {
    const now = performance.now();
    const queued = this.queuedTasks
      .map(t => ({ id: t.id, type: t.data.type, ageMs: now - t.enqueueTimeMs, position: (t.data as any).position }))
      .sort((a, b) => b.ageMs - a.ageMs)
      .slice(0, maxQueued);
    const active = Array.from(this.activeTasks.values())
      .map(t => ({ id: t.id, type: t.data.type, ageMs: now - t.startTimeMs, position: (t.data as any).position }))
      .sort((a, b) => b.ageMs - a.ageMs);
    return { queued, active };
  }
}
