import { WorkerMessageInit } from "./worker.types";

export class WorkerManager {
  private workers: Worker[];
  private instantWorkers: Worker[] = [];
  private onMessageHandler: ((event: MessageEvent) => void) | undefined;
  private messageHandlers: ((event: MessageEvent) => void)[] = [];
  private workerJobs: Map<Worker, number> = new Map();

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

  async queueTask(
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    data: { type: string } & { [key: string]: any },
    transferables?: Transferable[],
    instant = false
  ) {
    const worker = this.selectWorker(instant);

    if (!worker) {
      console.error("❌ No worker selected! Cannot queue task.");
      return;
    }

    try {
      if (transferables) {
        worker.postMessage(data, transferables);
      } else {
        worker.postMessage(data);
      }
    } catch (error) {
      console.error("❌ Error posting message to worker:", error);
    }

    this.workerJobs.set(worker, (this.workerJobs.get(worker) ?? 0) + 1);
  }

  private _onMessageHandler(event: MessageEvent) {
    // Call primary handler
    this.onMessageHandler?.(event);

    // Call all additional handlers
    this.messageHandlers.forEach(handler => handler(event));

    // Decrement job count only when mesh is updated (final step of chunk processing)
    if (event.data.type === "chunkMeshUpdated") {
      const worker = event.target as Worker;
      const currentJobs = this.workerJobs.get(worker) ?? 0;
      if (currentJobs > 0) {
        this.workerJobs.set(worker, currentJobs - 1);
      }
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
}
