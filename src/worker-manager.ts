export class WorkerManager {
  private workers: Worker[];
  private instantWorkers: Worker[] = [];
  private onMessageHandler: ((event: MessageEvent) => void) | undefined;
  private workerJobs: Map<Worker, number> = new Map();

  constructor(numWorkers: number = navigator.hardwareConcurrency || 4) {
    const numInstantWorkers = 2;
    this.workers = this.init(numWorkers - numInstantWorkers);
    this.instantWorkers = this.init(numInstantWorkers);
  }

  private init(numWorkers: number = navigator.hardwareConcurrency || 4) {
    const workers: Worker[] = [];
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker("./worker.js");
      workers.push(worker);
    }
    return workers;
  }

  private selectWorker(instant = false) {
    const workers = instant
      ? [...this.instantWorkers, ...this.workers]
      : this.workers;
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
    if (transferables) {
      worker.postMessage(data, transferables);
    } else {
      worker.postMessage(data);
    }
    worker.onmessage = (event: MessageEvent) => this._onMessageHandler(event);
    this.workerJobs.set(worker, (this.workerJobs.get(worker) ?? 0) + 1);
  }

  private _onMessageHandler(event: MessageEvent) {
    this.onMessageHandler?.(event);
    if (event.data.done) {
      this.workerJobs.set(
        event.target as Worker,
        (this.workerJobs.get(event.target as Worker) ?? 0) - 1
      );
    }
  }

  async setMessageHandler(fn: (event: MessageEvent) => void) {
    this.onMessageHandler = fn;
  }
}
