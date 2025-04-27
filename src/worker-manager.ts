export class WorkerManager {
  private workers: Worker[];
  private instantWorkers: Worker[] = [];
  private onMessageHandler: ((event: MessageEvent) => void) | undefined;
  private workerJobs: Map<Worker, number> = new Map();

  constructor(numWorkers: number = navigator.hardwareConcurrency || 4) {
    const numInstantWorkers = 1;
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
    const workers = instant ? this.instantWorkers : this.workers;
    const worker = workers.reduce((minWorker, worker) => {
      return (this.workerJobs.get(worker) ?? 0) <
        (this.workerJobs.get(minWorker) ?? 0)
        ? worker
        : minWorker;
    }, this.workers[0]);

    return worker;
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
