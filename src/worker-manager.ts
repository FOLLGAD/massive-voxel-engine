export class WorkerManager {
  private workers: Worker[];
  private workerIndex: number;

  constructor(numWorkers: number = navigator.hardwareConcurrency || 4) {
    this.workers = [];
    this.workerIndex = 0;
    this.init(numWorkers);
  }

  private init(numWorkers: number = navigator.hardwareConcurrency || 4) {
    for (let i = 0; i < numWorkers; i++) {
      const worker = new Worker("./worker.js");
      this.workers.push(worker);
    }
  }

  private selectWorker() {
    const worker = this.workers[this.workerIndex];
    this.workerIndex = (this.workerIndex + 1) % this.workers.length;
    return worker;
  }

  async queueTask(
    // biome-ignore lint/suspicious/noExplicitAny: <explanation>
    data: { type: string } & { [key: string]: any },
    transferables?: Transferable[]
  ) {
    const worker = this.selectWorker();
    if (transferables) {
      worker.postMessage(data, transferables);
    } else {
      worker.postMessage(data);
    }
  }

  async onMessageHandler(fn: (event: MessageEvent) => void) {
    for (const worker of this.workers) {
      worker.onmessage = fn;
    }
  }
}
