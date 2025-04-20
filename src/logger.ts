const enabledDomains: string[] = [];

// biome-ignore lint/suspicious/noExplicitAny: <explanation>
const log = (domain: string, message: string, ...args: any[]) => {
  if (enabledDomains.includes(domain)) {
    console.log(`[${domain}] ${message}`, ...args);
  }
};

// biome-ignore lint/suspicious/noExplicitAny: <explanation>
log.error = (domain: string, message: string, ...args: any[]) => {
  if (enabledDomains.includes(domain)) {
    console.error(`[${domain}] ${message}`, ...args);
  }
};

// biome-ignore lint/suspicious/noExplicitAny: <explanation>
log.warn = (domain: string, message: string, ...args: any[]) => {
  if (enabledDomains.includes(domain)) {
    console.warn(`[${domain}] ${message}`, ...args);
  }
};

export default log;
