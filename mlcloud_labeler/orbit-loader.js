// Store compressed chunks on disk; keep only nearby decoded chunks in RAM.
const MEMORY_LIMIT = 64 * 1024 * 1024;
const DECODED_LIMIT = 12;
let database;
function openDatabase() {
  if (!database) database = new Promise((resolve, reject) => {
    const request = indexedDB.open('mlcloud-local-orbits-v1', 1);
    const timeout = setTimeout(() => reject(Error('Browser storage is unavailable.')), 3000);
    request.onupgradeneeded = () => request.result.createObjectStore('chunks');
    request.onerror = () => { clearTimeout(timeout); reject(request.error); };
    request.onblocked = () => { clearTimeout(timeout); reject(Error('Browser storage is blocked.')); };
    request.onsuccess = () => { clearTimeout(timeout); resolve(request.result); };
  });
  return database;
}
async function storage() {
  try {
    const db = await openDatabase();
    const operation = (mode, action) => new Promise((resolve, reject) => {
      const tx = db.transaction('chunks', mode), request = action(tx.objectStore('chunks'));
      tx.oncomplete = () => resolve(request.result);
      tx.onerror = tx.onabort = () => reject(tx.error || Error('Browser storage is full or unavailable.'));
    });
    return {
      keys: prefix => operation('readonly', store => store.getAllKeys(IDBKeyRange.bound(prefix, prefix + '\uffff'))),
      get: key => operation('readonly', store => store.get(key)),
      put: (key, blob) => operation('readwrite', store => store.put(blob, key)),
      remove: key => operation('readwrite', store => store.delete(key)),
    };
  } catch {
    // Private browsing can disable IndexedDB. Bound the fallback memory usage.
    const memory = new Map(); let bytes = 0;
    return {
      keys: async () => [],
      get: async key => memory.get(key),
      remove: async key => {const blob=memory.get(key);if(blob)bytes-=blob.size;memory.delete(key);},
      put: async (key, blob) => {
        if (bytes + blob.size > MEMORY_LIMIT) throw Error('Browser storage is unavailable and the memory cache is full. Allow site storage, then reopen the orbit.');
        memory.set(key, blob); bytes += blob.size;
      },
    };
  }
}

export class OrbitLoader {
  constructor(id, metadata, onProgress) {
    this.id = id; this.meta = metadata; this.onProgress = onProgress;
    this.prefix = `${id === 'sample' ? 'sample-v1' : id}/`;
    this.total = Math.ceil(metadata.shape[0] / metadata.chunkFrames);
    this.downloaded = new Set(); this.inflight = new Map(); this.decoded = new Map();
    this.controller = new AbortController(); this.current = 0; this.error = ''; this.closed = false;
    this.ready = this.initialize();
  }
  key(index) { return this.prefix + index; }
  frameCount(index) { return Math.min(this.meta.chunkFrames, this.meta.shape[0] - index * this.meta.chunkFrames); }
  async initialize() {
    this.store = await storage();
    try {
      for (const key of await this.store.keys(this.prefix)) {
        const n = Number(key.slice(this.prefix.length));
        if (Number.isInteger(n) && n >= 0 && n < this.total) this.downloaded.add(n);
      }
    } catch(e) { this.error = e.message; }
    this.report();
  }
  report() {
    if (this.closed) return;
    const frames = [...this.downloaded].reduce((n, index) => n + this.frameCount(index), 0);
    this.onProgress({ frames, totalFrames: this.meta.shape[0], complete: this.downloaded.size === this.total, error: this.error });
  }
  async clear() {this.close();await this.ready;await Promise.allSettled([...this.inflight.values()]);for(const key of await this.store.keys(this.prefix))await this.store.remove(key);this.downloaded.clear();}
  close() { this.closed = true; this.controller.abort(); this.decoded.clear(); }
  remember(index, data) {
    if (this.closed) return;
    this.decoded.delete(index); this.decoded.set(index, data);
    while (this.decoded.size > DECODED_LIMIT) this.decoded.delete(this.decoded.keys().next().value);
  }
  async decode(index, blob) {
    const bytes = await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
    if (bytes.byteLength !== this.frameCount(index) * 90000 * 4) throw Error('Incomplete shared frame data. Retry the download.');
    return new Float32Array(bytes);
  }
  async download(index) {
    if (this.inflight.has(index)) return this.inflight.get(index);
    const task = (async () => {
      await this.ready;
      if (this.closed) throw new DOMException('Orbit changed', 'AbortError');
      if (this.downloaded.has(index)) {
        const saved = await this.store.get(this.key(index));
        if (saved) return saved;
        this.downloaded.delete(index); this.report();
      }
      const path = `/api/frames?id=${encodeURIComponent(this.id)}&start=${index*this.meta.chunkFrames}&gzip=1`;
      const response = await fetch(path, { signal: this.controller.signal });
      if (!response.ok) throw Error(`Download interrupted (${response.status}).`);
      const blob = await response.blob();
      const data = await this.decode(index, blob);
      if (this.closed) throw new DOMException('Orbit changed', 'AbortError');
      // Keep demand and nearby frames hot; background downloads do not evict them.
      if (Math.abs(index - this.current) <= 3) this.remember(index, data);
      try {
        await this.store.put(this.key(index), blob);
        if (!this.closed) this.downloaded.add(index);
      } catch(e) {
        this.error = e.name === 'QuotaExceededError' ? 'Browser storage is full. Free some site storage, then retry.' : e.message;
      }
      this.report();
      return blob;
    })();
    this.inflight.set(index, task);
    try { return await task; } finally { this.inflight.delete(index); }
  }
  async frame(index) {
    const start = performance.now(), offset = index - this.meta.firstFrame;
    const n = Math.floor(offset / this.meta.chunkFrames);
    this.current = n;
    let data = this.decoded.get(n);
    if (!data) {
      const blob = await this.download(n);
      data = this.decoded.get(n) || await this.decode(n, blob);
    }
    if (this.closed) throw new DOMException('Orbit changed', 'AbortError');
    this.remember(n, data); this.preload();
    const pos = this.meta.positions[offset] || [null, null];
    return { index, values: data.subarray((offset % this.meta.chunkFrames) * 90000, (offset % this.meta.chunkFrames + 1) * 90000), latitude: pos[0], longitude: pos[1], ms: performance.now() - start };
  }
  async preload() {
    if (this.running || this.closed || this.error) return;
    this.running = true;
    try {
      await this.ready;
      while (!this.closed && !this.error) {
        let next = -1;
        // Prioritize the current frame and upcoming frames, then wrap to earlier gaps.
        for (let i = 0; i < this.total; i++) {
          const n = (this.current + i) % this.total;
          if (!this.downloaded.has(n) && !this.inflight.has(n)) { next = n; break; }
        }
        if (next < 0) break;
        await this.download(next);
        // Yield to controls and painting between chunks.
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    } catch(e) { if (!this.closed) this.error = e.message; }
    finally { this.running = false; this.report(); }
  }
  retry() { this.error = ''; this.report(); this.preload(); }
}
