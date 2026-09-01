import { useMemo, useState } from "react";

const kernels = [
  { name: "Naive", gflops: 716.92, speedup: 1.00, note: "Baseline" },
  { name: "Coalesced", gflops: 967.83, speedup: 1.35, note: "+35.0%" },
  { name: "Shared16", gflops: 857.56, speedup: 1.20, note: "+19.6%" },
  { name: "Shared32", gflops: 836.07, speedup: 1.17, note: "+16.6%" },
  { name: "Register", gflops: 1614.97, speedup: 2.25, note: "+125.3%" },
  { name: "Register64", gflops: 2806.70, speedup: 3.91, note: "+291.5%" },
  { name: "Vectorized", gflops: 2671.83, speedup: 3.73, note: "+272.7%" },
  { name: "Warp", gflops: 103.46, speedup: 0.14, note: "-85.6%" },
  { name: "cuBLAS", gflops: 8460.93, speedup: 11.80, note: "Reference" },
];

const profileStats = [
  ["register64", "56", "66.67%", "64.75%"],
  ["WMMA", "40", "100%", "97.75%"],
  ["vectorized", "40", "100%", "92.54%"],
];

export default function App() {
  const [selected, setSelected] = useState("Register64");
  const selectedKernel = useMemo(() => kernels.find((k) => k.name === selected)!, [selected]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/90">
        <div className="mx-auto max-w-7xl px-6 py-5 flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs uppercase tracking-[0.22em] text-slate-500">CUDA GEMM Performance Lab</p>
            <h1 className="mt-1 text-2xl font-semibold tracking-tight">GPU Kernel Performance Dashboard</h1>
            <p className="mt-1 text-sm text-slate-400">NVIDIA GeForce RTX 4060 Laptop GPU · Ada Lovelace · SM 8.9</p>
          </div>
          <div className="grid grid-cols-3 gap-3 text-center text-xs font-mono">
            <Metric label="Experiments" value="589" />
            <Metric label="Raw artifacts" value="589" />
            <Metric label="Profiles" value="15" />
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-8 space-y-8">
        <section className="grid gap-4 lg:grid-cols-4">
          <StatCard title="Best custom FP32" value="2.81 TFLOPS" detail="Register64 · 1024³" />
          <StatCard title="Custom speedup" value="3.91×" detail="vs. naive FP32" />
          <StatCard title="cuBLAS FP32" value="8.46 TFLOPS" detail="1024³ median" />
          <StatCard title="cuBLAS FP16" value="28.84 TFLOPS" detail="2048³ median" />
        </section>

        <section className="grid gap-6 lg:grid-cols-[1.7fr_1fr]">
          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
            <div className="flex flex-wrap items-end justify-between gap-3">
              <div>
                <h2 className="font-semibold">1024³ FP32 benchmark</h2>
                <p className="mt-1 text-sm text-slate-400">Median kernel throughput · 10 warmups · 50 iterations</p>
              </div>
              <select value={selected} onChange={(e) => setSelected(e.target.value)} className="rounded-md border border-slate-700 bg-slate-950 px-3 py-2 text-sm">
                {kernels.map((k) => <option key={k.name}>{k.name}</option>)}
              </select>
            </div>

            <div className="mt-6 space-y-3">
              {kernels.map((kernel) => {
                const width = Math.max(2, Math.min(100, (kernel.gflops / 8460.93) * 100));
                const active = kernel.name === selected;
                return (
                  <button key={kernel.name} onClick={() => setSelected(kernel.name)} className={`w-full text-left ${active ? "rounded-lg bg-slate-800/70 p-2" : "p-2"}`}>
                    <div className="flex items-center justify-between gap-4 text-sm">
                      <span className="w-24 font-medium">{kernel.name}</span>
                      <div className="flex-1 rounded-full bg-slate-800 h-2 overflow-hidden">
                        <div className="h-full rounded-full bg-emerald-500" style={{ width: `${width}%` }} />
                      </div>
                      <span className="w-28 text-right font-mono text-slate-300">{kernel.gflops.toFixed(2)} GFLOPS</span>
                      <span className="w-20 text-right text-slate-500">{kernel.note}</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>

          <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
            <h2 className="font-semibold">Selected kernel</h2>
            <div className="mt-5 rounded-lg bg-slate-950 p-5">
              <p className="text-sm text-slate-500">{selectedKernel.name}</p>
              <p className="mt-1 text-4xl font-semibold tracking-tight">{selectedKernel.gflops.toFixed(2)}</p>
              <p className="text-sm text-slate-400">GFLOPS median</p>
              <div className="mt-5 grid grid-cols-2 gap-3 text-sm">
                <Metric label="Speedup" value={`${selectedKernel.speedup.toFixed(2)}×`} />
                <Metric label="Workload" value="1024³" />
              </div>
            </div>
            <p className="mt-4 text-sm leading-6 text-slate-400">Results are workload-specific measurements. Nsight Compute timing is diagnostic only and is not used to claim speedups.</p>
          </div>
        </section>

        <section className="rounded-xl border border-slate-800 bg-slate-900/70 p-5">
          <div className="flex items-end justify-between gap-4">
            <div>
              <h2 className="font-semibold">Microarchitectural evidence</h2>
              <p className="mt-1 text-sm text-slate-400">Representative Nsight Compute measurements from the preserved profile set.</p>
            </div>
          </div>
          <div className="mt-5 overflow-x-auto">
            <table className="w-full min-w-[620px] text-sm">
              <thead className="text-left text-slate-500">
                <tr>
                  <th className="pb-3">Kernel</th><th className="pb-3">Registers/thread</th><th className="pb-3">Theoretical occupancy</th><th className="pb-3">Achieved occupancy</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/70">
                {profileStats.map(([name, regs, theo, achieved]) => (
                  <tr key={name}>
                    <td className="py-3 font-medium">{name}</td><td className="py-3 font-mono">{regs}</td><td className="py-3 font-mono">{theo}</td><td className="py-3 font-mono">{achieved}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>

        <section className="grid gap-4 md:grid-cols-3">
          <InfoCard title="Tensor Cores" value="4,194,304 HMMA instructions" />
          <InfoCard title="Correctness" value="111 verified PASS records" />
          <InfoCard title="Release" value="EXP-000589 · frozen dataset" />
        </section>
      </main>

      <footer className="border-t border-slate-800 bg-slate-900/60">
        <div className="mx-auto max-w-7xl px-6 py-4 text-xs text-slate-500">CUDA 12.0.140 · GCC 13.3.0 · Ubuntu 24.04 / WSL2 · benchmark data preserved in results/</div>
      </footer>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="rounded-md border border-slate-800 bg-slate-950/80 px-3 py-2"><p className="text-[10px] uppercase tracking-wider text-slate-500">{label}</p><p className="mt-1 font-mono text-sm text-slate-200">{value}</p></div>;
}

function StatCard({ title, value, detail }: { title: string; value: string; detail: string }) {
  return <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5"><p className="text-xs uppercase tracking-wider text-slate-500">{title}</p><p className="mt-2 text-2xl font-semibold">{value}</p><p className="mt-1 text-sm text-slate-400">{detail}</p></div>;
}

function InfoCard({ title, value }: { title: string; value: string }) {
  return <div className="rounded-xl border border-slate-800 bg-slate-900/70 p-5"><p className="text-xs uppercase tracking-wider text-slate-500">{title}</p><p className="mt-2 text-sm leading-6 text-slate-300">{value}</p></div>;
}
