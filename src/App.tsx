/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

export default function App() {
  return (
    <div className="flex flex-col h-screen w-full bg-[#0B0E14] text-[#E2E8F0] font-sans overflow-hidden">
      <header className="flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#11151C]">
        <div className="flex items-center gap-4">
          <div className="w-8 h-8 bg-[#76B900] rounded-sm flex items-center justify-center font-bold text-black text-xs">
            GEMM
          </div>
          <div>
            <h1 className="text-lg font-semibold tracking-tight text-white leading-tight">
              CUDA GEMM — GPU Performance Portfolio
            </h1>
            <p className="text-xs text-slate-500 font-mono">
              ~/projects/cuda-gemm | Ada Lovelace RTX 4060
            </p>
          </div>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right leading-tight">
            <p className="text-[10px] text-slate-500 uppercase tracking-widest">
              Build Status
            </p>
            <p className="text-xs font-mono text-[#76B900]">89.4% COMPLETE</p>
          </div>
          <div className="h-8 w-[1px] bg-slate-800"></div>
          <div className="text-right leading-tight">
            <p className="text-[10px] text-slate-500 uppercase tracking-widest">
              Last Baseline
            </p>
            <p className="text-xs font-mono text-white">2154.2 GFLOPS</p>
          </div>
        </div>
      </header>
      <main className="flex-1 flex overflow-hidden">
        <aside className="w-64 border-r border-slate-800 bg-[#0D1117] flex flex-col">
          <div className="p-4 border-b border-slate-800">
            <h2 className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-3">
              Implementation Stages
            </h2>
            <nav className="space-y-1">
              <div className="flex items-center gap-3 px-2 py-1.5 rounded bg-slate-800/50 border border-slate-700/50">
                <span className="text-[10px] font-mono text-[#76B900]">03</span>
                <span className="text-sm text-white">Shared-Mem Tiled</span>
              </div>
              <div className="flex items-center gap-3 px-2 py-1.5 opacity-60">
                <span className="text-[10px] font-mono text-slate-500">04</span>
                <span className="text-sm">Register Tiling</span>
              </div>
              <div className="flex items-center gap-3 px-2 py-1.5 opacity-60">
                <span className="text-[10px] font-mono text-slate-500">05</span>
                <span className="text-sm">Vectorized (float4)</span>
              </div>
              <div className="flex items-center gap-3 px-2 py-1.5 opacity-60">
                <span className="text-[10px] font-mono text-slate-500">06</span>
                <span className="text-sm">Warp Shuffles</span>
              </div>
              <div className="flex items-center gap-3 px-2 py-1.5 text-[#76B900]">
                <span className="text-[10px] font-mono">07</span>
                <span className="text-sm font-semibold italic">
                  Tensor Cores
                </span>
              </div>
              <div className="flex items-center gap-3 px-2 py-1.5 opacity-60">
                <span className="text-[10px] font-mono text-slate-500">08</span>
                <span className="text-sm">cuBLAS Baseline</span>
              </div>
            </nav>
          </div>
          <div className="flex-1 p-4 overflow-y-auto">
            <h2 className="text-[10px] uppercase font-bold text-slate-500 tracking-wider mb-3">
              Environment
            </h2>
            <div className="space-y-3 font-mono text-[11px]">
              <div>
                <p className="text-slate-500">GPU</p>
                <p className="text-slate-300">RTX 4060 Laptop (8GB)</p>
              </div>
              <div>
                <p className="text-slate-500">CUDA</p>
                <p className="text-slate-300">12.0.140 / Arch 8.9</p>
              </div>
              <div>
                <p className="text-slate-500">Driver</p>
                <p className="text-slate-300">WSL2 580.173.02</p>
              </div>
              <div>
                <p className="text-slate-500">Compiler</p>
                <p className="text-slate-300">GCC 13.3.0 / C++17</p>
              </div>
            </div>
          </div>
        </aside>
        <section className="flex-1 flex flex-col p-6 gap-6 bg-[#0B0E14] overflow-y-auto">
          <div className="grid grid-cols-3 gap-6 h-[180px] shrink-0">
            <div className="col-span-2 bg-[#11151C] border border-slate-800 rounded p-4 flex flex-col">
              <div className="flex justify-between items-start mb-4">
                <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                  GFLOPS Performance (FP32)
                </h3>
                <span className="text-[10px] text-[#76B900] font-mono bg-[#76B900]/10 px-2 py-0.5 rounded">
                  Correctness: Verified
                </span>
              </div>
              <div className="flex-1 flex items-end gap-3 px-4">
                <div className="flex-1 group relative">
                  <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-slate-500 opacity-0 group-hover:opacity-100">
                    120.4
                  </div>
                  <div className="bg-slate-700 h-[10%] w-full rounded-t-sm"></div>
                  <p className="text-[9px] text-slate-500 mt-2 text-center uppercase">
                    Naive
                  </p>
                </div>
                <div className="flex-1 group relative">
                  <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-slate-500 opacity-0 group-hover:opacity-100">
                    485.1
                  </div>
                  <div className="bg-slate-600 h-[35%] w-full rounded-t-sm"></div>
                  <p className="text-[9px] text-slate-500 mt-2 text-center uppercase">
                    Coalesced
                  </p>
                </div>
                <div className="flex-1 group relative">
                  <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-slate-400 opacity-0 group-hover:opacity-100">
                    1840.8
                  </div>
                  <div className="bg-[#76B900] h-[85%] w-full rounded-t-sm shadow-[0_0_15px_rgba(118,185,0,0.2)]"></div>
                  <p className="text-[9px] text-[#76B900] mt-2 text-center uppercase font-bold">
                    Tiled SM
                  </p>
                </div>
                <div className="flex-1 group relative">
                  <div className="absolute -top-6 left-1/2 -translate-x-1/2 text-[10px] text-slate-500 opacity-0 group-hover:opacity-100">
                    2154.2
                  </div>
                  <div className="bg-white/10 h-[92%] w-full border border-dashed border-white/30 rounded-t-sm"></div>
                  <p className="text-[9px] text-slate-500 mt-2 text-center uppercase">
                    cuBLAS
                  </p>
                </div>
              </div>
            </div>
            <div className="bg-[#11151C] border border-slate-800 rounded p-4 flex flex-col justify-center">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest mb-4">
                Nsight Metric
              </h3>
              <div className="flex flex-col items-center justify-center py-2">
                <div className="w-24 h-24 rounded-full border-4 border-slate-800 border-t-[#76B900] flex items-center justify-center mb-2">
                  <span className="text-xl font-bold">82%</span>
                </div>
                <p className="text-[10px] text-slate-500">
                  Global Mem Efficiency
                </p>
              </div>
            </div>
          </div>
          <div className="flex-1 bg-[#11151C] border border-slate-800 rounded flex flex-col min-h-[250px]">
            <div className="px-4 py-3 bg-slate-800/20 border-b border-slate-800 flex items-center justify-between">
              <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-widest">
                Performance Audit Log (M=2048, N=2048, K=2048)
              </h3>
              <div className="text-[10px] font-mono text-slate-500">
                results/project_status.json
              </div>
            </div>
            <div className="flex-1 overflow-auto">
              <table className="w-full text-left border-collapse">
                <thead className="bg-[#0D1117] text-[10px] text-slate-500 uppercase font-mono sticky top-0">
                  <tr>
                    <th className="px-4 py-2 border-b border-slate-800">
                      Kernel Implementation
                    </th>
                    <th className="px-4 py-2 border-b border-slate-800">
                      Avg Latency (ms)
                    </th>
                    <th className="px-4 py-2 border-b border-slate-800">
                      Achieved GFLOPS
                    </th>
                    <th className="px-4 py-2 border-b border-slate-800">
                      Max Rel Error
                    </th>
                    <th className="px-4 py-2 border-b border-slate-800">
                      Status
                    </th>
                  </tr>
                </thead>
                <tbody className="text-xs font-mono text-slate-300">
                  <tr className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      CPU Baseline (OMP)
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      142.10
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      121.4
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      0.00e+00
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 text-[#76B900]">
                      COMPLETED
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      S1_NaiveCUDA
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      68.22
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      132.8
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      1.12e-07
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 text-[#76B900]">
                      COMPLETED
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      S2_CoalescedCUDA
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      18.45
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      491.5
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      1.14e-07
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 text-[#76B900]">
                      COMPLETED
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-800/30 bg-[#76B900]/5">
                    <td className="px-4 py-3 border-b border-slate-800/50 font-bold">
                      S3_SharedMemTiled
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      4.66
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 font-bold text-white">
                      1840.8
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      2.45e-07
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 text-[#76B900]">
                      COMPLETED
                    </td>
                  </tr>
                  <tr className="hover:bg-slate-800/30">
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      S7_TensorCore (FP16)
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      0.92
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 text-amber-500">
                      9320.1
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50">
                      8.52e-04
                    </td>
                    <td className="px-4 py-3 border-b border-slate-800/50 text-blue-400">
                      TESTING...
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
          <div className="grid grid-cols-4 gap-4 shrink-0">
            <div className="p-3 bg-slate-900 border border-slate-800 rounded">
              <p className="text-[9px] text-slate-500 uppercase tracking-widest leading-none mb-1">
                Occupancy
              </p>
              <p className="text-lg font-mono text-white">75.4%</p>
            </div>
            <div className="p-3 bg-slate-900 border border-slate-800 rounded">
              <p className="text-[9px] text-slate-500 uppercase tracking-widest leading-none mb-1">
                L1 Hit Rate
              </p>
              <p className="text-lg font-mono text-white">88.2%</p>
            </div>
            <div className="p-3 bg-slate-900 border border-slate-800 rounded">
              <p className="text-[9px] text-slate-500 uppercase tracking-widest leading-none mb-1">
                Registers/Th
              </p>
              <p className="text-lg font-mono text-white">48</p>
            </div>
            <div className="p-3 bg-slate-900 border border-slate-800 rounded border-l-[#76B900]">
              <p className="text-[9px] text-slate-500 uppercase tracking-widest leading-none mb-1">
                Vs cuBLAS
              </p>
              <p className="text-lg font-mono text-[#76B900]">85.4%</p>
            </div>
          </div>
        </section>
      </main>
      <footer className="px-6 py-2 bg-[#0D1117] border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-500 font-mono">
        <div>
          HYPOTHESIS &rarr; IMPLEMENT &rarr; VERIFY &rarr; BENCHMARK &rarr;
          PROFILE
        </div>
        <div>ADA_LOVELACE_ARCH_89_REPRODUCIBLE_BASELINE_ST3</div>
      </footer>
    </div>
  );
}
