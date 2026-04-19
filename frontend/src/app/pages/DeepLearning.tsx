import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";
import { Layers, ArrowRight } from "lucide-react";

const trainingData = [
  { epoch: 1, trainAcc: 65, valAcc: 62, trainLoss: 0.95, valLoss: 1.02 },
  { epoch: 5, trainAcc: 78, valAcc: 75, trainLoss: 0.68, valLoss: 0.74 },
  { epoch: 10, trainAcc: 85, valAcc: 82, trainLoss: 0.48, valLoss: 0.56 },
  { epoch: 15, trainAcc: 91, valAcc: 88, trainLoss: 0.32, valLoss: 0.41 },
  { epoch: 20, trainAcc: 94, valAcc: 91, trainLoss: 0.22, valLoss: 0.34 },
  { epoch: 25, trainAcc: 96, valAcc: 94, trainLoss: 0.15, valLoss: 0.28 },
  { epoch: 30, trainAcc: 98, valAcc: 96, trainLoss: 0.09, valLoss: 0.22 },
  { epoch: 35, trainAcc: 99, valAcc: 97, trainLoss: 0.05, valLoss: 0.18 },
  { epoch: 40, trainAcc: 99.5, valAcc: 97.8, trainLoss: 0.03, valLoss: 0.15 },
];

const layers = [
  { id: 1, name: "Input", size: "224×224×3" },
  { id: 2, name: "Conv2D", size: "64 filters" },
  { id: 3, name: "MaxPool", size: "2×2" },
  { id: 4, name: "Conv2D", size: "128 filters" },
  { id: 5, name: "MaxPool", size: "2×2" },
  { id: 6, name: "Dense", size: "256 units" },
  { id: 7, name: "Output", size: "5 classes" },
];

const comparison = [
  { model: "Random Forest (ML)", accuracy: 94.2, precision: 93.8, recall: 92.5, f1: 93.1, time: "2.3s" },
  { model: "CNN (DL)", accuracy: 97.8, precision: 97.5, recall: 97.2, f1: 97.3, time: "145s" },
];

export function DeepLearning() {
  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Deep Learning</h1>
        <p className="text-gray-400">CNN-based disease classification</p>
      </div>

      {/* CNN Architecture */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-6">CNN Architecture</h2>
        <div className="flex items-center justify-between overflow-x-auto pb-4">
          {layers.map((layer, index) => (
            <div key={layer.id} className="flex items-center">
              <div className="flex flex-col items-center min-w-[120px]">
                <div className="w-20 h-20 bg-gradient-to-br from-emerald-500/20 to-emerald-500/5 rounded-lg flex items-center justify-center border border-emerald-500/30 hover:border-emerald-500 transition-all">
                  <Layers className="w-8 h-8 text-emerald-500" />
                </div>
                <p className="mt-3 text-sm font-semibold">{layer.name}</p>
                <p className="mt-1 text-xs text-gray-400">{layer.size}</p>
              </div>
              {index < layers.length - 1 && (
                <ArrowRight className="w-6 h-6 text-gray-600 mx-2 flex-shrink-0" />
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Training Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Training & Validation Accuracy</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9ca3af" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
              <YAxis stroke="#9ca3af" label={{ value: 'Accuracy (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1d27",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="trainAcc"
                stroke="#10b981"
                strokeWidth={2}
                name="Training"
                dot={{ fill: "#10b981", r: 4 }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="valAcc"
                stroke="#f59e0b"
                strokeWidth={2}
                name="Validation"
                dot={{ fill: "#f59e0b", r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Training & Validation Loss</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={trainingData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9ca3af" label={{ value: 'Epoch', position: 'insideBottom', offset: -5 }} />
              <YAxis stroke="#9ca3af" label={{ value: 'Loss', angle: -90, position: 'insideLeft' }} />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1d27",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="trainLoss"
                stroke="#f43f5e"
                strokeWidth={2}
                name="Training"
                dot={{ fill: "#f43f5e", r: 4 }}
                isAnimationActive={false}
              />
              <Line
                type="monotone"
                dataKey="valLoss"
                stroke="#f59e0b"
                strokeWidth={2}
                name="Validation"
                dot={{ fill: "#f59e0b", r: 4 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model Comparison */}
      <div className="bg-[#1a1d27] rounded-xl border border-gray-800 overflow-hidden">
        <div className="p-6 border-b border-gray-800">
          <h2 className="text-xl font-semibold">ML vs DL Comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900/50">
              <tr>
                <th className="text-left px-6 py-3 text-sm font-semibold text-gray-400">Model</th>
                <th className="text-center px-6 py-3 text-sm font-semibold text-gray-400">Accuracy</th>
                <th className="text-center px-6 py-3 text-sm font-semibold text-gray-400">Precision</th>
                <th className="text-center px-6 py-3 text-sm font-semibold text-gray-400">Recall</th>
                <th className="text-center px-6 py-3 text-sm font-semibold text-gray-400">F1-Score</th>
                <th className="text-center px-6 py-3 text-sm font-semibold text-gray-400">Training Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {comparison.map((row, index) => (
                <tr key={index} className="hover:bg-gray-900/30 transition-colors">
                  <td className="px-6 py-4 text-sm font-semibold">{row.model}</td>
                  <td className="px-6 py-4 text-center">
                    <span className={`px-3 py-1 rounded-full text-sm font-semibold ${
                      index === 1 ? "bg-emerald-500/10 text-emerald-500" : "bg-amber-500/10 text-amber-500"
                    }`}>
                      {row.accuracy}%
                    </span>
                  </td>
                  <td className="px-6 py-4 text-center text-sm font-mono">{row.precision}%</td>
                  <td className="px-6 py-4 text-center text-sm font-mono">{row.recall}%</td>
                  <td className="px-6 py-4 text-center text-sm font-mono">{row.f1}%</td>
                  <td className="px-6 py-4 text-center text-sm text-gray-400">{row.time}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Training Summary */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Best Validation Accuracy</p>
          <h3 className="text-3xl font-bold text-emerald-500">97.8%</h3>
          <p className="text-xs text-gray-500 mt-2">Epoch 40</p>
        </div>
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Final Loss</p>
          <h3 className="text-3xl font-bold text-rose-500">0.15</h3>
          <p className="text-xs text-gray-500 mt-2">Validation</p>
        </div>
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Total Parameters</p>
          <h3 className="text-3xl font-bold text-blue-500">2.4M</h3>
          <p className="text-xs text-gray-500 mt-2">Trainable</p>
        </div>
      </div>
    </div>
  );
}
