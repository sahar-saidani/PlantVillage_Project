import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid, Legend } from "recharts";
import { CompactImageImport } from "../components/CompactImageImport";
import { useProjectData } from "../lib/use-project-data";
import { useLivePipeline } from "../lib/use-live-pipeline";

export function DeepLearning() {
  const { data, loading } = useProjectData();
  const live = useLivePipeline();

  if (loading || !data?.trainingSummary || !data?.comparisonSummary) {
    return <div className="text-gray-400">Loading deep learning metrics...</div>;
  }

  const history = data.trainingSummary.history.map((entry) => ({
    epoch: entry.epoch,
    trainAcc: (entry.train_acc ?? entry.train_accuracy ?? 0) * 100,
    valAcc: (entry.val_acc ?? entry.val_accuracy ?? 0) * 100,
    trainLoss: entry.train_loss ?? 0,
    valLoss: entry.val_loss ?? 0,
  }));

  const deepMetrics = data.deepMetrics;
  const bestClassical = data.comparisonSummary.classical;
  const classicalAccuracy = (bestClassical.test_accuracy ?? 0) * 100;
  const deepAccuracy = data.comparisonSummary.deep_learning.test_accuracy * 100;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Deep Learning</h1>
        <p className="text-gray-400">EfficientNet-B0 pretrained results exported from the project.</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Best Validation Accuracy</p>
          <h3 className="text-3xl font-bold text-emerald-500">
            {(data.trainingSummary.best_val_acc * 100).toFixed(2)}%
          </h3>
          <p className="text-xs text-gray-500 mt-2">{data.trainingSummary.epochs} epochs</p>
        </div>
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Test Accuracy</p>
          <h3 className="text-3xl font-bold text-blue-500">
            {(data.trainingSummary.test_accuracy * 100).toFixed(2)}%
          </h3>
          <p className="text-xs text-gray-500 mt-2">{data.comparisonSummary.deep_learning.model}</p>
        </div>
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <p className="text-sm text-gray-400 mb-1">Macro F1</p>
          <h3 className="text-3xl font-bold text-rose-500">
            {(data.trainingSummary.macro_avg_f1 * 100).toFixed(2)}%
          </h3>
          <p className="text-xs text-gray-500 mt-2">Deep learning evaluation</p>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Training & Validation Accuracy</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1d27",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="trainAcc" stroke="#10b981" strokeWidth={2} isAnimationActive={false} />
              <Line type="monotone" dataKey="valAcc" stroke="#f59e0b" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Training & Validation Loss</h2>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis dataKey="epoch" stroke="#9ca3af" />
              <YAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1d27",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Legend />
              <Line type="monotone" dataKey="trainLoss" stroke="#f43f5e" strokeWidth={2} isAnimationActive={false} />
              <Line type="monotone" dataKey="valLoss" stroke="#38bdf8" strokeWidth={2} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Deep Confusion Matrix</h2>
        <img src={data.confusionImages.deep} alt="Deep confusion matrix" className="rounded-lg border border-gray-800" />
      </div>

      <div className="bg-[#1a1d27] rounded-xl border border-gray-800 overflow-hidden">
        <div className="p-6 border-b border-gray-800">
          <h2 className="text-xl font-semibold">ML vs DL Comparison</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900/50 text-sm text-gray-400">
              <tr>
                <th className="px-6 py-3 text-left">Model</th>
                <th className="px-6 py-3 text-center">Accuracy</th>
                <th className="px-6 py-3 text-center">Macro F1</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800 text-sm text-gray-200">
              <tr>
                <td className="px-6 py-4">{bestClassical.best_model}</td>
                <td className="px-6 py-4 text-center">{classicalAccuracy.toFixed(2)}%</td>
                <td className="px-6 py-4 text-center">
                  {((bestClassical.test_macro_f1 ?? 0) * 100).toFixed(2)}%
                </td>
              </tr>
              <tr className="bg-emerald-500/5">
                <td className="px-6 py-4">{data.comparisonSummary.deep_learning.model}</td>
                <td className="px-6 py-4 text-center text-emerald-500 font-semibold">
                  {deepAccuracy.toFixed(2)}%
                </td>
                <td className="px-6 py-4 text-center text-emerald-500 font-semibold">
                  {(data.comparisonSummary.deep_learning.test_macro_f1 * 100).toFixed(2)}%
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {deepMetrics?.per_class && (
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Per-Class Deep Metrics</h2>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {Object.entries(deepMetrics.per_class).map(([label, item]) => (
              <div key={label} className="rounded-lg bg-gray-900/50 p-4 text-sm text-gray-300">
                <p className="font-semibold text-white mb-2">{label.replace("Tomato_", "").replaceAll("_", " ")}</p>
                <p>Precision: {(item.precision * 100).toFixed(2)}%</p>
                <p>Recall: {(item.recall * 100).toFixed(2)}%</p>
                <p>F1: {(item.f1 * 100).toFixed(2)}%</p>
                <p>Support: {item.support}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="space-y-6">
        <CompactImageImport
          loading={live.loading}
          onFileSelect={live.run}
          subtitle="Runs the deep model on the imported image and shows the live confidence scores."
        />

        {live.error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {live.error}
          </div>
        )}

        {live.result?.predictions.deep_learning && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-4">Live Deep Prediction</h2>
              <p className="text-3xl font-bold text-emerald-500">
                {live.result.predictions.deep_learning.label.replace("Tomato_", "").replaceAll("_", " ")}
              </p>
              {live.previewUrl && (
                <img src={live.previewUrl} alt="Imported leaf" className="mt-4 w-full rounded-lg border border-gray-800" />
              )}
            </div>
            <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-4">Deep Confidence Scores</h2>
              <div className="space-y-3">
                {Object.entries(live.result.predictions.deep_learning.scores ?? {}).map(([label, score]) => (
                  <div key={label}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span>{label.replace("Tomato_", "").replaceAll("_", " ")}</span>
                      <span>{(score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-800">
                      <div className="h-2 rounded-full bg-blue-500" style={{ width: `${score * 100}%` }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
