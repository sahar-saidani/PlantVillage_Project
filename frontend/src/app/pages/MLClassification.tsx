import { useState } from "react";
import { CompactImageImport } from "../components/CompactImageImport";
import { useProjectData } from "../lib/use-project-data";
import { useLivePipeline } from "../lib/use-live-pipeline";

const MODEL_LABELS: Record<string, string> = {
  svm: "SVM",
  random_forest: "Random Forest",
};

export function MLClassification() {
  const { data, loading } = useProjectData();
  const [selectedModel, setSelectedModel] = useState<"svm" | "random_forest">("svm");
  const live = useLivePipeline();

  if (loading || !data?.svmMetrics || !data?.randomForestMetrics || !data?.comparisonSummary) {
    return <div className="text-gray-400">Loading classical ML metrics...</div>;
  }

  const metrics = selectedModel === "svm" ? data.svmMetrics : data.randomForestMetrics;
  const confusionImage =
    selectedModel === "svm" ? data.confusionImages.svm : data.confusionImages.randomForest;
  const macroPrecision =
    Object.values(metrics.per_class).reduce((sum, item) => sum + item.precision, 0) /
    Object.keys(metrics.per_class).length;
  const macroRecall =
    Object.values(metrics.per_class).reduce((sum, item) => sum + item.recall, 0) /
    Object.keys(metrics.per_class).length;

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Machine Learning Classification</h1>
        <p className="text-gray-400">Real exported metrics for the classical models.</p>
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Select Model</h2>
        <div className="flex gap-3">
          {(["svm", "random_forest"] as const).map((model) => (
            <button
              key={model}
              onClick={() => setSelectedModel(model)}
              className={`rounded-lg px-4 py-2 transition-all ${
                selectedModel === model
                  ? "bg-emerald-500 text-white"
                  : "bg-gray-900/60 text-gray-300 hover:bg-gray-800"
              }`}
            >
              {MODEL_LABELS[model]}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {[
          { name: "Accuracy", value: metrics.accuracy },
          { name: "Precision", value: macroPrecision },
          { name: "Recall", value: macroRecall },
          { name: "Macro F1", value: metrics.macro_f1 },
        ].map((metric) => (
          <div key={metric.name} className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
            <p className="text-sm text-gray-400 mb-2">{metric.name}</p>
            <h3 className="text-3xl font-bold mb-2">{(metric.value * 100).toFixed(2)}%</h3>
            <div className="h-2 rounded-full bg-gray-800">
              <div
                className="h-2 rounded-full bg-emerald-500"
                style={{ width: `${metric.value * 100}%` }}
              />
            </div>
          </div>
        ))}
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Confusion Matrix</h2>
        <img src={confusionImage} alt={`${selectedModel} confusion matrix`} className="rounded-lg border border-gray-800" />
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Per-Class Metrics</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="text-left text-gray-400">
              <tr>
                <th className="pb-3">Class</th>
                <th className="pb-3">Precision</th>
                <th className="pb-3">Recall</th>
                <th className="pb-3">F1</th>
                <th className="pb-3">Support</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800 text-gray-200">
              {Object.entries(metrics.per_class).map(([label, item]) => (
                <tr key={label}>
                  <td className="py-3">{label.replace("Tomato_", "").replaceAll("_", " ")}</td>
                  <td className="py-3">{(item.precision * 100).toFixed(2)}%</td>
                  <td className="py-3">{(item.recall * 100).toFixed(2)}%</td>
                  <td className="py-3">{(item.f1 * 100).toFixed(2)}%</td>
                  <td className="py-3">{item.support}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Model Selection Outcome</h2>
        <p className="text-sm text-gray-300">
          Best classical model:{" "}
          <span className="font-semibold text-emerald-500">
            {data.comparisonSummary.classical.best_model}
          </span>
        </p>
      </div>

      <div className="space-y-6">
        <CompactImageImport
          loading={live.loading}
          onFileSelect={live.run}
          subtitle="Runs the classical model on the imported image and shows class probabilities."
        />

        {live.error && (
          <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
            {live.error}
          </div>
        )}

        {live.result?.predictions.classical_ml && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-4">Live Classical Prediction</h2>
              <p className="text-3xl font-bold text-emerald-500">
                {live.result.predictions.classical_ml.label.replace("Tomato_", "").replaceAll("_", " ")}
              </p>
              <p className="mt-2 text-sm text-gray-400">
                Feature dimension: {live.result.predictions.classical_ml.feature_dim ?? "n/a"}
              </p>
              {live.previewUrl && (
                <img src={live.previewUrl} alt="Imported leaf" className="mt-4 w-full rounded-lg border border-gray-800" />
              )}
            </div>
            <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-4">Class Probabilities</h2>
              <div className="space-y-3">
                {Object.entries(live.result.predictions.classical_ml.scores ?? {}).map(([label, score]) => (
                  <div key={label}>
                    <div className="mb-1 flex items-center justify-between text-sm">
                      <span>{label.replace("Tomato_", "").replaceAll("_", " ")}</span>
                      <span>{(score * 100).toFixed(2)}%</span>
                    </div>
                    <div className="h-2 rounded-full bg-gray-800">
                      <div className="h-2 rounded-full bg-emerald-500" style={{ width: `${score * 100}%` }} />
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
