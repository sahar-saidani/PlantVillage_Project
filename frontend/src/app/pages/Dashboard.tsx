import { ImageIcon, Layers, TrendingUp, Zap, ArrowRight } from "lucide-react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { StatCard } from "../components/StatCard";
import { useProjectData } from "../lib/use-project-data";

const pipelineSteps = [
  { id: 1, name: "Dataset", icon: ImageIcon },
  { id: 2, name: "Preprocessing", icon: ImageIcon },
  { id: 3, name: "Segmentation", icon: Layers },
  { id: 4, name: "Features", icon: Zap },
  { id: 5, name: "ML", icon: TrendingUp },
  { id: 6, name: "DL", icon: TrendingUp },
];

export function Dashboard() {
  const { data, loading } = useProjectData();

  if (loading || !data?.datasetSummary || !data?.comparisonSummary) {
    return <div className="text-gray-400">Loading project metrics...</div>;
  }

  const supportByClass = Object.entries(data.svmMetrics?.per_class ?? {}).map(([name, metrics]) => ({
    name: name.replace("Tomato_", "").replaceAll("_", " "),
    count: metrics.support,
  }));

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-gray-400">Real metrics exported from the PlantVillage pipeline</p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Images"
          value={data.datasetSummary.num_images.toLocaleString()}
          icon={ImageIcon}
          trend={`${data.datasetSummary.classes.length} tomato classes`}
          color="emerald"
        />
        <StatCard
          title="Best Segmentation"
          value={(data.segmentationBenchmark?.best_method ?? "n/a").replaceAll("_", " ")}
          icon={Layers}
          trend={`${data.segmentationBenchmark?.num_images_evaluated ?? 0} train images benchmarked`}
          color="blue"
        />
        <StatCard
          title="Best Accuracy (ML)"
          value={`${((data.comparisonSummary.classical.test_accuracy ?? 0) * 100).toFixed(2)}%`}
          icon={TrendingUp}
          trend={data.comparisonSummary.classical.best_model}
          color="amber"
        />
        <StatCard
          title="Best Accuracy (DL)"
          value={`${(data.comparisonSummary.deep_learning.test_accuracy * 100).toFixed(2)}%`}
          icon={Zap}
          trend={data.comparisonSummary.deep_learning.model}
          color="rose"
        />
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Test Set Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={supportByClass}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis dataKey="name" stroke="#9ca3af" />
            <YAxis stroke="#9ca3af" />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1a1d27",
                border: "1px solid #374151",
                borderRadius: "8px",
              }}
            />
            <Bar dataKey="count" fill="#10b981" radius={[8, 8, 0, 0]} isAnimationActive={false} />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-6">Processing Pipeline</h2>
        <div className="flex items-center justify-between">
          {pipelineSteps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 bg-emerald-500/10 rounded-xl flex items-center justify-center border border-emerald-500/20">
                  <step.icon className="w-8 h-8 text-emerald-500" />
                </div>
                <p className="mt-2 text-sm text-gray-400">{step.name}</p>
              </div>
              {index < pipelineSteps.length - 1 && <ArrowRight className="w-8 h-8 text-gray-600 mx-4" />}
            </div>
          ))}
        </div>
      </div>

      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Comparison Summary</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm text-gray-300">
          <div className="rounded-lg bg-gray-900/50 p-4">
            <p className="text-gray-400 mb-1">Best overall</p>
            <p className="font-semibold text-emerald-500">{data.comparisonSummary.best_overall}</p>
          </div>
          <div className="rounded-lg bg-gray-900/50 p-4">
            <p className="text-gray-400 mb-1">DL minus ML</p>
            <p className="font-semibold">
              {data.comparisonSummary.delta_accuracy_deep_minus_classical === null
                ? "n/a"
                : `${(data.comparisonSummary.delta_accuracy_deep_minus_classical * 100).toFixed(2)} pts`}
            </p>
          </div>
          <div className="rounded-lg bg-gray-900/50 p-4">
            <p className="text-gray-400 mb-1">Advanced segmentation</p>
            <p className="font-semibold">
              {data.comparisonSummary.modern_pipeline?.advanced_segmentation?.best_method?.replaceAll("_", " ") ??
                "n/a"}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
