import { StatCard } from "../components/StatCard";
import { ImageIcon, Layers, TrendingUp, Zap, ArrowRight } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

const classData = [
  { name: "Healthy", count: 450 },
  { name: "Rust", count: 320 },
  { name: "Blight", count: 280 },
  { name: "Mildew", count: 195 },
  { name: "Spot", count: 240 },
];

const pipelineSteps = [
  { id: 1, name: "Dataset", icon: ImageIcon },
  { id: 2, name: "Preprocessing", icon: ImageIcon },
  { id: 3, name: "Segmentation", icon: Layers },
  { id: 4, name: "Features", icon: Zap },
  { id: 5, name: "ML", icon: TrendingUp },
  { id: 6, name: "DL", icon: TrendingUp },
];

export function Dashboard() {
  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Dashboard</h1>
        <p className="text-gray-400">Overview of plant disease detection system</p>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Images"
          value="1,485"
          icon={ImageIcon}
          trend="+12% this week"
          color="emerald"
        />
        <StatCard
          title="Classes"
          value="5"
          icon={Layers}
          trend="Healthy + 4 diseases"
          color="blue"
        />
        <StatCard
          title="Best Accuracy (ML)"
          value="94.2%"
          icon={TrendingUp}
          trend="Random Forest"
          color="amber"
        />
        <StatCard
          title="Best Accuracy (DL)"
          value="97.8%"
          icon={Zap}
          trend="CNN Model"
          color="rose"
        />
      </div>

      {/* Class Distribution Chart */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-4">Class Distribution</h2>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={classData}>
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

      {/* Pipeline Flow */}
      <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
        <h2 className="text-xl font-semibold mb-6">Processing Pipeline</h2>
        <div className="flex items-center justify-between">
          {pipelineSteps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <div className="flex flex-col items-center">
                <div className="w-16 h-16 bg-emerald-500/10 rounded-xl flex items-center justify-center border border-emerald-500/20 hover:bg-emerald-500/20 transition-all">
                  <step.icon className="w-8 h-8 text-emerald-500" />
                </div>
                <p className="mt-2 text-sm text-gray-400">{step.name}</p>
              </div>
              {index < pipelineSteps.length - 1 && (
                <ArrowRight className="w-8 h-8 text-gray-600 mx-4" />
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
