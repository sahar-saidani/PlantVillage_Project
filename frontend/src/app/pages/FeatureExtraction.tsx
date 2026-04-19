import { Copy, Check } from "lucide-react";
import { useState } from "react";
import { RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, ResponsiveContainer, Tooltip } from "recharts";

const features = [
  { category: "Color Histogram", name: "Red Mean", value: 142.5 },
  { category: "Color Histogram", name: "Green Mean", value: 168.3 },
  { category: "Color Histogram", name: "Blue Mean", value: 98.7 },
  { category: "GLCM", name: "Contrast", value: 0.823 },
  { category: "GLCM", name: "Energy", value: 0.651 },
  { category: "GLCM", name: "Homogeneity", value: 0.742 },
  { category: "GLCM", name: "Correlation", value: 0.889 },
  { category: "Shape", name: "Area", value: 45820 },
  { category: "Shape", name: "Perimeter", value: 892.4 },
  { category: "Shape", name: "Circularity", value: 0.724 },
  { category: "Shape", name: "Eccentricity", value: 0.412 },
];

const radarData = [
  { feature: "Color", value: 85 },
  { feature: "Texture", value: 78 },
  { feature: "Contrast", value: 82 },
  { feature: "Energy", value: 65 },
  { feature: "Shape", value: 72 },
  { feature: "Homogeneity", value: 74 },
];

export function FeatureExtraction() {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    const vector = features.map(f => f.value).join(", ");
    navigator.clipboard.writeText(`[${vector}]`);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Feature Extraction</h1>
        <p className="text-gray-400">Extract quantitative features from processed images</p>
      </div>

      {/* Feature Table */}
      <div className="bg-[#1a1d27] rounded-xl border border-gray-800 overflow-hidden">
        <div className="p-6 border-b border-gray-800">
          <h2 className="text-xl font-semibold">Extracted Features</h2>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-900/50">
              <tr>
                <th className="text-left px-6 py-3 text-sm font-semibold text-gray-400">Category</th>
                <th className="text-left px-6 py-3 text-sm font-semibold text-gray-400">Feature Name</th>
                <th className="text-right px-6 py-3 text-sm font-semibold text-gray-400">Value</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {features.map((feature, index) => (
                <tr key={index} className="hover:bg-gray-900/30 transition-colors">
                  <td className="px-6 py-4 text-sm text-emerald-500">{feature.category}</td>
                  <td className="px-6 py-4 text-sm">{feature.name}</td>
                  <td className="px-6 py-4 text-sm text-right font-mono">{feature.value.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Feature Vector Visualization */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <h2 className="text-xl font-semibold mb-4">Feature Vector Radar</h2>
          <ResponsiveContainer width="100%" height={350}>
            <RadarChart data={radarData}>
              <PolarGrid stroke="#374151" />
              <PolarAngleAxis dataKey="feature" stroke="#9ca3af" />
              <PolarRadiusAxis stroke="#9ca3af" />
              <Tooltip
                contentStyle={{
                  backgroundColor: "#1a1d27",
                  border: "1px solid #374151",
                  borderRadius: "8px",
                }}
              />
              <Radar
                name="Feature Values"
                dataKey="value"
                stroke="#10b981"
                fill="#10b981"
                fillOpacity={0.3}
                isAnimationActive={false}
              />
            </RadarChart>
          </ResponsiveContainer>
        </div>

        <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Feature Vector</h2>
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 px-4 py-2 bg-emerald-500 hover:bg-emerald-600 text-white rounded-lg transition-all"
            >
              {copied ? <Check className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
              {copied ? "Copied!" : "Copy Vector"}
            </button>
          </div>
          <div className="bg-gray-900 rounded-lg p-4 font-mono text-sm overflow-x-auto">
            <pre className="text-gray-300 whitespace-pre-wrap break-all">
              {`[\n  ${features.map(f => f.value.toFixed(3)).join(",\n  ")}\n]`}
            </pre>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-gray-400 mb-1">Dimension</p>
              <p className="text-lg font-semibold text-emerald-500">{features.length}D</p>
            </div>
            <div>
              <p className="text-xs text-gray-400 mb-1">Normalization</p>
              <p className="text-lg font-semibold text-emerald-500">Min-Max</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
