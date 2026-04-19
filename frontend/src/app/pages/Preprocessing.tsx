import { useState } from "react";
import { Upload, Image as ImageIcon } from "lucide-react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

const rgbHistogramData = [
  { bin: "0-25", red: 120, green: 80, blue: 60 },
  { bin: "26-50", red: 180, green: 140, blue: 100 },
  { bin: "51-75", red: 240, green: 200, blue: 160 },
  { bin: "76-100", red: 300, green: 280, blue: 220 },
  { bin: "101-125", red: 320, green: 340, blue: 280 },
  { bin: "126-150", red: 280, green: 320, blue: 300 },
  { bin: "151-175", red: 220, green: 280, blue: 260 },
  { bin: "176-200", red: 180, green: 240, blue: 220 },
  { bin: "201-225", red: 140, green: 180, blue: 160 },
  { bin: "226-255", red: 100, green: 120, blue: 100 },
];

export function Preprocessing() {
  const [imageUploaded, setImageUploaded] = useState(false);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setImageUploaded(true);
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      <div>
        <h1 className="text-3xl font-bold mb-2">Preprocessing</h1>
        <p className="text-gray-400">Image enhancement and preparation</p>
      </div>

      {/* Upload Zone */}
      <div
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onClick={() => setImageUploaded(true)}
        className="bg-[#1a1d27] rounded-xl p-12 border-2 border-dashed border-gray-700 hover:border-emerald-500/50 transition-all cursor-pointer"
      >
        <div className="flex flex-col items-center justify-center text-center">
          <div className="w-16 h-16 bg-emerald-500/10 rounded-full flex items-center justify-center mb-4">
            <Upload className="w-8 h-8 text-emerald-500" />
          </div>
          <h3 className="text-lg font-semibold mb-2">Upload Plant Image</h3>
          <p className="text-sm text-gray-400 mb-4">Drag and drop or click to select</p>
          <p className="text-xs text-gray-500">Supports JPG, PNG (Max 10MB)</p>
        </div>
      </div>

      {imageUploaded && (
        <>
          {/* Processed Images Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {["Original", "Grayscale", "HSV", "Denoised"].map((label, index) => (
              <div key={label} className="bg-[#1a1d27] rounded-xl p-4 border border-gray-800 hover:border-emerald-500/30 transition-all">
                <h3 className="text-sm font-semibold mb-3 text-gray-300">{label}</h3>
                <div className="aspect-square bg-gray-900 rounded-lg flex items-center justify-center mb-3 overflow-hidden">
                  <div className={`w-full h-full ${
                    index === 0 ? "bg-gradient-to-br from-green-900 to-emerald-700" :
                    index === 1 ? "bg-gradient-to-br from-gray-700 to-gray-900" :
                    index === 2 ? "bg-gradient-to-br from-emerald-600 to-yellow-600" :
                    "bg-gradient-to-br from-green-800 to-emerald-600"
                  } flex items-center justify-center`}>
                    <ImageIcon className="w-12 h-12 text-white/20" />
                  </div>
                </div>
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-400">Quality</span>
                  <span className="text-xs font-semibold text-emerald-500">
                    {index === 0 ? "100%" : index === 1 ? "92%" : index === 2 ? "95%" : "98%"}
                  </span>
                </div>
              </div>
            ))}
          </div>

          {/* RGB Histograms */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-4">Original RGB Histogram</h2>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={rgbHistogramData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="bin" stroke="#9ca3af" fontSize={12} />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1d27",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Bar dataKey="red" fill="#ef4444" isAnimationActive={false} />
                  <Bar dataKey="green" fill="#10b981" isAnimationActive={false} />
                  <Bar dataKey="blue" fill="#3b82f6" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>

            <div className="bg-[#1a1d27] rounded-xl p-6 border border-gray-800">
              <h2 className="text-xl font-semibold mb-4">Processed RGB Histogram</h2>
              <ResponsiveContainer width="100%" height={250}>
                <BarChart data={rgbHistogramData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis dataKey="bin" stroke="#9ca3af" fontSize={12} />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#1a1d27",
                      border: "1px solid #374151",
                      borderRadius: "8px",
                    }}
                  />
                  <Legend />
                  <Bar dataKey="red" fill="#ef4444" isAnimationActive={false} />
                  <Bar dataKey="green" fill="#10b981" isAnimationActive={false} />
                  <Bar dataKey="blue" fill="#3b82f6" isAnimationActive={false} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
